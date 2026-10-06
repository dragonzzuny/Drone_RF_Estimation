"""Bounded training-only 1/2/3-source input audit; no model training/evaluation."""
import argparse
import collections
import gc
import itertools
import json
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import torch

from drone_rf.context_data import contextual_mixture
from drone_rf.context_model import ContextualSeparator
from drone_rf.data import ScheduledMixtures, sha256
from drone_rf.model import ComplexSeparator
import drone_rf.context_data as context_data
import drone_rf.context_model as context_model
import drone_rf.model as model_module
import drone_rf.temporal as temporal_module
import drone_rf.data as data_module


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    torch.set_num_threads(1)
    started = time.time()
    # Reuse pinned manifest/split and per-cache hash/stat checks. No calls to
    # validation __getitem__; held-out waveforms are outside this manifest.
    library = ScheduledMixtures(args.manifest, args.schedule, 'train_pack', max_open=6)
    groups = collections.defaultdict(list)
    for i, clip in enumerate(library.clips):
        if clip['role'] == 'train_pack':
            groups[clip['category']].append(i)
    if len(groups) != 5:
        raise ValueError('Expected the five audited development categories')
    config = dict(seed=0, fine_window_samples=63872, fine_fft=512, fine_hop=128,
                  context_samples=2097152, context_fft=1024, bands=64, pool_frames=8,
                  source_counts=[1, 2, 3], train_only=True, max_open=6)
    write(args.output / 'CONFIG.json', config)
    rng = np.random.default_rng(0)
    cases = []
    for count in (1, 2, 3):
        levels = [[0.]] if count == 1 else ([[-10., 0.], [0., 0.], [10., 0.]] if count == 2 else
                  [[0., 0., 0.]] + [[level if i == weak else 0. for i in range(3)]
                                    for level in (-10., 10.) for weak in range(3)])
        for categories in itertools.combinations(sorted(groups), count):
            for level in levels:
                indices = [int(rng.choice(groups[category])) for category in categories]
                clips = [library.clips[i] for i in indices]
                if any(c['samples'] != 2097152 or c['fs_hz'] != 100000000 for c in clips):
                    raise ValueError('Unexpected context geometry')
                start = int(rng.integers(0, 2097152 - 63872 + 1))
                phases = rng.uniform(-np.pi, np.pi, count)
                item = contextual_mixture([library._array(i) for i in indices],
                    [c['mean_power'] for c in clips], level, phases, start)
                if item['context_features'].shape != (65, 256) or not np.isfinite(item['context_features']).all():
                    raise ValueError('Invalid observed context features')
                error = float(np.max(np.abs(item['mixture'] - item['references'].sum(0))))
                if error != 0 or item['count_eligible']:
                    raise ValueError('Input replay or count-label policy violated')
                power = np.mean(np.abs(item['references'].astype(np.complex128)) ** 2, axis=1)
                cases.append(dict(construction_count=count, categories=list(categories),
                    clip_ids=[c['clip_id'] for c in clips], pack_ids=[c['pack_id'] for c in clips],
                    center_frequencies_hz=[c['center_hz'] for c in clips], levels_db=level,
                    phases=phases.tolist(), crop_start=start, local_component_power=power.tolist(),
                    mixture_sum_max_error=error, count_eligible=False))
                write(args.output / 'PROGRESS.json', dict(stage='INPUT_AUDIT', completed=len(cases),
                    total=105, time=time.time(), pid=os.getpid(), gpu_used=False))
    models = {}
    for kind in ('plain', 'tcn', 'transformer', 'lstm'):
        torch.manual_seed(0)
        model = ComplexSeparator(3) if kind == 'plain' else ContextualSeparator(kind)
        models[kind] = dict(parameters=sum(p.numel() for p in model.parameters()),
                           untrained=True, max_waveform_sources=3, background_outputs=1)
        if kind != 'plain':
            with torch.no_grad():
                encoded = model.context_encoder(torch.from_numpy(item['context_features'])[None])
            if encoded.shape != (1, 128, 256) or not torch.isfinite(encoded).all():
                raise ValueError('Context encoder failed on actual mixed input')
        del model
        gc.collect()
    write(args.output / 'CASES.json', cases)
    result = dict(status='THREE_SOURCE_CONTEXT_INPUTS_AUDITED_NOT_TRAINED',
        time=time.time(), elapsed_seconds=time.time() - started, cases=len(cases),
        cases_by_count={str(n): sum(c['construction_count'] == n for c in cases) for n in (1, 2, 3)},
        checked_cache_clips=len(library._verified), source_manifest_sha256=sha256(args.manifest),
        config=config, models=models, maximum_mixture_sum_error=max(c['mixture_sum_max_error'] for c in cases),
        context_shape=[65, 256], context_token_us=81.92, context_duration_ms=20.97152,
        fine_window_ms=.63872, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        files={name: sha256(args.output / name) for name in ('CASES.json', 'CONFIG.json')},
        sources={Path(path).name: sha256(path) for path in (__file__, context_data.__file__,
            context_model.__file__, model_module.__file__, temporal_module.__file__, data_module.__file__)},
        validation_read=False, heldout_read=False, gpu_used=False, separator_trained=False,
        count_performance_evaluated=False, waveform_performance_evaluated=False,
        physical_aircraft_count_labels_approved=False,
        interpretation='CPU structure/input checks only. Mixtures use different aircraft-category '
        'recordings rebased to complex baseband, not actual simultaneous flights. Construction '
        'count is not physical-aircraft or local transmission count; no eligible count labels yet.')
    write(args.output / 'COMPLETE.json', result)
    write(args.output / 'PROGRESS.json', dict(stage='COMPLETE', **result))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', default='14,15')
    run(parser.parse_args())
