"""Training-only RFUAV recurrence scan; CPU-only and no separation claims."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from drone_rf.data import sha256, DEVELOPMENT_CATEGORIES
from drone_rf.temporal import lag_similarity, local_peak, block_shuffle_rank, power_features
import drone_rf.temporal as temporal_module
import drone_rf.data as data_module


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    started = time.time()
    config = dict(fft_size=1024, bands=64, min_lag=128, max_lag=512,
        block_shuffle_frames=64, shuffle_repeats=63, candidate_min_correlation=.2,
        candidate_max_shuffle_rank=.05, train_only=True, seed=0)
    write(args.output / 'CONFIG.json', config)
    manifest = json.loads(args.manifest.read_text())
    clips = [c for c in manifest['clips'] if c['role'] == 'train_pack']
    if not clips or any(c['category'] not in DEVELOPMENT_CATEGORIES for c in clips):
        raise ValueError('Unexpected development category')
    rows, curves = [], []
    for i, clip in enumerate(clips):
        if clip['samples'] != 2097152 or clip['fs_hz'] != 100000000:
            raise ValueError('Unexpected observation geometry')
        path = Path(clip['cache_path'])
        if sha256(path) != clip['cache_sha256']:
            raise ValueError('Changed cache')
        iq = np.load(path, mmap_mode='r', allow_pickle=False)
        envelope, profile = power_features(iq)
        acf = lag_similarity(envelope, 513)
        spectral = lag_similarity(profile, 513)
        peak = local_peak(acf, 128, 512)
        short_peak = local_peak(acf, 2, 127)
        spectral_peak = local_peak(spectral, 128, 512)
        shuffled = block_shuffle_rank(envelope, peak['correlation'] if peak else 1.,
            128, 512, seed=int(clip['clip_id'][:8], 16))
        candidate = bool(peak and peak['correlation'] >= .2 and shuffled['exceedance_fraction'] <= .05)
        row = dict(clip_id=clip['clip_id'], source_sha256=clip['source_sha256'], pack_id=clip['pack_id'],
            category=clip['category'], duration_ms=1000 * len(iq) / clip['fs_hz'],
            envelope_candidate=candidate,
            envelope_peak_ms=peak['lag_frames'] * .01024 if peak else None,
            envelope_peak_correlation=peak['correlation'] if peak else None,
            short_peak_ms=short_peak['lag_frames'] * .01024 if short_peak else None,
            short_peak_correlation=short_peak['correlation'] if short_peak else None,
            long_to_short_lag_ratio=peak['lag_frames'] / short_peak['lag_frames'] if peak and short_peak else None,
            block_shuffle_exceedance=shuffled['exceedance_fraction'] if peak else None,
            block_shuffle_max_q95=shuffled['control_max_q95'],
            spectral_profile_peak_ms=spectral_peak['lag_frames'] * .01024 if spectral_peak else None,
            spectral_profile_peak_correlation=spectral_peak['correlation'] if spectral_peak else None)
        rows.append(row); curves.append(np.stack([acf, spectral]))
        write(args.output / 'PROGRESS.json', dict(stage='TRAINING_CACHE_SCAN', completed=i + 1,
            total=len(clips), time=time.time(), pid=os.getpid(), gpu_used=False))
    groups = []
    for pack in sorted({r['pack_id'] for r in rows}):
        group = [r for r in rows if r['pack_id'] == pack]
        found = [r for r in group if r['envelope_candidate']]
        periods = [r['envelope_peak_ms'] for r in found]
        groups.append(dict(pack_id=pack, category=group[0]['category'], clips=len(group),
            candidate_clips=len(found), candidate_period_median_ms=float(np.median(periods)) if periods else None,
            candidate_period_q10_q90_ms=np.quantile(periods, [.1, .9]).tolist() if periods else None))
    by_file = collections.defaultdict(list)
    for row in rows:
        by_file[row['source_sha256']].append(row)
    consistent = 0
    for group in by_file.values():
        if len(group) == 3 and all(r['envelope_candidate'] for r in group):
            periods = [r['envelope_peak_ms'] for r in group]
            consistent += (max(periods) - min(periods)) / np.median(periods) <= .05
    with (args.output / 'CLIPS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.save(args.output / 'CORRELATIONS.npy', np.stack(curves), allow_pickle=False)
    result = dict(status='TRAINING_ONLY_TEMPORAL_DIAGNOSTIC_COMPLETE', time=time.time(),
        elapsed_seconds=time.time() - started, training_clips=len(rows), source_files=len(by_file),
        training_packs=len(groups), categories=len({r['category'] for r in rows}),
        candidate_clips=sum(r['envelope_candidate'] for r in rows),
        files_with_three_candidates_and_periods_within_5pct=int(consistent), groups=groups,
        config=config, fft_frame_ms=.01024, examined_lag_ms=[1.31072, 5.24288],
        nominal_model_input_ms=.63872, diagnostic_context_ms=20.97152,
        source_manifest_sha256=sha256(args.manifest), script_sha256=sha256(__file__),
        helper_sha256=sha256(temporal_module.__file__), data_helper_sha256=sha256(data_module.__file__),
        files={n: sha256(args.output / n) for n in ['CLIPS.csv', 'CORRELATIONS.npy', 'CONFIG.json']},
        validation_read=False, heldout_read=False, gpu_used=False, separator_trained=False,
        interpretation='Exploratory power-envelope recurrence with block-shuffle controls. '
        'A candidate is not a decoded hop period, formal significance test, cyclostationary spectrum, '
        'airframe-only transmission proof, or demonstrated source separation benefit.')
    write(args.output / 'COMPLETE.json', result)
    write(args.output / 'PROGRESS.json', dict(stage='COMPLETE', **result))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', default='')
    run(parser.parse_args())
