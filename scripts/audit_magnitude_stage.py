"""Independent CPU replay of one matched legacy magnitude-loss stage."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from summarize_power_trial import check_pair, summarize, read, sha


def run(args):
    os.nice(10)
    os.sched_setaffinity(0, {14, 15})
    if args.output.exists():
        raise FileExistsError('Keep earlier audited report unchanged')
    started = time.time()
    baseline = args.root / 'drff_v103_matched_continuation/no_nmf/run/supervised'
    magnitude = args.root / 'drff_v139_n_local_amplitude_20261006/run/supervised'
    records, selections = check_pair(baseline, magnitude, args.stage)
    verifier_path = args.root / 'drff_v118_d1_wave_validation_20261006/verify118.py'
    if sha(verifier_path) != '434009c84d021b481a50a00624382f8fe0b437e0fa85298f587243b8e7868ac6':
        raise ValueError('Frozen independent verifier changed')
    spec = importlib.util.spec_from_file_location('frozen_verifier', verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    frozen = read(verifier_path.parent / 'FREEZE.json')
    arrays = {}
    source_hashes = {}
    for entry in frozen['source_cache']:
        path = Path(entry['path'])
        if sha(path) != entry['npy_sha256']:
            raise ValueError('Reference cache changed')
        arrays[tuple(entry['key'])] = np.load(path, mmap_mode='r', allow_pickle=False)
        source_hashes[path.name] = entry['npy_sha256']
    recipes = {r['case_id']: r for r in frozen['recipes'] if r['kind'] == 'mixture'}
    max_nmse_error, max_si_error = 0., 0.
    hashes = {}
    cases = []
    for i, folder in enumerate((baseline, magnitude)):
        for row in records[i]:
            path = folder / f'evaluation_{args.stage}' / row['case_id'] / 'output.npy'
            if sha(path) != row['output_sha256']:
                raise ValueError('Saved output changed')
            output = np.load(path, allow_pickle=False).astype(np.complex128)
            _, reference = verifier.inputs(recipes[row['case_id']], arrays)
            reference = reference.astype(np.complex128)
            for k in (0, 1):
                nmse = verifier.power(output[k] - reference[k]) / verifier.power(reference[k])
                max_nmse_error = max(max_nmse_error, abs(nmse - row['nmse'][k]))
                score, status = verifier.si(output[k], reference[k])
                if status != row['si_status'][k]:
                    raise ValueError('SI-SDR status disagrees')
                if status == 'finite':
                    max_si_error = max(max_si_error, abs(score - row['si_db'][k]))
            hashes[f'{i}/{row["case_id"]}'] = row['output_sha256']
    if max_nmse_error > 1e-8 or max_si_error > 1e-8:
        raise ValueError('Saved waveform metrics did not replay')
    scores = [summarize(rows) for rows in records]
    for a, b in zip(*records):
        cases.append(dict(case_id=a['case_id'], baseline_nmse=a['mean_complex_nmse'],
                          magnitude_nmse=b['mean_complex_nmse']))
    result = dict(status='MATCHED_MAGNITUDE_STAGE_WAVEFORMS_AUDITED', time=time.time(),
        epoch_budget=args.stage, updates_per_arm=selections[0]['updates'],
        selected_epochs=[r['selected_epoch'] for r in selections],
        parent_checkpoint_sha256=selections[0]['parent_checkpoint_sha256'],
        baseline=scores[0], magnitude=scores[1],
        relative_nmse_change_percent=100 * (scores[1]['overall_nmse'] / scores[0]['overall_nmse'] - 1),
        cases=cases, saved_outputs_replayed=24, max_nmse_replay_error=max_nmse_error,
        max_si_sdr_replay_error=max_si_error, output_sha256=hashes,
        reference_cache_sha256=source_hashes, verifier_sha256=sha(verifier_path), script_sha256=sha(Path(__file__)),
        elapsed_seconds=time.time() - started, gpu_used=False, independent_confirmation=False,
        dataset='DRFF-R2 D1 repeatedly used development evaluation; 12 synthetic mixtures',
        final_50_epoch_conclusion=args.stage == 50,
        amplitude_metric_audit_included=False, single_source_control_audit_included=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'epoch_budget', 'baseline', 'magnitude',
        'relative_nmse_change_percent', 'max_nmse_replay_error', 'max_si_sdr_replay_error')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=[5, 10, 25, 50], default=25)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
