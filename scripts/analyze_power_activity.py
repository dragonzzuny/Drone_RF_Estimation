"""CPU diagnostic of saved N waveforms using previously fixed training thresholds."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import summarize_power_trial as comparison_module
from summarize_power_trial import check_pair, sha, read


def energy(x):
    x = np.asarray(x, np.complex128)
    return float(np.vdot(x, x).real)


def run(args):
    os.nice(10)
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    started = time.time()
    if args.output.exists():
        raise FileExistsError('Preserve completed diagnostic')
    baseline = args.root / 'drff_v103_matched_continuation/no_nmf/run/supervised'
    trial = args.root / 'drff_v128_base_preserved_power_20261006/run/supervised'
    if read(trial.parent.parent / 'RESULT_AUDIT.json')['status'] != 'PASS_FULL_STAGED_D1_AND_ACTUAL_OUTPUTS':
        raise ValueError('Full output replay must pass first')
    record_sets, _ = check_pair(baseline, trial, 50)
    vpath = args.root / 'drff_v118_d1_wave_validation_20261006/verify118.py'
    if sha(vpath) != '434009c84d021b481a50a00624382f8fe0b437e0fa85298f587243b8e7868ac6':
        raise ValueError('Reference reconstruction verifier changed')
    spec = importlib.util.spec_from_file_location('frozen_reference', vpath)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    freeze = read(vpath.parent / 'FREEZE.json')
    thresholds_path = args.root / 'drff_v127_local_activity_20261006/ACTIVITY_THRESHOLDS.json'
    thresholds = read(thresholds_path)
    arrays = {}
    source_hashes = {}
    for entry in freeze['source_cache']:
        path = Path(entry['path'])
        if sha(path) != entry['npy_sha256']:
            raise ValueError('Reference cache changed')
        arrays[tuple(entry['key'])] = np.load(path, mmap_mode='r')
        source_hashes[path.name] = entry['npy_sha256']
    recipes = {r['case_id']: r for r in freeze['recipes'] if r['kind'] == 'mixture'}
    labels = {0: 'Air2', 2: 'Air2S', 3: 'Mavic3'}
    rows, replay_error, partition_error = [], 0., 0.
    for base, changed in zip(*record_sets):
        cid = base['case_id']
        mixture, reference = verifier.inputs(recipes[cid], arrays)
        specs = recipes[cid]['native_recipe']['sources']
        names = [labels[s['file_index']] for s in specs]
        full_energy = [energy(x) for x in reference]
        for label, folder, record in [('window_power', baseline, base), ('record_power', trial, changed)]:
            path = folder / 'evaluation_50' / cid / 'output.npy'
            if sha(path) != record['output_sha256']:
                raise ValueError('Saved prediction hash mismatch')
            output = np.load(path).astype(np.complex128)
            reference = reference.astype(np.complex128)
            error = [energy(output[k] - reference[k]) for k in (0, 1)]
            for k in (0, 1):
                replay_error = max(replay_error, abs(error[k] / full_energy[k] - record['nmse'][k]))
            partitioned = [0., 0.]
            for start in range(0, reference.shape[1], 63872):
                end = min(start + 63872, reference.shape[1])
                for k in (0, 1):
                    target_energy = energy(reference[k, start:end])
                    residual_energy = energy(output[k, start:end] - reference[k, start:end])
                    partitioned[k] += residual_energy
                    relative_power = target_energy / (end - start) / (full_energy[k] / reference.shape[1])
                    db = float(10 * np.log10(relative_power))
                    q = thresholds[names[k]]
                    activity = 'low' if db < q['q25_db'] else 'mid' if db < q['q75_db'] else 'high'
                    rows.append(dict(case_id=cid, model=label, source=names[k], source_index=k,
                        nominal_sir_db=record['sir_db'], nominal_weak=record['weak_source_index'] == k,
                        start=start, end=end, activity=activity, relative_activity_db=db,
                        reference_energy=target_energy, error_energy=residual_energy,
                        global_nmse_contribution=residual_energy / full_energy[k]))
            for k in (0, 1):
                partition_error = max(partition_error, abs(partitioned[k] - error[k]) / max(error[k], 1e-30))
    if replay_error > 1e-8 or partition_error > 1e-10:
        raise ValueError('Waveform/partition replay failed')
    groups = []
    for model in ('window_power', 'record_power'):
        for sir in (-10, 10):
            for activity in ('low', 'mid', 'high'):
                part = [r for r in rows if r['model'] == model and r['nominal_sir_db'] == sir
                        and r['nominal_weak'] and r['activity'] == activity]
                groups.append(dict(model=model, nominal_sir_db=sir, activity=activity, windows=len(part),
                    energy_pooled_nmse=sum(r['error_energy'] for r in part) / sum(r['reference_energy'] for r in part) if part else None,
                    contribution_to_four_case_mean_nmse=sum(r['global_nmse_contribution'] for r in part) / 4))
    args.output.mkdir(parents=True)
    with (args.output / 'WINDOWS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='SAVED_WAVEFORM_ACTIVITY_REPLAY_PASS', time=time.time(), elapsed_seconds=time.time() - started,
        epoch_budget=50, output_waveforms=24, source_segments=len(rows),
        maximum_global_nmse_replay_error=replay_error, maximum_partition_energy_relative_error=partition_error,
        thresholds=thresholds, thresholds_sha256=sha(thresholds_path), reference_hashes=source_hashes,
        source_code_sha256=sha(Path(__file__)), comparison_helper_sha256=sha(Path(comparison_module.__file__)),
        reference_verifier_sha256=sha(vpath), windows_sha256=sha(args.output / 'WINDOWS.csv'), groups=groups,
        independent_confirmation=False, gpu_used=False, threshold_fitted_on_evaluation=False,
        interpretation='Relative recorded power groups using pre-existing training quartiles; not ground-truth RF transmitter activity. '
        'Energy-pooled subgroup NMSE is descriptive, not a new checkpoint selection metric.')
    (args.output / 'COMPLETE.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', default='')
    run(parser.parse_args())
