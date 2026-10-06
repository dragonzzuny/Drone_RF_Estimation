"""CPU-only audit of real synthesized windows; no separator is evaluated."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from drone_rf.data import ScheduledMixtures, sha256


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    if args.cpus:
        os.sched_setaffinity(0, {int(n) for n in args.cpus.split(',')})
    started = time.time()
    rows, checked_ids = [], set()
    max_sir_error, exact_sum = 0., True
    for role in ('train_pack', 'validation_pack'):
        dataset = ScheduledMixtures(args.manifest, args.schedule, role)
        indices = np.flatnonzero(dataset.rows['epoch'] == 1)
        for position, index in enumerate(indices):
            item = dataset[int(index)]
            recipe = dataset.rows[index]
            power = np.mean(np.abs(item['references'].astype(np.complex128)) ** 2, axis=1)
            exact_sum &= np.array_equal(item['references'].sum(0), item['mixture'])
            if position < 10:
                if not np.array_equal(item['mixture'], dataset[int(index)]['mixture']):
                    raise ValueError('Mixture replay is not deterministic')
            start = int(recipe['crop_start'])
            local_fractions = []
            for field in ('first', 'second'):
                i = int(recipe[field])
                checked_ids.add(dataset.clips[i]['clip_id'])
                raw = dataset._array(i)[start:start + dataset.window_samples].astype(np.complex128)
                raw_power = float(np.vdot(raw, raw).real / len(raw))
                local_fractions.append(raw_power / dataset.clips[i]['mean_power'])
            local_sir = None
            if min(power) > 0:
                local_sir = float(10 * np.log10(power[0] / power[1]))
                expected = float(recipe['sir_db']) + 10 * np.log10(local_fractions[0] / local_fractions[1])
                error = abs(local_sir - expected)
                if error > 1e-4:
                    raise ValueError('Local power ratio disagrees with long-context synthesis')
                max_sir_error = max(max_sir_error, error)
            rows.append(dict(role=role, index=int(index), pair=' + '.join(item['categories']),
                nominal_sir_db=item['nominal_sir_db'], local_sir_db=local_sir,
                first_local_context_power_ratio=local_fractions[0],
                second_local_context_power_ratio=local_fractions[1],
                first_power=float(power[0]), second_power=float(power[1]),
                original_center_difference_hz=abs(item['center_frequencies_hz'][0] - item['center_frequencies_hz'][1])))
            if position % 100 == 0:
                write(args.output / 'PROGRESS.json', dict(stage='SYNTHESIS_AUDIT', role=role,
                    completed=position + 1, total=len(indices), time=time.time(),
                    elapsed_seconds=time.time() - started, pid=os.getpid(), gpu_used=False))
        print(f'{role}: {len(indices)} real mixtures checked', flush=True)
    if not exact_sum:
        raise ValueError('Stored source sum differs from input')
    summaries = []
    for role in ('train_pack', 'validation_pack'):
        for sir in (-10., 0., 10.):
            group = [r for r in rows if r['role'] == role and r['nominal_sir_db'] == sir]
            valid = [r for r in group if r['local_sir_db'] is not None]
            local = np.array([r['local_sir_db'] for r in valid])
            summaries.append(dict(role=role, nominal_sir_db=sir, examples=len(group),
                undefined_sir_examples=len(group) - len(valid),
                local_sir_quantiles_db=np.quantile(local, [.01, .1, .5, .9, .99]).tolist() if len(local) else None,
                local_sir_min_db=float(local.min()) if len(local) else None,
                local_sir_max_db=float(local.max()) if len(local) else None,
                deviation_over_3db=int(np.count_nonzero(np.abs(local - sir) > 3)),
                stronger_source_reversed=int(np.count_nonzero(local * sir < 0)) if sir else None,
                source_windows_below_1pct_context_power=sum(r[k] < .01 for r in group for k in
                    ('first_local_context_power_ratio', 'second_local_context_power_ratio'))))
    csv_path = args.output / 'LOCAL_POWER.csv'
    with csv_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import drone_rf.data as data_module
    result = dict(status='REAL_MIXTURE_INPUT_AUDIT_COMPLETE', time=time.time(),
        elapsed_seconds=time.time() - started, training_epoch_audited=1,
        training_examples=sum(r['role'] == 'train_pack' for r in rows),
        validation_examples=sum(r['role'] == 'validation_pack' for r in rows),
        unique_clips_used=len(checked_ids), exact_mixture_sum=exact_sum,
        maximum_local_sir_replay_error_db=max_sir_error, summaries=summaries,
        original_centers_differ_examples=sum(r['original_center_difference_hz'] != 0 for r in rows),
        csv_sha256=sha256(csv_path), worker_sha256=sha256(__file__), loader_sha256=sha256(data_module.__file__),
        manifest_sha256=sha256(args.manifest), schedule_receipt_sha256=sha256(args.schedule / 'COMPLETE.json'),
        gpu_used=False, model_performance_evaluated=False, training_admitted=False,
        heldout_read=False,
        interpretation='Synthetic frequency-rebased baseband; power includes receiver noise. '
        'Numerical nonzero/low-power windows are not annotated transmitter activity or physical aircraft counts.')
    write(args.output / 'COMPLETE.json', result)
    write(args.output / 'PROGRESS.json', dict(stage='COMPLETE', **result))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--schedule', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpus', default='')
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.output.exists() and not (args.output / 'COMPLETE.json').exists():
            write(args.output / 'FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
