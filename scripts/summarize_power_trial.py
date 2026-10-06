"""Compare completed, matched development evaluations without retraining."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pair(baseline, trial, stage):
    rows, selections, receipts = [], [], []
    for folder in (baseline, trial):
        evaluation = folder / f'evaluation_{stage}'
        complete = read(evaluation / 'COMPLETE.json')
        selection = read(folder / f'STAGE_{stage}.json')
        data = read(evaluation / 'RECORDS.json')
        if complete['status'] != 'COMPLETE' or not complete.get('canonical_metrics'):
            raise ValueError('Incomplete or noncanonical evaluation')
        if (complete['records'] != 12 or len(data) != 12 or
                selection['completed_epochs'] != stage or
                complete['selected_checkpoint_sha256'] != selection['best_sha256']):
            raise ValueError('Selection/evaluation mismatch')
        if len({r['case_id'] for r in data}) != 12:
            raise ValueError('Duplicated case')
        for r in data:
            if r['status'] != 'SCORED' or r['stage'] != stage or r['day'] != 'D1':
                raise ValueError('Wrong scope or incomplete case')
            if len(r['nmse']) != 2 or not all(math.isfinite(v) and v >= 0 for v in r['nmse']):
                raise ValueError('Invalid waveform NMSE')
            if abs(statistics.mean(r['nmse']) - r['mean_complex_nmse']) > 1e-9:
                raise ValueError('Mean NMSE inconsistent with components')
        rows.append(sorted(data, key=lambda r: r['case_id']))
        selections.append(selection)
        receipts.append(read(evaluation / 'SOURCE_RECEIPTS.json'))
    if receipts[0] != receipts[1]:
        raise ValueError('Source receipts differ')
    if selections[0]['updates'] != selections[1]['updates']:
        raise ValueError('Unequal update budget')
    if selections[0]['parent_checkpoint_sha256'] != selections[1]['parent_checkpoint_sha256']:
        raise ValueError('Different initial checkpoint')
    for a, b in zip(*rows):
        for key in ('case_id', 'pair', 'sir_db', 'window', 'weak_source_index'):
            if a[key] != b[key]:
                raise ValueError('Different evaluation input or reference role')
        # Identical source hashes are required above; reductions differ at ~1e-13 dB.
        if not all(math.isclose(x, y, rel_tol=0, abs_tol=1e-10) for x, y in zip(a['input_si_db'], b['input_si_db'])):
            raise ValueError('Input SI-SDR differs beyond numerical roundoff')
    return rows, selections


def summarize(records):
    result = dict(overall_nmse=statistics.mean(r['mean_complex_nmse'] for r in records),
        reported_pit_nmse=statistics.mean(v for r in records for v in r['pit_nmse']),
        both_sources_positive_si_gain=sum(r['both_gain_positive'] for r in records),
        weak={})
    for sir, name, index in ((-10, 'air', 0), (10, 'mavic3', 1)):
        group = [r for r in records if r['sir_db'] == sir]
        if len(group) != 4 or not all(r['weak_source_index'] == index for r in group):
            raise ValueError('Unexpected weak-source conditions')
        finite = all(r['si_status'][index] == 'finite' for r in group)
        result['weak'][name] = dict(cases=4,
            nmse=statistics.mean(r['nmse'][index] for r in group),
            si_sdr_db=statistics.mean(r['si_db'][index] for r in group) if finite else None,
            all_si_sdr_finite=finite)
    return result


def run(args):
    comparisons, files = [], {}
    for stage in (5, 10, 25, 50):
        if not (args.trial / f'evaluation_{stage}/COMPLETE.json').exists():
            continue
        rows, selections = check_pair(args.baseline, args.trial, stage)
        metrics = [summarize(r) for r in rows]
        comparisons.append(dict(epoch_budget=stage, updates_per_arm=selections[0]['updates'],
            selected_epochs=[s['selected_epoch'] for s in selections],
            selected_checkpoint_sha256=[s['best_sha256'] for s in selections],
            parent_checkpoint_sha256=selections[0]['parent_checkpoint_sha256'],
            baseline=metrics[0], record_power=metrics[1],
            relative_nmse_change_pct=100 * (metrics[1]['overall_nmse'] / metrics[0]['overall_nmse'] - 1),
            cases=[dict(case_id=a['case_id'], baseline_nmse=a['mean_complex_nmse'],
                record_power_nmse=b['mean_complex_nmse']) for a, b in zip(*rows)]))
        for label, folder in (('baseline', args.baseline), ('trial', args.trial)):
            for name in ('RECORDS.json', 'COMPLETE.json', 'SOURCE_RECEIPTS.json'):
                files[f'{label}/evaluation_{stage}/{name}'] = sha(folder / f'evaluation_{stage}' / name)
            files[f'{label}/STAGE_{stage}.json'] = sha(folder / f'STAGE_{stage}.json')
    if not comparisons:
        raise ValueError('No completed matched evaluations')
    audit = args.trial.parent.parent / 'RESULT_AUDIT.json'
    audit_pass = audit.exists() and read(audit)['status'] == 'PASS_FULL_STAGED_D1_AND_ACTUAL_OUTPUTS'
    if audit_pass:
        files['trial/RESULT_AUDIT.json'] = sha(audit)
    final = comparisons[-1]['epoch_budget'] == 50 and audit_pass
    result = dict(status='FULL_50_EPOCH_COMPARISON_AUDITED' if final else 'INTERIM_COMPARISON',
        checked_at=datetime.datetime.now().astimezone().isoformat(),
        dataset='DRFF-R2 D1 repeatedly used development evaluation, 12 synthetic mixtures',
        independent_confirmation=False, seed=0, baseline_reused=True,
        full_84_output_replay_audit_pass=audit_pass, comparisons=comparisons, source_hashes=files)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'power_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--trial', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
