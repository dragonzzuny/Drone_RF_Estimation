"""Independently recheck saved paired results; no I/Q or checkpoints are read."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(study, report_path, output):
    report = read(report_path)
    epochs = report['common_completed_epoch']
    if not 1 <= epochs <= 5:
        raise ValueError('Unexpected completed epoch count')
    if report['protocol_sha256'] != digest(study/'PROTOCOL.json'):
        raise ValueError('Protocol mismatch')
    generator = Path(__file__).with_name('paired_context_report.py')
    if report['source_sha256'] != digest(generator):
        raise ValueError('Report generator changed')
    rows = {(r['epoch'], r['count']): r for r in report['rows']}
    if len(rows) != len(report['rows']) or set(rows) != {
        (e, c) for e in range(1, epochs+1) for c in (2, 3)
    }:
        raise ValueError('Missing or duplicate summary rows')
    checked = 0
    for epoch in range(1, epochs+1):
        paths = {arm: study/arm/f'VALIDATION_{epoch:03d}.json'
                 for arm in ('local', 'long')}
        values = {}
        for arm, path in paths.items():
            raw = read(path)['rows']
            values[arm] = {r['index']: r for r in raw}
            if len(raw) != 630 or len(values[arm]) != 630:
                raise ValueError('Incomplete validation')
        if values['local'].keys() != values['long'].keys():
            raise ValueError('Mixture indices changed')
        for count in (2, 3):
            stored = rows[epoch, count]
            if stored['cases'] != 210 or stored['validation_sha256'] != {
                arm: digest(path) for arm, path in paths.items()
            }:
                raise ValueError('Row count or validation hash mismatch')
            deltas_n, deltas_s = [], []
            both = weak = every = 0
            for index, local in values['local'].items():
                long = values['long'][index]
                if local['count'] != long['count']:
                    raise ValueError('Reference count changed')
                if local['count'] != count:
                    continue
                if local['weakest_index'] != long['weakest_index']:
                    raise ValueError('Weak reference changed')
                if any(len(r[k]) != count for r in (local, long)
                       for k in ('nmse', 'si_sdr')):
                    raise ValueError('Wrong source metric dimensions')
                nd = [long['nmse'][j]-local['nmse'][j] for j in range(count)]
                sd = [long['si_sdr'][j]-local['si_sdr'][j] for j in range(count)]
                if not all(math.isfinite(v) for v in nd+sd):
                    raise ValueError('Nonfinite paired differences')
                n, s = math.fsum(nd)/count, math.fsum(sd)/count
                deltas_n.append(n)
                deltas_s.append(s)
                both += n < 0 and s > 0
                j = local['weakest_index']
                weak += nd[j] < 0 and sd[j] > 0
                every += all(nd[j] < 0 and sd[j] > 0 for j in range(count))
            if len(deltas_n) != 210:
                raise ValueError('Missing matched mixtures')
            expected = dict(
                mean_nmse_delta=math.fsum(deltas_n)/210,
                median_nmse_delta=statistics.median(deltas_n),
                mean_si_delta=math.fsum(deltas_s)/210,
                median_si_delta=statistics.median(deltas_s))
            for field, value in expected.items():
                if not math.isclose(stored[field], value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Difference aggregate mismatch: '+field)
            if (both, weak, every) != (
                stored['mixture_mean_both_better'], stored['weakest_both_better'],
                stored['every_source_both_better']):
                raise ValueError('Joint success counts mismatch')
            checked += 1
    audit = dict(status='PASS', report_sha256=digest(report_path),
                 auditor_sha256=digest(Path(__file__)), common_epochs=epochs,
                 validated_rows=checked, independent_source_delta_aggregation=True,
                 all_three_success_counts_match=True, waveform_reads=0, heldout_read=False)
    output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('study', 'report', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    run(a.study, a.report, a.output)
