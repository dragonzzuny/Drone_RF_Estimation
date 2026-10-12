"""Descriptive paired comparison, without new inference or threshold selection.

Only finite saved waveform scores are compared; undefined SI-SDR stays undefined.
Rows are dependent development mixtures, not independent source recordings.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    if not values:
        return None
    if any(x is None or not math.isfinite(x) for x in values):
        return dict(n=len(values), undefined=sum(x is None or not math.isfinite(x) for x in values))
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def comparison(base, candidate):
    assert len(base) == len(candidate)
    groups = defaultdict(list)
    for left, right in zip(base, candidate):
        for key in ('index', 'count', 'categories', 'weakest_index'):
            assert left[key] == right[key], (left['index'], key)
        for a, b in zip(left['reference_power'], right['reference_power']):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-14)
        k = left['count']
        assert len(left['nmse']) == len(right['nmse']) == k
        for j, category in enumerate(left['categories']):
            power = left['reference_power'][j]
            other = sum(left['reference_power']) - power
            sir = 10 * math.log10(power / other) if other > 0 else None
            row = dict(base_nmse=left['nmse'][j], candidate_nmse=right['nmse'][j],
                       base_si=left['si_sdr'][j], candidate_si=right['si_sdr'][j])
            groups[(k, 'all')].append(row)
            groups[(k, category)].append(row)
            if j == left['weakest_index']:
                groups[(k, 'weakest')].append(row)
            if sir is not None:
                groups[(k, 'actual_SIR_below_minus20' if sir < -20 else 'actual_SIR_at_least_minus20')].append(row)
    result = []
    for (k, group), rows in sorted(groups.items()):
        n = [r['candidate_nmse'] - r['base_nmse'] for r in rows]
        sd = [r['candidate_si'] - r['base_si']
              if r['candidate_si'] is not None and r['base_si'] is not None else None for r in rows]
        result.append(dict(count=k, group=group, sources=len(rows),
            base_nmse=describe([r['base_nmse'] for r in rows]),
            candidate_nmse=describe([r['candidate_nmse'] for r in rows]),
            nmse_change=describe(n), si_change=describe(sd),
            nmse_improved=sum(v < 0 for v in n), nmse_worsened=sum(v > 0 for v in n),
            both_improved=sum(v < 0 and s is not None and s > 0 for v, s in zip(n, sd)),
            both_worsened=sum(v > 0 and s is not None and s < 0 for v, s in zip(n, sd))))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--base-sha256', required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.base) == a.base_sha256 and sha(a.candidate) == a.candidate_sha256
    base, candidate = read(a.base), read(a.candidate)
    assert base['rows_sha256'] == candidate['rows_sha256']
    result = dict(status='VERIFIED_SAVED_ROWS', base_sha256=a.base_sha256,
                  candidate_sha256=a.candidate_sha256, rows_sha256=base['rows_sha256'],
                  comparisons=comparison(base['rows'], candidate['rows']),
                  inferential_statistics=False, new_inference=False,
                  caveat='Same-record B development mixtures; dependent windows and sources; not new-recording generalization')
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps([r for r in result['comparisons'] if r['group'] in ('all', 'weakest')], indent=2))


if __name__ == '__main__':
    main()
