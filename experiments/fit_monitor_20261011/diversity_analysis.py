"""Descriptive paired analysis of every saved TRAIN-probe source; no selection."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'local/tfgridnet_diversity_20261011_v1'
PUBLIC = ROOT / 'reports/2026-10-11'


def read(path):
    return json.loads(path.read_text())


def main(added, kind='diversity'):
    run = RUN if kind == 'diversity' else ROOT / 'local/tfgridnet_head_adapt_20261011_v1'
    prefix = 'TFGRIDNET_DIVERSITY' if kind == 'diversity' else 'TFGRIDNET_HEAD_ADAPT'
    before_path = run / 'ADDED_000.json'
    after_path = run / f'ADDED_{added:03d}.json'
    before, after = read(before_path), read(after_path)
    assert added in (16, 32) and after['added_updates'] == added
    assert [r['index'] for r in before['rows']] == [r['index'] for r in after['rows']]
    components, mixtures = [], []
    for a, b in zip(before['rows'], after['rows']):
        assert a['reference_power'] == b['reference_power'] and a['categories'] == b['categories']
        total = sum(a['reference_power'])
        for j, (category, power) in enumerate(zip(a['categories'], a['reference_power'])):
            ratio = 10 * math.log10(power / (total - power))
            components.append(dict(index=a['index'], source=j, count=a['count'], category=category,
                power_ratio_db=ratio, power_bin='below_-25dB' if ratio < -25 else
                    '-25_to_-15dB' if ratio < -15 else 'at_least_-15dB',
                weakest=j == a['weakest_index'], before_nmse=a['nmse'][j], after_nmse=b['nmse'][j],
                before_si_sdr=a['si_sdr'][j], after_si_sdr=b['si_sdr'][j],
                nmse_down=b['nmse'][j] < a['nmse'][j], si_up=b['si_sdr'][j] > a['si_sdr'][j]))
        mixtures.append(dict(index=a['index'], count=a['count'],
            before_active_energy_error=sum(n*p for n,p in zip(a['nmse'],a['reference_power']))/total,
            after_active_energy_error=sum(n*p for n,p in zip(b['nmse'],b['reference_power']))/total,
            before_predicted_count=a['predicted_count'], after_predicted_count=b['predicted_count']))

    def group(rows):
        return dict(sources=len(rows), before_nmse=stats.mean(r['before_nmse'] for r in rows),
            after_nmse=stats.mean(r['after_nmse'] for r in rows),
            before_median_nmse=stats.median(r['before_nmse'] for r in rows),
            after_median_nmse=stats.median(r['after_nmse'] for r in rows),
            before_si_sdr=stats.mean(r['before_si_sdr'] for r in rows),
            after_si_sdr=stats.mean(r['after_si_sdr'] for r in rows),
            jointly_improved=sum(r['nmse_down'] and r['si_up'] for r in rows),
            both_worsened=sum(not r['nmse_down'] and not r['si_up'] and
                r['before_nmse'] != r['after_nmse'] and r['before_si_sdr'] != r['after_si_sdr'] for r in rows),
            nmse_down_si_not_up=sum(r['nmse_down'] and not r['si_up'] for r in rows),
            si_up_nmse_not_down=sum(r['si_up'] and not r['nmse_down'] for r in rows),
            after_source_nmse_ge_one=sum(r['after_nmse'] >= 1 for r in rows),
            before_max_nmse=max(r['before_nmse'] for r in rows),
            after_max_nmse=max(r['after_nmse'] for r in rows))

    groups = []
    for count in (2, 3):
        all_rows = [r for r in components if r['count'] == count]
        groups.append(dict(count=count, subgroup='all', **group(all_rows)))
        groups.append(dict(count=count, subgroup='weakest', **group([r for r in all_rows if r['weakest']])))
        for label in ('below_-25dB', '-25_to_-15dB', 'at_least_-15dB'):
            subset = [r for r in all_rows if r['power_bin'] == label]
            if subset:
                groups.append(dict(count=count, subgroup=label, **group(subset)))
    categories = [dict(category=category, **group([r for r in components if r['category'] == category]))
                  for category in sorted({r['category'] for r in components})]
    mix_summary = []
    for count in (2, 3):
        rows = [r for r in mixtures if r['count'] == count]
        mix_summary.append(dict(count=count, mixtures=len(rows),
            before_mean_active_energy_error=stats.mean(r['before_active_energy_error'] for r in rows),
            after_mean_active_energy_error=stats.mean(r['after_active_energy_error'] for r in rows),
            before_construction_count_correct=sum(r['before_predicted_count'] == count for r in rows),
            after_construction_count_correct=sum(r['after_predicted_count'] == count for r in rows)))
    result = dict(added_updates=added, groups=groups, categories=categories, mixture_summary=mix_summary,
        components=components, mixtures=mixtures, validation_read=False, heldout_read=False,
        limitations=['TRAIN30 diagnosis; original recordings may be shared with TRAIN128',
            'Power bins are descriptive, transferred from preceding exploratory analysis, not causal or selection criteria',
            'Active-energy error excludes inactive slots and background; denominator is sum of separate reference powers',
            'Construction count is number of mixed recordings, not verified aircraft or emitting-source count',
            'No confidence intervals: sources and mixtures share recordings and are not independent replications'],
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (before_path, after_path, Path(__file__))})
    target = PUBLIC / f'{prefix}_ADDED{added:03d}_ANALYSIS.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('added_updates','groups','mixture_summary')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--added', type=int, choices=[16,32], required=True)
    parser.add_argument('--kind', choices=['diversity','head'], default='diversity')
    args = parser.parse_args()
    main(args.added, args.kind)
