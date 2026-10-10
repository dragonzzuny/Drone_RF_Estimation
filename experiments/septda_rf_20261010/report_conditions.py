"""Describe every saved DEV stratum; no inference, I/Q reads or selection changes."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/architecture_audit_20261009'))
import watch_epochs as w

SIR_BINS = [(-math.inf, -20, '<-20'), (-20, -10, '[-20,-10)'),
            (-10, 0, '[-10,0)'), (0, 10, '[0,10)'),
            (10, 20, '[10,20)'), (20, math.inf, '>=20')]


def metric_summary(items):
    nmse = [r['nmse'] for r in items]
    si = [r['si_sdr'] for r in items]
    finite_si = [s for s in si if s is not None and math.isfinite(s)]
    return dict(source_cases=len(items),
                mixture_cases=len({r['index'] for r in items}),
                mean_nmse=statistics.mean(nmse),
                median_nmse=statistics.median(nmse),
                mean_si_sdr=statistics.mean(finite_si) if len(finite_si) == len(si) else None,
                nonfinite_si_sdr=len(si) - len(finite_si))


def describe(paths, identities=None):
    summaries, data, sources, strata, confusion = {}, {}, {}, [], []
    for method, path in paths.items():
        summary, checked = w.validate(path, identities)
        if identities is None:
            identities = checked
        summaries[method] = summary
        data[method] = {r['index']: r for r in w.read(path)['rows']}
        flat, groups = {}, defaultdict(list)
        for case in data[method].values():
            count = case['count']
            power = case['reference_power']
            assert len(power) == count and all(math.isfinite(v) and v > 0 for v in power)
            order = sorted(range(count), key=lambda j: (-power[j], j))
            assert case['weakest_index'] == min(range(count), key=lambda j: (power[j], j))
            for j in range(count):
                sir = None if count == 1 else 10 * math.log10(power[j] / sum(v for k, v in enumerate(power) if k != j))
                bucket = 'single' if sir is None else next(label for lo, hi, label in SIR_BINS if lo <= sir < hi)
                row = dict(index=case['index'], source=j, count=count,
                           category=case['categories'][j], recording_group=case['pack_ids'][j],
                           power_rank=order.index(j) + 1, sir_db=sir, sir_bin=bucket,
                           nmse=case['nmse'][j], si_sdr=case['si_sdr'][j])
                flat[(case['index'], j)] = row
                for field in ('category', 'recording_group', 'power_rank', 'sir_bin'):
                    groups[(count, field, str(row[field]))].append(row)
        assert len(flat) == 1260
        sources[method] = flat
        for (count, field, label), items in sorted(groups.items()):
            strata.append(dict(method=method, count=count, stratum=field, label=label,
                               **metric_summary(items)))
        for count in (1, 2, 3):
            rows = [r for r in data[method].values() if r['count'] == count]
            assert all(r['predicted_count'] in (1, 2, 3) for r in rows)
            confusion.append(dict(method=method, constructed_count=count, cases=len(rows),
                                  predicted_1=sum(r['predicted_count'] == 1 for r in rows),
                                  predicted_2=sum(r['predicted_count'] == 2 for r in rows),
                                  predicted_3=sum(r['predicted_count'] == 3 for r in rows)))
    comparisons = []
    if 'septda_e1' in sources:
        for reference in ('parent', 'same_schedule_control'):
            groups = defaultdict(list)
            for key, row in sources['septda_e1'].items():
                other = sources[reference][key]
                dn = row['nmse'] - other['nmse']
                si, ref_si = row['si_sdr'], other['si_sdr']
                ds = si - ref_si if si is not None and ref_si is not None and math.isfinite(si) and math.isfinite(ref_si) else None
                change = dict(index=row['index'], nmse=dn, si_sdr=ds)
                for field in ('category', 'recording_group', 'power_rank', 'sir_bin'):
                    groups[(row['count'], field, str(row[field]))].append(change)
            for (count, field, label), items in sorted(groups.items()):
                summary = metric_summary(items)
                comparisons.append(dict(reference=reference, count=count, stratum=field, label=label,
                    source_cases=summary['source_cases'], mixture_cases=summary['mixture_cases'],
                    mean_nmse_delta=summary['mean_nmse'], mean_si_sdr_delta=summary['mean_si_sdr'],
                    nonfinite_si_pairs=summary['nonfinite_si_sdr'],
                    both_improved=sum(r['nmse'] < 0 and r['si_sdr'] is not None and r['si_sdr'] > 0 for r in items),
                    both_worsened=sum(r['nmse'] > 0 and r['si_sdr'] is not None and r['si_sdr'] < 0 for r in items)))
    return dict(summaries=summaries, strata=strata, comparisons=comparisons,
                construction_count_confusion=confusion)


def run(root, output, baseline_only=False):
    p = w.read(root / 'PROTOCOL.json')
    paths = dict(parent=Path(p['original_protocol']['baseline']),
                 same_schedule_control=Path(p['comparison_control']))
    if not baseline_only:
        audit = w.read(ROOT / 'reports/2026-10-10/SEPTDA_RF_AUDIT.json')
        assert audit['status'] == 'PASS' and audit['protocol_sha256'] == w.digest(root / 'PROTOCOL.json')
        assert audit['hashes']['VALIDATION_001.json'] == w.digest(root / 'VALIDATION_001.json')
        paths['septda_e1'] = root / 'VALIDATION_001.json'
    result = describe(paths)
    result.update(status='BASELINE_PREPARATION' if baseline_only else 'COMPLETE_CHECKED',
                  protocol_sha256=w.digest(root / 'PROTOCOL.json'),
                  validation_sha256={k: w.digest(v) for k, v in paths.items()},
                  reporter_sha256=w.digest(Path(__file__)), all_630_identities_checked=True,
                  iq_reads=0, new_inference=0, heldout_read=False, independent_test=False,
                  sir_definition='Reference source power divided by SUM of other reference powers in the evaluated window; excludes coherent cross terms.',
                  limitation='All strata reported, descriptive repeated DEV with 5 recording groups. Category and recording are confounded. Synthetic record count is not physical aircraft count. No threshold or model selection from these strata.')
    output.mkdir(parents=True, exist_ok=True)
    w.write(output / 'SEPTDA_RF_CONDITIONS.json', result)
    lines = ['# SepTDA 참고 분리부: 전체 조건별 개발 결과', '',
             'RFUAV 같은 원 RF 대역, native 100 MS/s, TRAIN 8 / DEV 5 원기록 묶음, seed 0. '
             '같은 부모와 schedule3의 2,400혼합·75업데이트 비교다. 전체630혼합의 정답·기종·기록·전력을 대조하고 저장된 지표를 집계했다. '
             '파형 읽기와 추가 추론은 없다. 신호 순위1이 가장 강하다.', '',
             '국소 SIR은 평가 창에서 한 정답 성분 전력 / 나머지 정답 전력의 합이다. '
             '복소 교차항을 포함한 간섭 파형 전력과 구별한다. 모든 기종·기록·세기 순위·전력비 구간과 개수 혼동행렬은 JSON에 보존한다.', '',
             '| 성분 수 | 전력 순위 | 사례 수 | 부모 NMSE | 대조 NMSE | 새 분리부 NMSE | 부모 SI-SDR | 대조 SI-SDR | 새 분리부 SI-SDR |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def cell(value):
        return '미정의' if value is None else f'{value:.6f}'
    for count in (1, 2, 3):
        for rank in range(1, count + 1):
            entries = {r['method']: r for r in result['strata'] if r['count'] == count and r['stratum'] == 'power_rank' and r['label'] == str(rank)}
            values = [entries.get(m) for m in ('parent', 'same_schedule_control', 'septda_e1')]
            cells = [cell(v[key]) if v else '결과 대기' for key in ('mean_nmse', 'mean_si_sdr') for v in values]
            lines.append(f"|{count}|{rank}|{values[0]['source_cases']}|" + '|'.join(cells) + '|')
    lines += ['', '같은 DEV의 반복 분석이며 독립 확인 평가가 아니다. 각 기종과 DEV 원기록이 묶여 있어 두 효과를 분리할 수 없다. '
              '유리한 하위집합으로 채택 기준을 바꾸지 않으며 원래 공동 기준을 유지한다. '
              '개수 정확도는 합성한 원기록 수를 기준으로 하며 실제 드론 대수 검증을 뜻하지 않는다.', '',
              '[모든 조건·대응 변화·개수 혼동행렬](SEPTDA_RF_CONDITIONS.json)']
    w.write(output / 'SEPTDA_RF_CONDITIONS_KO.md', '\n'.join(lines) + '\n')
    print(dict(status=result['status'], methods=list(paths), strata=len(result['strata']), paired_strata=len(result['comparisons'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve(), args.baseline_only)
