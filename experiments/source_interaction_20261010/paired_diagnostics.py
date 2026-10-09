"""Read-only paired development diagnostics for the registered source-head study.

Uses completed validation receipts only. No I/Q/checkpoint reads or model
selection changes. Crops and source components are not independent replicates.
The SIR strata reuse the previous source_strata_report definitions.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/architecture_audit_20261009'))
import watch_epochs as watch
from source_strata_report import sir_label

ARMS = ('retained_unet', 'source_interaction')
IDENTITY = ('index', 'count', 'categories', 'pack_ids', 'nominal_levels_db', 'reference_power')


def component_pairs(before, after):
    """Align whole-crop PIT metrics by reference, never by predicted slot."""
    def indexed(rows):
        result = {r['index']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate validation index')
        return result
    old, new = indexed(before), indexed(after)
    if old.keys() != new.keys():
        raise ValueError('Different cases')
    items = []
    for index in sorted(old):
        a, b = old[index], new[index]
        if any(a[k] != b[k] for k in IDENTITY):
            raise ValueError('Different reference identities/powers')
        count = a['count']
        if count == 1:
            continue  # Preserved in the original validation, outside this diagnostic.
        power = a['reference_power']
        if count not in (2, 3) or len(power) != count or any(
                not math.isfinite(v) or v <= 0 for v in power):
            raise ValueError('Invalid reference power/count')
        if any(len(r[k]) != count for r in (a, b) for k in ('nmse', 'si_sdr')):
            raise ValueError('Incomplete component metrics')
        for j in range(count):
            values = [a['nmse'][j], b['nmse'][j], a['si_sdr'][j], b['si_sdr'][j]]
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError('Nonfinite metric: no silent exclusions')
            if any(v < 0 for v in values[:2]):
                raise ValueError('Negative NMSE')
            sir = 10 * math.log10(power[j] / sum(p for k, p in enumerate(power) if k != j))
            items.append(dict(index=index, reference_index=j, count=count,
                category=a['categories'][j], combination=' + '.join(sorted(a['categories'])),
                sir_bin=sir_label(sir), source_sir_db=sir, reference_power=power[j],
                weakest=power[j] == min(power), nmse_before=values[0], nmse_after=values[1],
                si_before=values[2], si_after=values[3]))
    return items


def aggregate(items):
    if not items:
        raise ValueError('Empty comparison')
    dn = [r['nmse_after'] - r['nmse_before'] for r in items]
    ds = [r['si_after'] - r['si_before'] for r in items]
    return dict(source_cases=len(items), mixture_cases=len({r['index'] for r in items}),
        nmse_before=stats.mean(r['nmse_before'] for r in items),
        nmse_after=stats.mean(r['nmse_after'] for r in items),
        mean_delta_nmse=stats.mean(dn), median_delta_nmse=stats.median(dn),
        si_before=stats.mean(r['si_before'] for r in items),
        si_after=stats.mean(r['si_after'] for r in items),
        mean_delta_si=stats.mean(ds), median_delta_si=stats.median(ds),
        nmse_better=sum(d < -1e-12 for d in dn), si_better=sum(d > 1e-12 for d in ds),
        both_better=sum(n < -1e-12 and s > 1e-12 for n, s in zip(dn, ds)),
        both_worse=sum(n > 1e-12 and s < -1e-12 for n, s in zip(dn, ds)))


def compare(before, after, before_label, after_label):
    items = component_pairs(before['rows'], after['rows'])
    if len(items) != 1050 or len({r['index'] for r in items}) != 420:
        raise ValueError('Expected all 420 multiple-source mixtures / 1050 components')
    tables = {}
    for name, fields in [('count', ('count',)), ('category', ('count', 'category')),
                         ('combination', ('count', 'combination')),
                         ('power', ('count', 'sir_bin')),
                         ('category_power', ('count', 'category', 'sir_bin'))]:
        groups = defaultdict(list)
        for item in items:
            groups[tuple(item[k] for k in fields)].append(item)
        tables[name] = [dict(zip(fields, key), **aggregate(group))
                       for key, group in sorted(groups.items())]
        if sum(r['source_cases'] for r in tables[name]) != 1050:
            raise ValueError('Strata lost cases')
    for count in (2, 3):
        got = next(r for r in tables['count'] if r['count'] == count)
        for data, suffix in ((before, 'before'), (after, 'after')):
            expected = next(r for r in data['by_count'] if r['count'] == count)
            if not watch.close(got['nmse_' + suffix], expected['mean_nmse']) or not watch.close(
                    got['si_' + suffix], expected['mean_si_sdr']):
                raise ValueError('Component means differ from registered aggregate')
    weakest = []
    joint = []
    for count in (2, 3):
        group = [r for r in items if r['count'] == count and r['weakest']]
        if len(group) != 210:
            raise ValueError('Tied weakest references need a predeclared tie rule')
        weakest.append(dict(count=count, **aggregate(group)))
        mixes = defaultdict(list)
        for r in items:
            if r['count'] == count:
                mixes[r['index']].append(r)
        row = dict(count=count, mixtures=len(mixes))
        for suffix in ('before', 'after'):
            row['all_nmse_below_one_' + suffix] = sum(
                all(r['nmse_' + suffix] < 1 for r in group) for group in mixes.values())
            row['all_absolute_si_positive_' + suffix] = sum(
                all(r['si_' + suffix] > 0 for r in group) for group in mixes.values())
        row['every_source_improved_both'] = sum(all(
            r['nmse_after'] < r['nmse_before'] - 1e-12 and r['si_after'] > r['si_before'] + 1e-12
            for r in group) for group in mixes.values())
        joint.append(row)
    return dict(before=before_label, after=after_label, **tables, weakest=weakest, all_source=joint)


def run(study, output):
    snapshot = watch.snapshot(study)
    plan = watch.read(study / 'PROTOCOL.json')
    if tuple(plan['arms']) != ARMS:
        raise ValueError('Wrong registered study')
    parent_path = Path(plan['validation_identity_template'])
    parent = watch.read(parent_path)
    comparisons, hashes, events = [], {'parent': watch.digest(parent_path)}, {}
    for event in snapshot['events']:
        arm, epoch = event['arm'], event['epoch']
        label = f'{arm}/e{epoch}'
        path = study / arm / f'VALIDATION_{epoch:03d}.json'
        data = watch.read(path)
        comparisons.append(compare(parent, data, 'parent/e0', label))
        hashes[label] = watch.digest(path)
        events[(arm, epoch)] = data
    for epoch in range(1, snapshot['common_epoch'] + 1):
        comparisons.append(compare(events[(ARMS[0], epoch)], events[(ARMS[1], epoch)],
            f'{ARMS[0]}/e{epoch}', f'{ARMS[1]}/e{epoch}'))
    result = dict(status=snapshot['status'], observed_at=snapshot['reported_at'],
        protocol_sha256=snapshot['protocol_sha256'], source_sha256=watch.digest(Path(__file__)),
        common_completed_epoch=snapshot['common_epoch'], validation_sha256=hashes,
        comparisons=comparisons, heldout_read=False, waveform_reads=0, checkpoint_reads=0,
        optimizer_updates=0, independent_test=False, selection_changed=False,
        inference='unchanged single pass without references; references used only to evaluate outputs',
        sir_definition='10log10(reference power / sum of other reference powers), without complex cross terms',
        limitation='descriptive correlated crops from five recording groups; no significance/independent-sample claims')
    watch.write(output.with_suffix('.json'), result)
    lines = ['# 성분 상호작용 비교: 같은 사례의 조건별 변화', '',
        f"확인 시각 {result['observed_at']}. 두 군 공통 완료 {result['common_completed_epoch']} epoch.", '',
        '학습 중 저장이 끝난 결과만 CPU에서 읽었다. 매 비교의 420혼합·1,050성분을 유지하며 새 I/Q·'
        '체크포인트·보류 자료는 읽지 않았다. 동일한 정답 성분끼리 비교하므로 출력 슬롯의 순서 변경은 '
        '성능 차이로 세지 않는다. 기종 식별의 성공을 뜻하지 않는다.', '',
        '아래 변화는 후보/추가 학습 후 − 비교 전이다. NMSE는 음수, SI-SDR은 양수여야 개선이다. '
        '두 지표 동시 개선 수는 독립 성공률 추정치가 아닌 상관된 개발 사례의 기술 통계다.', '',
        '| 비교 전 → 후 | 성분 수 | Δ NMSE ↓ | Δ SI-SDR ↑ dB | 두 지표 동시 개선 성분 | 약신호 Δ NMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        for row in c['count']:
            weak = next(w for w in c['weakest'] if w['count'] == row['count'])
            lines.append(f"| {c['before']} → {c['after']} | {row['count']} | "
                f"{row['mean_delta_nmse']:+.6f} | {row['mean_delta_si']:+.3f} | "
                f"{row['both_better']}/{row['source_cases']} | {weak['mean_delta_nmse']:+.6f} |")
    paired = [c for c in comparisons if c['before'].startswith(ARMS[0])]
    for title, key, field in [('같은 epoch의 성분별 전력 조건', 'power', 'sir_bin'),
                               ('같은 epoch의 기종 조건', 'category', 'category')]:
        lines += ['', '## ' + title, '',
            '| 추가 epoch | 성분 수 | 조건 | 성분 사례 | Δ NMSE ↓ | Δ SI-SDR ↑ dB | 두 지표 동시 개선 |',
            '|---|---:|---|---:|---:|---:|---:|']
        for c in paired:
            for row in c[key]:
                lines.append(f"| {c['after'].split('/')[-1]} | {row['count']} | {row[field]} | "
                    f"{row['source_cases']} | {row['mean_delta_nmse']:+.6f} | "
                    f"{row['mean_delta_si']:+.3f} | {row['both_better']}/{row['source_cases']} |")
    lines += ['', 'SIR 구간은 기존 분석의 <−20, [−20,−10), [−10,0), [0,10), ≥10 dB를 재사용했다. '
        '정답 전력/다른 정답 전력 합이며 복소 교차항·원 수신 SNR을 뜻하지 않는다. 기종·전력·원기록 '
        '조건이 얽혀 있어 원인을 확정하지 않는다. 전체 조합·기종×전력·모든 성분 동시 성공 수는 JSON에 보존한다. '
        '1성분은 원래 epoch 보고서에 유지하며 이 다중 성분 진단에서 제외했다. 기존 모델 선택 규칙은 바꾸지 않았다.', '']
    watch.write(output.with_suffix('.md'), '\n'.join(lines))
    print(dict(status=result['status'], comparisons=len(comparisons), common_epoch=snapshot['common_epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
