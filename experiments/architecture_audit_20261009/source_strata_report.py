"""Post-hoc category/power strata of an already completed architecture study.

Reads saved validation JSON only; never fits models or opens I/Q/checkpoints.
Source SIR means reference-component energy / sum of other reference energies,
not mixture energy with cross terms, transmitter SNR, or physical source count.
"""
import argparse
import collections
import math
from pathlib import Path
import statistics as stats

import watch_epochs as watch


def sir_label(value):
    for upper, label in [(-20, '<-20'), (-10, '[-20,-10)'),
                         (0, '[-10,0)'), (10, '[0,10)'), (math.inf, '>=10')]:
        if value < upper:
            return label
    raise ValueError('Nonfinite SIR')


def aggregate(items):
    if not items:
        raise ValueError('Empty stratum')
    return dict(source_cases=len(items), mixture_cases=len({r['index'] for r in items}),
        mean_nmse=stats.mean(r['nmse'] for r in items),
        median_nmse=stats.median(r['nmse'] for r in items),
        mean_si_sdr=stats.mean(r['si_sdr'] for r in items),
        mean_si_sdr_gain=stats.mean(r['gain'] for r in items),
        fraction_nmse_below_one=stats.mean(r['nmse'] < 1 for r in items),
        fraction_si_positive=stats.mean(r['si_sdr'] > 0 for r in items),
        fraction_gain_positive=stats.mean(r['gain'] > 0 for r in items))


def run(study, public):
    snap = watch.snapshot(study)
    if snap['status'] != 'COMPLETED' or snap['common_epoch'] != 5:
        raise ValueError('Requires completed three-arm five-epoch study')
    protocol = watch.read(study/'PROTOCOL.json')
    result = dict(status='COMPLETE', source_sha256=watch.digest(Path(__file__)),
        protocol_sha256=snap['protocol_sha256'], preparation_sha256=protocol['preparation_sha256'],
        per_arm_budget=dict(epochs=5, updates=375), heldout_read=False, waveform_reads=0,
        optimizer_updates=0, selection_changed=False, independent_test=False,
        interpretation='Post-hoc descriptive strata; correlated crops and category/recording confounding',
        sir_definition='10log10(P_reference / sum(other reference powers)); no cross terms',
        arms=[])
    identities = None
    for arm in protocol['arms']:
        selected = snap['common_selected'][arm]['epoch']
        path = study/arm/f'VALIDATION_{selected:03d}.json'
        summary, identities = watch.validate(path, identities)
        rows = watch.read(path)['rows']
        items = []
        for row in rows:
            k = row['count']
            if k == 1:
                continue
            power = row['reference_power']
            if any(p <= 0 or not math.isfinite(p) for p in power):
                raise ValueError('Invalid reference power')
            for j in range(k):
                sir = 10*math.log10(power[j]/sum(p for i,p in enumerate(power) if i != j))
                vals = [row['nmse'][j], row['si_sdr'][j], row['si_sdr_gain'][j], sir]
                if not all(v is not None and math.isfinite(v) for v in vals):
                    raise ValueError('Nonfinite source metric; no silent exclusion')
                items.append(dict(index=row['index'], count=k, category=row['categories'][j],
                    combination=' + '.join(sorted(row['categories'])),
                    source_sir_db=sir, sir_bin=sir_label(sir),
                    nmse=vals[0], si_sdr=vals[1], gain=vals[2]))
        tables = {}
        for name, fields in [('category', ('count', 'category')),
                             ('combination', ('count', 'combination')),
                             ('power', ('count', 'sir_bin')),
                             ('category_power', ('count', 'category', 'sir_bin'))]:
            groups = collections.defaultdict(list)
            for item in items:
                groups[tuple(item[k] for k in fields)].append(item)
            tables[name] = [dict(zip(fields,key), **aggregate(group))
                            for key,group in sorted(groups.items())]
            if sum(g['source_cases'] for g in tables[name]) != 1050:
                raise ValueError('Strata lost or duplicated source cases')
        for count in (2,3):
            got = aggregate([r for r in items if r['count']==count])
            ref = next(r for r in summary['by_count'] if r['count']==count)
            if not watch.close(got['mean_nmse'], ref['mean_nmse']) or not watch.close(got['mean_si_sdr'], ref['mean_si_sdr']):
                raise ValueError('Source aggregate does not recover registered metrics')
        result['arms'].append(dict(arm=arm, selected_epoch=selected,
            validation_sha256=watch.digest(path), source_cases=1050, mixture_cases=420, **tables))
    watch.write(public.with_suffix('.json'), result)
    lines = ['# 완료된 세 구조: 기종·조합·국소 전력별 파형 성능', '',
        '각5epoch/375업데이트가 끝난 뒤 기존 개발 NMSE 규칙으로 선택된 가중치의 저장 결과를 재집계했다. '
        '새 I/Q·체크포인트를 열거나 모델을 학습하지 않았다. 각 모델의2/3성분420혼합·1,050성분을 모두 유지했다. '
        '원기록5묶음의 상관된 창이며 기종과 기록 조건이 얽혀 있다. 아래 차이를 기종 자체의 인과 효과나 '
        '독립 시험으로 해석하지 않는다. 세 구조의 파라미터 수와 계산량은 다르다.', '',
        'SIR은 해당 정답의 국소 전력/나머지 정답 전력 합의 dB값이다. 복소 교차항을 포함하는 '
        '혼합 전력비나 원 수신 SNR이 아니다. SIR 구간은 사후 기술 통계이며 모델 선택을 바꾸지 않는다.', '']
    for table, title, group in [('category','기종','category'),
                               ('combination','혼합 조합','combination'),
                               ('power','성분별 실제 SIR','sir_bin')]:
        lines += [f'## {title}', '',
            '| 모델 | 선택 e | 성분 수 | 조건 | 성분 사례 | 평균 NMSE ↓ | NMSE 중앙값 | SI-SDR ↑ dB | SI-SDR 개선량 ↑ dB | SI-SDR>0 비율 |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
        for model in result['arms']:
            for row in model[table]:
                lines.append(f"|{model['arm']}|{model['selected_epoch']}|{row['count']}|{row[group]}|"
                    f"{row['source_cases']}|{row['mean_nmse']:.6f}|{row['median_nmse']:.6f}|"
                    f"{row['mean_si_sdr']:.3f}|{row['mean_si_sdr_gain']:.3f}|{row['fraction_si_positive']:.1%}|")
        lines += ['']
    lines += ['기종×SIR의 세부 셀과 NMSE<1·혼합 대비 개선 비율은 같은 이름의 JSON에 모두 포함했다. '
        '세 표는 같은 사례를 다른 기준으로 나눈 것이므로 서로의 사례 수를 합산하지 않는다. '
        '단일 성분 조건은 이 다중 성분 진단의 범위 밖이며 기존 전체 검증 결과에 보존돼 있다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    print('PASS: three selected models, each 420 mixtures/1050 sources, all strata retained')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--public', required=True, type=Path)
    args = p.parse_args()
    run(args.study.resolve(), args.public.resolve())
