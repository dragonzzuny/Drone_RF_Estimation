"""Reaggregate completed phase-context development rows; no waveform reads."""
import argparse
import math
from pathlib import Path
import statistics as stats

import watch_epochs as watch


def power_gap(row):
    return 10 * math.log10(max(row['reference_power']) / min(row['reference_power']))


def aggregate(rows, label):
    return dict(stratum=label, cases=len(rows),
        mean_nmse=stats.mean(v for r in rows for v in r['nmse']),
        median_case_nmse=stats.median(stats.mean(r['nmse']) for r in rows),
        mean_si_sdr=stats.mean(v for r in rows for v in r['si_sdr']),
        weakest_nmse=stats.mean(r['nmse'][r['weakest_index']] for r in rows),
        weakest_si_sdr=stats.mean(r['si_sdr'][r['weakest_index']] for r in rows))


def run(study, public):
    protocol = watch.read(study/'PROTOCOL.json')
    events = []
    for arm in ('local', 'long'):
        for epoch in range(1, 6):
            receipt = study/arm/f'EPOCH_{epoch:03d}.json'
            if not receipt.exists():
                break
            saved = watch.read(receipt)
            if saved['protocol_sha256'] != watch.digest(study/'PROTOCOL.json'):
                raise ValueError('Receipt from another study')
            path = study/arm/f'VALIDATION_{epoch:03d}.json'
            watch.validate(path, None)
            validation = watch.read(path)
            summaries = []
            for count in (1, 2, 3):
                rows = [r for r in validation['rows'] if r['count'] == count]
                subsets = [('all', rows)]
                if count > 1:
                    subsets += [(label, [r for r in rows if lo <= power_gap(r) < hi])
                                for label, lo, hi in [('<10dB', 0, 10), ('10-20dB', 10, 20),
                                                     ('>=20dB', 20, math.inf), ('<20dB', 0, 20)]]
                for label, subset in subsets:
                    if subset:
                        summaries.append(dict(count=count, **aggregate(subset, label)))
            events.append(dict(arm=arm, epoch=epoch, updates=saved['updates'],
                validation_sha256=watch.digest(path), receipt_sha256=watch.digest(receipt),
                summaries=summaries))
    if not events:
        raise ValueError('No completed epoch')
    result = dict(status='COMPLETE_SNAPSHOT', study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        preparation_sha256=protocol['preparation_sha256'], source_sha256=watch.digest(Path(__file__)),
        events=events, scope='Existing development rows; post-hoc descriptive strata, no excluded cases',
        gap_definition='10log10(max reference crop power/min reference crop power)',
        heldout_read=False, waveform_reads=0, selection_changed=False,
        overlapping_strata='all and <20dB overlap other rows; do not sum their case counts')
    watch.write(public.with_suffix('.json'), result)
    lines = ['# 긴 복소문맥 비교: 국소 전력차별 완료 epoch 진단', '',
        '저장된 개발검증 630행을 재집계했다. 실제 전력차는 복원 창 안에서 가장 강한 정답과 '
        '가장 약한 정답의 전력비다. 명목 합성 전력차와 다를 수 있다. 모든 행을 원래 평균에 유지하며 '
        '아래 층화는 사후 진단이다. `<20dB`와 `all`은 다른 행과 겹치므로 표의 사례 수를 합산하지 않는다.', '',
        '| 군 | epoch | 성분 수 | 실제 전력차 | 혼합 수 | 평균 NMSE ↓ | 혼합별 NMSE 중앙값 | 평균 복소 SI-SDR ↑ dB | 약신호 NMSE ↓ |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for event in events:
        for s in event['summaries']:
            lines.append(f"|{event['arm']}|{event['epoch']}|{s['count']}|{s['stratum']}|{s['cases']}|"
                f"{s['mean_nmse']:.6f}|{s['median_case_nmse']:.6f}|{s['mean_si_sdr']:.3f}|{s['weakest_nmse']:.6f}|")
    lines += ['', '평균과 중앙값의 큰 차이는 일부 큰 오차의 영향을 드러낸다. '
        '이 집계만으로 낮은 활동량, 수신 잡음, 학습 최적화, 합 일치 보정 중 하나를 원인으로 확정하지 않는다. '
        '기존 규약의 모델 선택과 630개 전체 성능 보고는 유지한다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    print(f'Wrote {len(events)} completed epoch snapshots, no waveform reads')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--public', required=True, type=Path)
    a = p.parse_args()
    run(a.study.resolve(), a.public.resolve())
