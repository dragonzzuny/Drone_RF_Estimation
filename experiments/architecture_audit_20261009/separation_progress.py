"""Supplementary all-source diagnostics from already saved validation metrics.

No inference, raw data access, checkpoint selection, or training modification.
Thresholds describe the development rows; they are not success criteria for
publication. In particular, positive SI-SDR improvement is not clean recovery.
"""
import argparse
import math
from pathlib import Path
import statistics

import watch_epochs as watch


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError('Empty group')
    for row in rows:
        k = row['count']
        if any(len(row[field]) != k for field in
               ('nmse', 'si_sdr', 'input_si_sdr', 'si_sdr_gain')):
            raise ValueError('Incomplete source metrics')
        for out, inp, gain in zip(row['si_sdr'], row['input_si_sdr'], row['si_sdr_gain']):
            if any(v is None or not math.isfinite(v) for v in (out, inp, gain)):
                raise ValueError('Nonfinite multi-source metric')
            if not math.isclose(out - inp, gain, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError('SI-SDR improvement mismatch')
    weakest = [r['weakest_index'] for r in rows]
    return dict(cases=n,
        mean_nmse=statistics.mean(v for r in rows for v in r['nmse']),
        mean_si_sdr=statistics.mean(v for r in rows for v in r['si_sdr']),
        mean_input_si_sdr=statistics.mean(v for r in rows for v in r['input_si_sdr']),
        mean_si_sdr_gain=statistics.mean(v for r in rows for v in r['si_sdr_gain']),
        all_sources_gain_gt_1e_6_db=sum(min(r['si_sdr_gain']) > 1e-6 for r in rows),
        all_sources_absolute_si_sdr_gt_zero_db=sum(min(r['si_sdr']) > 0 for r in rows),
        all_sources_nmse_lt_one=sum(max(r['nmse']) < 1 for r in rows),
        weakest_si_sdr=statistics.mean(r['si_sdr'][i] for r, i in zip(rows, weakest)),
        weakest_si_sdr_gain=statistics.mean(r['si_sdr_gain'][i] for r, i in zip(rows, weakest)),
        weakest_nmse=statistics.mean(r['nmse'][i] for r, i in zip(rows, weakest)),
        predicted_count_histogram={str(k): sum(r['predicted_count'] == k for r in rows)
                                   for k in (1, 2, 3)})


def collect(run):
    snapshot = watch.snapshot(run)
    result = dict(protocol_sha256=snapshot['protocol_sha256'],
        reported_at=snapshot['reported_at'], common_epoch=snapshot['common_epoch'],
        dataset='RFUAV native same-band; development validation; 210 cases each count',
        thresholds='descriptive only; positive gain >1e-6 dB, positive absolute SI-SDR >0 dB, NMSE<1',
        limitations='One seed; reused development validation; correlated windows; no hypothesis test; no claim of clean recovery',
        forward_uses_references=False, evaluation_uses_references_for_permutation=True,
        heldout_read=False, histories={})
    for arm, history in snapshot['histories'].items():
        records = []
        for epoch in history:
            path = run / arm / f"VALIDATION_{epoch['epoch']:03d}.json"
            rows = watch.read(path)['rows']
            groups = [dict(count=k, **summarize([r for r in rows if r['count'] == k]))
                      for k in (2, 3)]
            for value, stored in zip(groups, epoch['by_count'][1:]):
                for key in ('mean_nmse', 'mean_si_sdr', 'weakest_nmse'):
                    if not watch.close(value[key], stored[key]):
                        raise ValueError('Aggregate does not match audited validation')
            records.append(dict(epoch=epoch['epoch'], validation_sha256=watch.digest(path),
                                by_count=groups))
        result['histories'][arm] = records
    return result


def markdown(result):
    lines = ['# 모든 성분의 복원 향상 여부: 저장된 검증 결과 보조 집계', '',
        f"확인: {result['reported_at']} / 세 모델 공통 완료: {result['common_epoch']} epoch", '',
        'RFUAV 동일 대역·원 수신 중심 간격을 보존한 개발 검증 자료다. '
        '각 혼합 개수별210개, seed0이며 학습하지 않은 별도 최종 시험 결과는 아니다. '
        '이미 저장된 파형 평가값만 읽었고 학습·선택 규칙을 바꾸지 않았다.', '',
        '「모두 향상」은 혼합 입력을 그대로 각 정답과 비교했을 때보다 '
        '모든 성분의 복소 SI-SDR이1e-6dB보다 많이 향상된 사례 수다. '
        '이는 혼합 대비 개선이며 깨끗한 복원을 뜻하지 않는다. '
        '「모두 양수」는 모든 출력의 절대 SI-SDR>0dB인 사례 수다. '
        '「모두 NMSE<1」은 각 성분이 영 출력보다 I/Q 제곱 오차가 작은 사례 수다. '
        '세 문턱은 추가 설명용이고 사후 채택 기준으로 쓰지 않는다.', '',
        '| 모델 | epoch | 성분 수 | 모두 향상/210 | 모두 양수/210 | 모두 NMSE<1/210 | 약한 성분 SI-SDR dB | 약한 성분 SI-SDR 향상 dB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, history in result['histories'].items():
        for record in history:
            for g in record['by_count']:
                lines.append(f"| {watch.NAMES[arm]} | {record['epoch']} | {g['count']} | "
                    f"{g['all_sources_gain_gt_1e_6_db']} | {g['all_sources_absolute_si_sdr_gt_zero_db']} | "
                    f"{g['all_sources_nmse_lt_one']} | {g['weakest_si_sdr']:.3f} | {g['weakest_si_sdr_gain']:.3f} |")
    lines += ['', '모델은 정답 없이 고정된 세 성분 슬롯과 배경 슬롯을 낸다. '
        '평가에서는 정답과 출력의 순서를 한 구간 전체에 걸쳐 맞춘다. '
        '이 순열 매칭은 출력의 기체 식별이나 구간 간 추적을 입증하지 않는다. '
        '개수 머리의 결과는 별도 진단이고, 파형 지표를 낼 때 개수 예측으로 출력을 잘라내지 않는다. '
        '따라서 이 표만으로 자동 개수 결정부터 기체 식별까지 완성됐다고 해석할 수 없다.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run.resolve())
    watch.write(args.output.with_suffix('.json'), result)
    watch.write(args.output.with_suffix('.md'), markdown(result))
    print(f"Audited saved metrics through common epoch {result['common_epoch']}")
