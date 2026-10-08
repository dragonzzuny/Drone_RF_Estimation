"""Reaggregate saved local/global fusion epochs; no I/Q or GPU evaluation."""
import argparse
import json
from pathlib import Path
from summarize_waveform_context import summarize, audit_completion

ARMS = ('local_only', 'local_global')


def markdown(result):
    lines = ['# 국소 I/Q 복원 + 긴 복소 문맥 결합: 결과', '',
        f"갱신: {result['updated_at']} · {result['status']}", '',
        'RFUAV 개발 검증 630혼합(1·2·3성분 각각 210개), 같은 원 RF 대역 내 중심 정렬 합성.',
        '동일 원기록 그룹 분할·동일 학습 목록, seed 0. 보류 Autel 원기록은 읽지 않았다.',
        '두 군 모두 33,734,467파라미터이며 기존 짧은 문맥 e5(1,500업데이트)를 공통 부모로 이어받았다.',
        '아래 epoch는 추가 학습이다. 동일 12,000혼합 목록을 다시 사용하며 각 5 epoch·1,500업데이트, effective batch 8/microbatch 4다.',
        '국소 복원기는 같은 0.63872ms 복소 입력을 받는다. 추가 경로만 창 밖을 가린 입력(local_only)과 10.48576ms 전체 I/Q(local_global)로 나뉜다.',
        '양쪽 모두 기존 약 21ms 전력 특징의 시간 평균을 보조 정보로 받는다. 개수 헤드는 이 기존 전력 특징을 사용한다.',
        '복소 문맥은 pack16→4단계 학습 stride4→256시간 토큰→7단계 dilated TCN을 거쳐, U-Net의 세밀한 64채널·압축된 1024채널 특징을 조절한다.',
        '기존 경로 학습률은 cosine 1e-5→1e-6, 추가 경로는 같은 일정의 10배다. 두 군에 동일하게 적용했다.', '',
        '| 군 | 추가 epoch | 학습 손실 | 2성분 NMSE ↓ | 3성분 NMSE ↓ | 2성분 복소 SI-SDR dB ↑ | 3성분 복소 SI-SDR dB ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for row in result['arms'][arm]['epochs']:
            g = {v['count']: v for v in row['metrics']['by_count']}
            lines.append(f"| {arm} | {row['epoch']} | {row['average_loss']:.6f} | {g[2]['mean_nmse']:.6f} | {g[3]['mean_nmse']:.6f} | {g[2]['mean_si_sdr']:.3f} | {g[3]['mean_si_sdr']:.3f} |")
    lines += ['', f"공통 완료: 추가 {result['common_completed_epoch']} epoch / 각 {result['updates_per_arm']}업데이트.",
        '선택은 초기 추가 epoch 0을 포함한 2·3성분 평균 NMSE 최솟값이다. 아래 표는 공통 예산 내 선택값이다.', '',
        '| 군 | 선택 추가 epoch | 1성분 NMSE | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR | 3성분 SI-SDR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        row = result['selected_at_common_budget'][arm]
        g = {v['count']: v for v in row['by_count']}
        lines.append(f"| {arm} | {row['epoch']} | {g[1]['mean_nmse']:.6f} | {g[2]['mean_nmse']:.6f} | {g[3]['mean_nmse']:.6f} | {g[2]['mean_si_sdr']:.3f} | {g[3]['mean_si_sdr']:.3f} |")
    outcome = '충족' if result['long_context_meets_prespecified_acceptance'] else '미충족'
    lines += ['', f'긴 문맥 정보의 사전 채택 기준(2·3성분 각각 NMSE 감소와 SI-SDR 증가): **{outcome}**.',
        '모든 저장 평가 행을 재집계했고 사례·참조 전력·공통 업데이트 예산을 확인했다. 불리한 조건이나 아주 약한 참조를 삭제하지 않는다.',
        '복소 NMSE는 정답 배율·위상으로 보정하지 않는다. 복소 SI-SDR은 평균 제거와 상수 복소 배율(전체 위상 포함)을 허용한다.',
        '학습 손실에는 참조 전력의 수치 바닥값이 있지만 보고 NMSE는 양의 참조 전력을 그대로 쓴다. 두 수치를 같은 척도로 비교하지 않는다.',
        '이 결과는 한 seed의 추가 학습 개발 대조다. 독립 미지 기종·실제 동시 수신·물리 드론 수의 증거가 아니며, 단순한 파라미터 증가의 효과와도 구분한다.',
        '원 중심주파수 간격 미보존, VTSBW 분할 차이, 수신 잡음 포함 정답, 문맥 가림에 따른 추가 경로 정규화 변화가 남는다.',
        '특징의 배율·편향을 문맥으로 조절하는 방식은 [Perez 외의 FiLM](https://arxiv.org/abs/1709.07871)을 참고한 응용이다. 이 원 논문의 시각 추론 결과를 드론 복원 성능 근거로 옮기지 않는다.', '']
    if 'completion_audit' in result:
        lines += ['완료 검산: 소스·스냅샷·학습 목록·공통 부모 해시, 최종 선택 사본, 원 규모 가중치의 유한성, 양쪽 추가 1,500업데이트를 확인했다.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-completion', action='store_true')
    args = parser.parse_args()
    result = summarize(args.run, ARMS)
    if args.verify_completion:
        result['completion_audit'] = audit_completion(args.run, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps({k: result[k] for k in ('status', 'common_completed_epoch', 'updates_per_arm',
        'long_context_meets_prespecified_acceptance', 'all_rows_reaggregated')}))
