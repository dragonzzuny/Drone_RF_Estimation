"""Combine the completed, bounded diagnoses without changing training."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'reports/2026-10-11'


def read(path):
    return json.loads(path.read_text())


def main():
    power_path = PUBLIC / 'DATA_POWER_AUDIT.json'
    branch_path = PUBLIC / 'BRANCH_AUDIT_E020.json'
    power, branch = read(power_path), read(branch_path)
    assert power['status'] == branch['status'] == 'COMPLETE_CHECKED'
    groups = defaultdict(list)
    for row in branch['rows']:
        if row['role'] == 'train_pack':
            groups[(row['mode'], row['count'])].append(row)
    for row in branch['prior_rows']:
        if row['model'] in ('parent/e0', 'septda_e1'):
            groups[(row['model'], row['count'])].append(row)
    summaries = []
    for (mode, count), rows in sorted(groups.items()):
        assert len(rows) == 2
        summaries.append(dict(mode=mode, count=count, cases=2,
            nmse=mean(x for row in rows for x in row['nmse']),
            si_sdr=mean(x for row in rows for x in row['si_sdr'])))
    folder = ROOT / 'local/septda_continuation_20261010_v1/septda'
    base_path, current_path = folder / 'VALIDATION_000.json', folder / 'VALIDATION_026.json'
    base, current = read(base_path), read(current_path)
    e20_path = folder / 'VALIDATION_020.json'
    e20 = read(e20_path)
    assert hashlib.sha256(e20_path.read_bytes()).hexdigest() == read(folder / 'EPOCH_020.json')['validation_sha256']
    assert hashlib.sha256(current_path.read_bytes()).hexdigest() == read(folder / 'EPOCH_026.json')['validation_sha256']
    bins = defaultdict(list)
    identity = ('index', 'count', 'categories', 'pack_ids', 'nominal_levels_db', 'reference_power')
    for old, new in zip(base['rows'], current['rows']):
        assert [old[k] for k in identity] == [new[k] for k in identity]
        if old['count'] == 1:
            continue
        contrast = 10 * math.log10(max(old['reference_power']) / min(old['reference_power']))
        band = '0–10 dB' if contrast <= 10 else '10–20 dB' if contrast <= 20 else '>20 dB'
        bins[(old['count'], band)].append((mean(old['nmse']), mean(new['nmse'])))
    changes = [dict(count=count, power_contrast=band, cases=len(values),
                   baseline_nmse=mean(v[0] for v in values), e26_nmse=mean(v[1] for v in values),
                   worse_cases=sum(v[1] > v[0] for v in values))
               for (count, band), values in sorted(bins.items())]
    evidence = [power_path, branch_path, base_path, e20_path, current_path, Path(__file__),
                ROOT / 'experiments/septda_rf_20261010/architecture.py',
                ROOT / 'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/context_model.py',
                ROOT / 'experiments/source_interaction_20261010/train_comparison.py',
                ROOT / 'experiments/rfuav_dense_gated_20261008/models.py']
    result = dict(status='COMPLETE_CHECKED', train6_endpoint=summaries, dev_power_bins=changes,
        dev_e20=e20['by_count'], dev_baseline=base['by_count'],
        context=dict(full_ms=20.8896, waveform_ms=.63872, current_mode='mean',
                     long_context_temporal_order_preserved=False,
                     added_recurrent_branch_input='short-window complex STFT'),
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        training_changed=False, new_optimizer_updates=0, heldout_read=False)
    (PUBLIC / 'DATA_AND_ARCHITECTURE_DIAGNOSIS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 데이터와 구조 진단: RFUAV·SepTDA 참고 후보', '',
        '2026-10-11. 학습 8·검증 5개 원기록 묶음, 같은 RF 대역·native 100 MS/s·seed 0. '
        '원래 50구간 학습을 유지하며 CPU에서 별도 진단했다. 새 optimizer 업데이트와 보류 자료 접근은 없다. '
        'GPU 결과를 재현한 20구간 체크포인트에서 추론 경로를 비교했다. 전체 검증 추세는 26구간까지다.', '',
        '## 데이터에서 확인한 사실', '',
        '현재 학습·검증은 동일한 긴 구간 전력 기반 합성식을 사용한다. 짧은 창별 재정규화 차이는 '
        '현재 실험의 원인으로 제시하지 않는다. 12,000개 합성 예제와 원기록 다양성은 다르며 '
        '창을 독립 녹음으로 세지 않는다. 학습·검증 원기록 묶음의 교집합은 0이다.', '',
        '| 자료 | 혼합 수 | 사례 수 | 실제 최대/최소 전력차 중앙값 | 20 dB 초과 사례 |',
        '|---|---:|---:|---:|---:|']
    for row in power['summaries']:
        lines.append(f"|{row['role']}|{row['count']}|{row['cases']}|{row['contrast_db_quantiles']['p50']:.2f} dB|"
                     f"{row['contrast_above_20_db']}/{row['cases']} ({100*row['contrast_above_20_db']/row['cases']:.1f}%)|")
    lines += ['', '학습은 결과를 보기 전에 seed 20261011로 일정별·개수별 16개씩 뽑은 240사례이고 '
              '검증은 고정 630사례 전체다. 전력차는 잡음 대비 SNR이 아니다. '
              '국소 전력이 작다는 사실만으로 잡음 전용 구간이라고 표시하지 않는다. '
              '학습 표본에도 큰 전력차가 존재해 검증에서만 약신호가 등장했다는 설명은 지지되지 않는다.', '',
              '## 전력차별 검증 오차', '',
              '| 혼합 수 | 국소 전력차 | 사례 수 | 기준 NMSE | e26 NMSE | 악화 사례 |',
              '|---|---|---:|---:|---:|---:|']
    for row in changes:
        lines.append(f"|{row['count']}|{row['power_contrast']}|{row['cases']}|{row['baseline_nmse']:.6f}|"
                     f"{row['e26_nmse']:.6f}|{row['worse_cases']}|")
    lines += ['', '분석한 모든 전력차 집단에서 평균 NMSE가 증가했다. '
              '큰 전력차만으로 현재 악화 전체를 설명하지 않는다. 같은 DEV 반복 분석이며 통계적 독립 표본 검정은 아니다.', '',
              '## 실제 구조의 시간 정보', '',
              '현재 생성 경로는 retained_unet → build(unet_mean) → ContextualSeparator(context_mode=mean)이다. '
              '약 20.8896 ms 문맥 특징은 시간 평균 뒤 반복되어 입력된다. 따라서 긴 문맥의 시간 순서는 제거된다. '
              'SepTDA의 LSTM·attention 경로는 약 0.63872 ms의 복소 STFT를 처리한다. '
              '짧은 창 안의 시간 구조는 보지만 긴 호핑·패킷 반복 순서까지 보존하는 구조는 아니다.', '',
              '출력은 세 복소 신호와 배경이며 합 일치 투영을 적용한다. 정답도 같은 합을 만족하므로 '
              '이 출력 제약 자체가 세 신호 복원을 수학적으로 금지하지 않는다. '
              '초기 추가 경로의 출력이 0일 때 기준 모델과 동일했던 기존 검사는 보존되어 있다. '
              '현재 공동 학습된 본체에서 추가 경로를 끄는 것은 원래 기준 가중치로 되돌리는 것과 다르다.', '',
              '## 고정 TRAIN6의 e20 추론 비교', '',
              '| 조건 | 혼합 수 | 사례 수 | NMSE | 복소 SI-SDR dB |',
              '|---|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"|{row['mode']}|{row['count']}|{row['cases']}|{row['nmse']:.6f}|{row['si_sdr']:.3f}|")
    lines += ['', '선택된 두·세 신호 훈련 사례에서는 기준보다 NMSE와 SI-SDR이 함께 개선됐지만, '
              '동일 e20의 DEV630에서는 두·세 신호 NMSE가 각각 0.466896→0.565789, '
              '0.690253→0.708695로 증가했다. 일반화 문제의 단서이며, '
              '전체 훈련 성능을 측정한 결과나 과적합 원인 확정은 아니다. '
              '추가 경로를 켜고 끈 훈련 NMSE 차이는 두 신호 약 0.00157, 세 신호 약 0.00042로 작았다.', '',
              '이 여섯 사례는 기존 TRAIN48에서 각 개수별 처음 두 사례로 고정했다. '
              '원 규모 모델과 원 창 길이를 유지했다. 기준/e1의 저장 결과를 재사용하고 참조 전력을 검산했다. '
              '현재 e20의 CPU 결과는 개수별 한 DEV 사례에서 기존 GPU 결과와 일치했다. '
              '소수 훈련 사례로 일반화나 원인을 확정하지 않으며, 경로 끄기는 별도 학습 구조 대조가 아니다.', '',
              '## 해석과 후속 우선순위', '',
              '데이터는 원녹음 다양성·국소 전력·수신 잡음 측면의 제약이 있고, 현재 구조는 긴 시간 순서를 버린다. '
              '둘 중 하나를 단독 원인으로 확정할 근거는 없다. 독립 수신 잡음까지 원래 녹음별로 '
              '완벽 분배하는 목표와 실제 드론 신호 복원을 구별해야 하며, 현재 오차가 잡음 한계라는 증거는 없다.', '',
              '기존 비교를 보존하고, 다음 설계에서는 같은 데이터셋·대역의 원녹음 다양성 확대와 '
              '실제 활동/전력 조건을 기록한 혼합, 긴 시간 순서를 보존한 경로를 우선 검토한다. '
              '대기 중인 ordered-context 후보는 긴 문맥 순서 전달을 검증하는 별도 후보다. '
              '이 진단으로 50구간 실행 순서를 자동 변경하거나 새로운 학습을 시작하지 않았다.']
    (PUBLIC / 'DATA_AND_ARCHITECTURE_DIAGNOSIS_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='COMPLETE_CHECKED', train6_endpoint=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
