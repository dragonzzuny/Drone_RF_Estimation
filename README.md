# Drone RF Estimation

혼합 복소 I/Q에서 1–3개 원기록의 기여 파형을 복원하고, 학습하지 않은 기록·기종·조합에 대한 일반화를 검증하는 연구입니다.

## 현재 진행 — 2026-10-10 00:31 KST

**기존 최선 U-Net과 성분 상호작용 출력층을 추가한 U-Net의 GPU 본 비교를 시작했습니다.** 기존 U-Net 대조군 e1을 학습 중입니다. 두 군 모두 같은 부모·전체 본체·원 손실·AdamW 1e-5·자료 일정·실효 batch 32·각 추가 3 epoch/225업데이트를 사용합니다. 첫 검증 630개에서 부모 성능의 재현을 확인했습니다.

추가층은 37,888파라미터입니다. 시작 예측을 정확히 보존하며 세 성분 사이의 보정 합을 0으로 만듭니다. 정답 개수·기종·원파형은 추론 입력으로 사용하지 않습니다. 본 비교의 완료 epoch는 아직 없습니다.

[현재 목표와 계획](docs/PLAN_KO.md) · [본 비교 규약](reports/2026-10-10/SOURCE_INTERACTION_MAIN_PLAN_KO.md) · [epoch별 결과](reports/2026-10-10/SOURCE_INTERACTION_PROGRESS.md) · [구조·수식·CPU 검사](reports/2026-10-10/SOURCE_INTERACTION_PLAN_KO.md)

## 현재 보존 중인 최선

이전에 학습한 native RF U-Net 선택 e2입니다. 아래 수치는 같은 개발검증의 결과입니다. 이전 학습 이력이 더 길어 새 초기화 구조 대조와 구분합니다.

| 추론 조건 | 2성분 NMSE ↓ | 3성분 NMSE ↓ | 2성분 복소 SI-SDR ↑ dB | 3성분 복소 SI-SDR ↑ dB |
|---|---:|---:|---:|---:|
| 단일 추론 |0.466896|0.690253|3.777|−3.109|
| 네 위상 평균 |0.446896|0.689089|4.174|−2.784|

네 위상 평균은 네 번의 추론과 예측끼리의 순서 정렬을 사용합니다. 세 성분 약신호 NMSE는 이 조건에서도 **0.948934**로 정밀 복원이 남은 과제입니다. 완벽 복원·범용 드론 분리·기종 식별 향상을 입증한 결과로 표시하지 않습니다. [추론 조건과 전체 결과](reports/2026-10-09/NATIVE_RF_INFERENCE_DIAGNOSIS.md).

## 이번에 완료한 비교와 검토

- **손실·학습률 비교 완료:** 기존/log1p 손실을 학습률 5e-4와 1e-4에서 각각 총 3 epoch까지 확인했습니다. 낮은 학습률의 선택은 기존 e1·log1p e2입니다. 어느 군도 시작 모델의 2·3성분 NMSE와 SI-SDR을 모두 개선하지 못했습니다. [모든 epoch·개수·성분별 결과](reports/2026-10-09/LOW_LR_CONTINUATION_FINAL.md), [전체 학습 곡선 PDF](reports/2026-10-09/LOW_LR_CONTINUATION_EPOCHS.pdf), [체크포인트·optimizer 검산](reports/2026-10-09/LOW_LR_CONTINUATION_FINAL_AUDIT.json).
- **새 출력층의 GPU 사전 검사 완료:** 같은 TRAIN4·각 32업데이트에서 NMSE 2/3는 기존 0.073123/0.368011, 후보 0.072147/0.367325였습니다. 작게 개선됐지만 고정 학습 사례의 적합도이며 새 기록 성능은 아닙니다. 검사 가중치는 폐기했습니다. [전체 단계와 실행 시간](reports/2026-10-10/SOURCE_INTERACTION_PREFLIGHT_AUDIT.md).
- **세 신호 선행·수학 검토:** IQUMamba-1D, SepFormer, SepTDA, ADS-B SplitNet-3 등의 입력·출력 목표를 구분했습니다. 출력 투영과 PIT의 구현을 검산했습니다. [원문 근거·수식·대조 설계](reports/2026-10-09/THREE_SOURCE_DESIGN_REVIEW_KO.md).
- **시간 문맥과 잔차 진단:** 평균 문맥은 호핑 순서를 없애지만 TCN의 경계 위치 영향은 남았습니다. 약신호 잔차 대부분은 창 전체의 상수 배율·위상·DC 보정으로 제거되지 않았습니다. [문맥 해석](reports/2026-10-10/THREE_SOURCE_RESEARCH_UPDATE_KO.md), [전체 배율 진단](reports/2026-10-09/CURRENT_AFFINE_RESIDUALS.md).

## 자료와 평가 원칙

RFUAV 한 데이터셋의 같은 원 RF 대역끼리 합성하며 native 100MS/s, 수신 중심 간격, 원기록 단위 분할을 유지합니다. 명시적 조종기·결합 라벨은 제외하고 송신 방향의 불확실성은 공개합니다. 영상 여부는 편입 조건이 아닙니다.

현재 학습은 8개 원기록 묶음, 개발검증은 5개 묶음입니다. 두 성분은 4개 기종 조합, 세 성분은 한 조합에 제한됩니다. 630창을 630개 독립 기록으로 세지 않습니다. Autel과 예약 확인용 파일 6개는 미개봉이며 모델 선택에 사용하지 않습니다. 합성 기록 수를 물리 드론 대수로 바꾸지 않습니다.

[자료 정책](docs/DATA_POLICY_KO.md) · [자료 선정·조합 명세](docs/DIVERSITY_MIXTURE_PLAN_KO.md) · [RFUAV 37범주 검토](reports/2026-10-07/RFUAV_SCOPE_CORRECTION_KO.md)

## 이전 비교

실패 결과를 포함한 원 규모 WaveNet·U-Net, 긴 복소 문맥, 출력 표현, 기울기, 전력 조건, 기종 구분의 결과는 [직전 계획과 보고서 목록](docs/PLAN_HISTORY_THROUGH_20261010_0029_KO.md)에 보존했습니다. 현재 NMF 없는 U-Net 성과를 Deep NMF 개선으로 표시하지 않습니다. 기존 detector 및 DR-NMF D2는 중단 지시를 유지합니다.

실행 상태는 문서의 기록 시점 기준입니다. 원자료·체크포인트·인증정보는 저장소에 포함하지 않습니다.
