# Drone RF Estimation

RFUAV의 같은 RF 대역 혼합 I/Q에서 최대3개 원기록의 기여 파형을 복원하는 연구입니다.
원기록을 먼저 나누고 native100MS/s와 수신 중심주파수 차이를 유지합니다.

## 현재 최선 — 2026-10-11

**native e2 복소 STFT U-Net + 8위상 평균 추론**입니다. 같은630개 개발 혼합의
4위상 대8위상 비교와 CPU 검산을 완료했습니다. 가중치는 그대로입니다.

|추론|두 성분 NMSE↓|세 성분 NMSE↓|두 성분 복소SI-SDR↑ dB|세 성분 복소SI-SDR↑ dB|
|---|---:|---:|---:|---:|
|4위상|0.446896|0.689089|4.174|−2.784|
|8위상·선택|0.444601|0.688948|4.249|−2.736|

개선 폭은 작고 추론량은2배입니다. 세 성분 최약NMSE0.948644로 큰 잔차가 남습니다.
완벽 복원·물리 드론 대수·미학습 기종 일반화·드론 식별 향상을 입증한 결과는 아닙니다.
[선택 설정과 근거](docs/CURRENT_BEST_KO.md) · [전체 결과](reports/2026-10-11/PHASE_EIGHT_RESULT_KO.md).

## 완료한 최근 실험

- 별도 시간 변화 경로: 전체2400혼합·75업데이트와 DEV630 평가 완료. 약한 성분 오차가
  악화해 비채택했습니다. 원 U-Net131가중치를 보존하며 새735,232파라미터가 학습된 것을
  확인했습니다. [결과](reports/2026-10-11/ORDERED_BRANCH_RESULT_KO.md).
- TF-GridNet 전체/출력층 적응: 공동 개선 실패로 종료했습니다.
  [전체 적응](reports/2026-10-11/TFGRIDNET_DIVERSITY_RESULT_KO.md),
  [출력층 적응](reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_RESULT_KO.md).
- TF-GridNet4위상 추론: TRAIN30 진단에서 개선했습니다. DEV 최선과 직접 순위를
  매기지 않습니다. [결과](reports/2026-10-11/TFGRIDNET_PHASE_PROBE_RESULT_KO.md).

이번 실험은 모두 종료됐으며 자동 후속 GPU 대기열은 없습니다. 중지한 SepTDA와
기존5후보 대기열,detector,DR-NMF D2는 재개하지 않았습니다.
[현행 계획](docs/PLAN_KO.md) · [이전 실행 기록](docs/README_HISTORY_20261010.md).

## 자료와 해석

TRAIN8개·DEV5개 원기록 묶음이며630창을 독립 기록630개로 세지 않습니다.
명시적 조종기·결합 라벨은 제외했고 기체명 기록 내 송신 방향의 불확실성은 유지합니다.
Autel과 예약 확인 자료는 미개봉입니다. 정답·기종·실제 합성 개수는 추론에 주지 않습니다.
NMSE는 상대 오차 에너지이며 정확도 백분율이 아닙니다. 현재 모델은 NMF를 사용하지 않습니다.

원자료·체크포인트·인증정보는 저장소에 포함하지 않습니다.
[자료 정책](docs/DATA_POLICY_KO.md) · [선정·합성 규칙](docs/DIVERSITY_MIXTURE_PLAN_KO.md).
