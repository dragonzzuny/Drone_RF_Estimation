# 같은 표본의 창 위치·병합 진단 결과

고정 TRAIN6, 원 규모 native 부모 U-Net, CPU 18회 추론이다. 세 창이 공유하는 311.04μs의 동일 I/Q를 평가했다. 신규 학습이나 개발·독립 확인 결과가 아니다. 정답은 지표 계산에만 사용했고 추론 순서 정렬·평균에는 쓰지 않았다.

| 방법 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB |
|---|---:|---:|---:|---:|
| central | 0.374493 | 9.903 | 0.636257 | 0.126 |
| earlier_window | 0.508409 | 8.504 | 0.645797 | -0.349 |
| later_window | 1.129458 | -2.454 | 0.626245 | 0.151 |
| uniform_average | 0.591301 | 8.362 | 0.633347 | 0.089 |
| center_weighted_average | 0.531229 | 6.439 | 0.637494 | 0.030 |

두·세 평균 NMSE 감소, 복소 SI-SDR 증가, 최약 NMSE 비증가의 공동 조건:
- earlier_window: 미충족
- later_window: 미충족
- uniform_average: 미충족
- center_weighted_average: 미충족

정답을 사용한 독립 PIT 대응과 중앙 예측의 고정 대응이 달랐던 행: 4/30. 모든 수치는 고정 대응을 사용한다.

30행의 상대·절대 오차 관계, 전 조건 집계, 코드·체크포인트 해시, 실행 시 동일 표본 검사 기록을 검산했다. 검산에서 파형 추론을 다시 수행한 것은 아니다. 개별 창 이동은 문맥 위치·정규화도 바꾸므로 경계 padding만의 효과로 해석하지 않는다.

평균으로 일부 지표가 좋아져도 고정 TRAIN6의 관찰이며 전체 DEV 성능은 별도 검증 대상이다. 단일 창의 638.72μs 지표나 기존 DEV 표와 직접 섞어 비교하지 않는다.

[실행 전 계획](WINDOW_OVERLAP_PLAN_KO.md) · [선행 근거](WINDOW_OVERLAP_RELATED_WORK_KO.md) · [전체 개별 수치](WINDOW_OVERLAP_RESULT.json) · [검산](WINDOW_OVERLAP_AUDIT.json)
