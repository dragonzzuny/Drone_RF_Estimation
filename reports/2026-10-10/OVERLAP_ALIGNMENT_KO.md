# 창 간 출력 정렬과 분리 오차의 구분

기존 TRAIN6의 모든 사례를 다시 사용한 사후 진단이다. 같은 부모 모델을 CPU에서18회 추론했다. 정답 없이 MSE로 정렬, 정답 없이 복소 coherence로 정렬, 정답 PIT 대응을 사용한 진단용 정렬을 비교했다. 마지막 조건은 실제 적용 가능한 방법이 아니다.

| 정렬·병합 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB |
|---|---:|---:|---:|---:|
| central / none | 0.374493 | 9.903 | 0.636257 | 0.126 |
| prediction_mse / uniform | 0.591301 | 8.362 | 0.633347 | 0.089 |
| prediction_mse / center_weighted | 0.531229 | 6.439 | 0.637494 | 0.030 |
| prediction_coherence / uniform | 0.591984 | 8.433 | 0.633347 | 0.089 |
| prediction_coherence / center_weighted | 0.531827 | 6.413 | 0.637494 | 0.030 |
| oracle_reference_assignment_diagnostic_only / uniform | 0.305468 | 7.102 | 0.593745 | 0.136 |
| oracle_reference_assignment_diagnostic_only / center_weighted | 0.310756 | 9.107 | 0.608428 | -0.731 |

복소 상관으로 바꿔도 두·세 성분의 공동 개선은 없었다. 정답으로 순서를 알려준 단순 평균에서는 두·세 NMSE가 중앙 단일 창보다 낮아졌지만 두 성분 SI-SDR은 악화했다. 따라서 정렬 오류가 일부 영향을 주지만 순서만 완벽히 알면 정밀 분리가 해결된다고 해석할 수 없다.

42개 중앙/평균 행,54개 개별 창 행,21개 집계를 검산했다. 이전 MSE 평균은 같은 값으로 재현됐다. 주 표는 중앙 창의 정답 대응을 고정하며, 평균 결과의 전역 출력 순서를 다시 PIT로 평가한 수치는 검산 JSON에 함께 보존했다. 검산에서 원파형 추론을 다시 수행한 것은 아니다.

이번 판단은 고정 TRAIN6의311.04μs 공통 구간에 한정한다. 부모 DEV 성능이나 새로운 독립 평가를 대체하지 않는다. 상관 기반 정렬을 새 최선으로 채택하지 않는다. 겹치는 두 창의 지도학습과 추가 일관성을 구분하는 후속 가설을 유지한다.

[전체 결과](OVERLAP_ALIGNMENT_RESULT.json) · [검산 및 독립 PIT 점수](OVERLAP_ALIGNMENT_AUDIT.json) · [기존 창 비교](WINDOW_OVERLAP_KO.md) · [후속 학습 규약](PAIRED_WINDOW_PLAN_KO.md)
