# 두 창 지도학습과 추가 일관성: 같은 예산 비교

RFUAV 같은 원 RF 대역·native100MS/s·원기록 묶음 분할·seed0·기존 부모 전체32.14M을 유지했다. 각 군은 같은2400혼합의4800창,75업데이트이며 각 쌍당3forward/2backward다. 새 기록4800개나 기존 단일 창 대조와 같은 계산량을 뜻하지 않는다.

| 군 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB | 선택 epoch |
|---|---:|---:|---:|---:|---:|
| parent | 0.466896 | 3.777 | 0.690253 | -3.109 | 0 |
| paired_supervision | 0.483732 | 3.493 | 0.696766 | -3.518 | 0 |
| paired_consistency | 0.489281 | 3.502 | 0.698193 | -3.229 | 0 |

수치는 실제 e1이며, 선택이e0여도 학습 결과를 숨기지 않는다.
- paired_supervision의 사전 공동 개선 기준: 미충족
- paired_consistency의 사전 공동 개선 기준: 미충족

추가 일관성 항은 새 정답 정보가 아니라 창 의존성에 대한 가중치 변경이다. 원래 개발630혼합 전체를 평가했고 Autel·예약 확인6파일은 읽지 않았다. 반복 사용한 DEV와 한seed의 결과여서 독립 일반화 확인을 대신하지 않는다.

[계획](PAIRED_WINDOW_PLAN_KO.md) · [수식·선행](PAIRED_WINDOW_METHOD_KO.md) · [모든 수치](PAIRED_WINDOW_RESULT.json) · [최종 검산](PAIRED_WINDOW_AUDIT.json)
