# 주파수 배열 전체 검증: 실행 상태

2026-10-09 13:50 KST에 CPU worker PID3962166을 확인했다. 학습이 아니라 고정 U-Net e3/225업데이트의 기존 개발검증630개에 대한 추론 비교다. 원래 배열과 중앙 정렬 배열을 모두 같은 CPU로 평가하고, 원래 추론의 전체 점수를 기존 GPU 결과와도 검산한다. 보류 자료는 열지 않는다.

원래 배열/중앙 정렬 각각 630회이므로 TRAIN48의 측정 속도로 약 2시간 20분을 예상한다. CPU12/13·낮은 우선순위로 GPU 학습과 병행한다. 원 규모·가중치·입력·출력은 동일하며 새 모델 학습·원 규약의 체크포인트 변경은 없다.

실행 위치는 `local/frequency_order_validation_20261009_v1`, 코드: [frequency_order_validation.py](../../experiments/architecture_audit_20261009/frequency_order_validation.py). `STATE.json`에 10혼합마다 진행을 남기고 완료 시 `FREQUENCY_ORDER_VALIDATION.md/json`을 출력한다. 중간 행을 완료 수치로 간주하지 않는다.

등록 규약 SHA-256: `c6698790982f92f3f58c680e01e0657353e6b3b6ce8c0155bd5a4d3b33fd59f9`. [선행 TRAIN 진단](FREQUENCY_ORDER_TRAIN.md).
