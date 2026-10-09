# 주파수 배열 전체 검증: 완료

2026-10-09 16:06 KST에 CPU 비교가 정상 종료했다. 고정 U-Net e3/225업데이트의 기존 개발검증 630개를 원래 배열과 중앙 정렬 배열로 평가했다. 두 방식 모두 같은 CPU를 사용했으며 새 학습과 보류 자료 조회는 없었다. 실제 추론 비교에는 2시간 17분 25초가 걸렸다.

630개 원래 추론의 CPU/GPU 점수를 허용 오차 안에서 재현했고 활성 출력 대응은 모두 같았다. 중앙 정렬은 2·3성분의 평균 NMSE와 SI-SDR을 소폭 개선했다. 큰 잔차를 해결한 결과이거나 최종 선택 e4에서도 개선된다는 뜻은 아니다. [완료 결과](FREQUENCY_ORDER_VALIDATION.md), [행별 수치](FREQUENCY_ORDER_VALIDATION.json).

실행 위치: `local/frequency_order_validation_20261009_v1`. 코드: [frequency_order_validation.py](../../experiments/architecture_audit_20261009/frequency_order_validation.py). 등록 규약 SHA-256: `c6698790982f92f3f58c680e01e0657353e6b3b6ce8c0155bd5a4d3b33fd59f9`. 독립 재집계에서 전체 행의 평균·소스 및 스냅샷 해시를 다시 검산했다.
