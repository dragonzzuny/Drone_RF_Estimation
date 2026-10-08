# 원 주파수 배치 실험의 최신 상태

본 학습5 epoch·375업데이트와 후속 위상 평균·혼합 적합도 진단을 완료했다. 선택 모델은 e2다.

- [전체 결과와 조건별 수치](NATIVE_RF_FINAL.md)
- [학습 곡선 PDF](NATIVE_RF_EPOCHS.pdf)
- [위상 평균·혼합 적합도 진단](NATIVE_RF_INFERENCE_DIAGNOSIS.md)
- [CPU 개수·정답 보조 진단](NATIVE_RF_CPU_DIAGNOSTICS.md)
- [기울기 진단과 다음 직접 대조](NATIVE_RF_GRADIENT_DIAGNOSIS.md)

개수 손실의 공통 인코더 기울기를 차단하는 동일 예산 후속 GPU 학습을 시작했다. 구체적인 실행 상태는 worker의 상태·epoch 결과를 확인한다. 중복 보고서·그림 미리보기의 삭제 이력은 [정리 기록](CLEANUP_LEDGER.json)에 보존했다.
