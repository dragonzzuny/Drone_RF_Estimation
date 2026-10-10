# SepTDA 참고 후보의 구현·사전 검사

기존 U-Net 32,142,859개와 새 분리부 22,453,248개를 합쳐 총 **54,596,107파라미터**다. 원문의 주요 분리부 규모인 128차원, 방향별 LSTM 256, attention 4 head, attractor 2층, triple-path 8블록을 사용했다. 복소 STFT 입출력·잔차 결합·기존 개수 head와 손실은 RF 비교에 맞게 유지한 적용 후보다.

CPU에서 실제 TRAIN 1·2·3개 혼합, 원래 63,872 복소 표본 길이로 검사했다. 세 초기 출력은 부모와 bitwise 일치했다. chunk overlap-add의 값·기울기 복원, 출력 합 일치, 전체 유한 gradient를 확인했다. 0 readout일 때 query gradient는 0이고, readout을 진단용으로 활성화하면 query·인코더·초기/최종 시간 처리·신호 간 attention 모두에 비영 gradient가 전달됐다. 진단 후 초기 상태를 복구했고 optimizer 업데이트는 0회다.

이 결과는 구현 연결의 확인이며 복원 성능이나 일반화의 성과가 아니다. 다음 GPU 비교는 선행 시간 문맥 실험의 완료와 검산을 확인한 뒤 수행한다.

[검사 수치·지문](SEPTDA_RF_CPU_CHECK.json) · [규약](SEPTDA_RF_PLAN_KO.md) · [구조 그림과 차이](../../experiments/septda_rf_20261010/README.md)

GPU의 직접 계산과 실제 학습용 중첩 재계산을 비교해 출력 일치 허용오차와 전체 gradient 상대 L2 오차 1.56×10⁻⁷를 확인했다. GPU 검사 최대 할당량은 약 2.28 GiB이며 본학습 optimizer 상태를 포함한 최대치는 아니다. 진단용 readout을 0으로 복구했고 검사 중 optimizer 업데이트는 0회다. [GPU 검산](SEPTDA_RF_GPU_CHECK.json).
