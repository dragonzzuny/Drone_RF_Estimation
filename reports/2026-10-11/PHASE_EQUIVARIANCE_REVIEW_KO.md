# 위상 평균 관찰과 물리계층 선행 근거

Gansekoele 외, *Relative Phase Equivariant Deep Neural Systems for Physical
Layer Communications*, TMLR,2025. [출판본](https://ir.cwi.nl/pub/36105/36105.pdf),
[기관 서지](https://ir.cwi.nl/pub/36105/),
[저자 코드](https://github.com/awgansekoele/relative-phase-equivariant-deep-neural-systems).

본문3절은 공통 위상 회전을 유한 순환군으로 표현하고, 위상별 특징을
순환 convolution으로 처리한다.4절 실험은 **OFDM·송신안테나1/수신안테나2**,
pilot와 LS 채널 추정 입력을 쓰는 수신기다. 최종 출력은 bit LLR이다.
AdamW150epochs, 초기1e-3,100/125epoch에서1/10,10반복과 BER 평가를 사용한다.
따라서 정답 pilot가 없는 단일채널 드론3기록의 복소 파형 분리 성능이나
우리 추가1epoch의 충분한 수렴 근거로 사용하지 않는다.

우리 코드에 대한 해석은 별도다. 복원기는 입력의 공통 위상을 돌리면 출력
I/Q도 같은 만큼 돌아가는 것이 자연스럽다. 단순 크기 불변 분류와 다르다.
기존 네 위상 추론은 역회전한 출력을 예측끼리 대응한 뒤 평균한다.
TF64 TRAIN30에서73성분 중72개가 NMSE와 SI-SDR을 함께 개선했다.
하지만 이는 추가 분리 정보를 얻은 것이 아니며, 모델의 위상 의존 오차가
평균 과정에서 줄었을 가능성이라는 해석까지다.

정답 없는 출력 대응이 입력에 따라 바뀌므로 현재 구현 전체가 엄밀한
연속 U(1) equivariance를 만족한다고 주장하지 않는다. 신호원마다 다른
위상 회전·CFO·채널 변화와도 구분한다. 네 위상 성공으로 그 조건들을
검증했다고 쓰지 않는다.

후속 판단 후보는 위상별 출력 편차를 직접 분석하고, 필요하면 평균 출력에
대한 학습 또는 별도 위상 일관성 경로를 비교하는 것이다. 이전 canonical
phase·복소 마스크 실험은 이미 실패했으므로 이름만 바꿔 재실행하지 않는다.
현재 시간 변화 경로의 DEV 결과를 먼저 확인한다. 이 문서는 자동 후속 대기열이 아니다.

PDF·저자 코드 hash는 `PHASE_EQUIVARIANCE_PROVENANCE.json`에 남긴다.
코드를 실행·의존성 설치하지 않고 관련 연산과 설정을 정적으로 확인했다.
