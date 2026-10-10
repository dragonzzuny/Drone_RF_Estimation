# TF-GridNet 이식의 초기화 차이 추가 명세

진행 중인 v1의 소스·규약·가중치는 바꾸지 않고 코드 대조 결과를 추가한다.
전체6블록/D128/LSTM192/4heads는 유지하지만, 음성 recipe를 정확히
재현한 초기화라고 해석해서는 안 된다.

현재 RF 구현은 행렬·합성곱·LSTM 가중치에 Xavier uniform을 적용하되,
사용자 정의 정규화의 gamma=1/beta=0을 보존한다. 이름이 `bias`로 끝나는
항은0으로 만들고, LSTM의 `bias_ih_l0` 등은 PyTorch 생성 시 값이 남는다.

같은 ESPnet commit의 공통 `initialize.py`는2차원 이상 파라미터 모두에
Xavier를 적용하고1차원 이름에 `.bias`가 포함되면0으로 만든다. 따라서
사용자 정의 다차원 gamma/beta와 LSTM bias 처리가 RF v1과 다르다.
기존 규약의 “Xavier 규칙”은 가중치 계열을 뜻하며, 이 추가 명세에 따라
정확한 recipe 초기화와 구분한다. 파형·손실·주파수 양측 처리도 이미
별도의 RF 변경이므로 현재 결과는 음성 논문의 재현값이 아니다.

CPU에서 검증한 저자 블록과 chunk 처리의 일치는 **동일한 블록 가중치**를
복사했을 때의 연산 일치다. 학습 초기화 전체 또는 음성 학습 결과 일치를
검증한 것으로 확대하지 않는다.32업데이트 결과를 보고하고, 초기화
선택과 충분한 수렴을 분리해서 판단한다. 진행 도중 초기화를 바꾸거나
불리한 초기 수치를 지우지 않는다.

[ESPnet 초기화 원문](https://github.com/espnet/espnet/blob/3826362c7313d69d92eca721ec49441b6908c0b0/espnet2/torch_utils/initialize.py)
