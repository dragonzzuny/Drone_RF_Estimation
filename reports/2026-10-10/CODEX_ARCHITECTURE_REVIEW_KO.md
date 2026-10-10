**핵심은 입력 길이보다, 성분별 표현을 시간축에 걸쳐 만들고 복원까지 유지하는 방식입니다.** 다만 현재 실패를 “문맥 부족”이나 “잔여 오차” 하나로 확정할 근거는 없습니다. 아래 수치는 기존 보고서의 관측이며, 이번에는 텍스트만 검토하고 재평가하지 않았습니다.

1. **관측 사실과 정보 손실을 구분해야 합니다.** 긴 경로는 전력 특징 생성에서 위상을, 이후 시간 평균에서 순서를 버립니다. 평균을 반복한 TCN 입력으로 원래 시간 변화를 복구할 수는 없습니다. 반면 국소 복소 STFT와 고해상도 skip은 남아 있으므로 **모델 전체가 시간·위상 정보를 잃었다는 설명은 틀립니다.** 병목의 평균 pooling도 해당 경로를 압축하지만 skip까지 지우지는 않습니다. [전력 특징](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/rfuav_dense_gated_20261008/vendor/drone_rf/context_data.py:22), [시간 평균](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/rfuav_dense_gated_20261008/vendor/drone_rf/context_model.py:125), [기반 모델](/home/pyj/문서/GitHub/Drone_RF_Estimation/src/drone_rf/model.py:49)

2. **기존 추가층의 실패는 실제 오디오 구조의 실패가 아닙니다.** `source_head`는 각 TF점에서 `[공유 특징, 해당 출력의 RI]`로 세 슬롯끼리 attention합니다. 성분별 시계열 상태가 없고, 보정합이 0이라 본체를 고정하면 성분 총합과 배경을 바꾸지 못합니다. 세 슬롯 입력이 모두 같으면 보정도 정확히 0입니다. `adapter`는 압축된 병목에서 두 축 BLSTM을 수행하며 cross-frame attention이 없습니다. 두 모듈의 영초기화 readout은 초기 함수를 보존하지만, **첫 backward에서 추가 모듈 내부로 가는 데이터 손실 gradient는 0**입니다. 이후까지 학습 불능이라는 뜻은 아닙니다. [source_head.py](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/source_interaction_20261010/source_head.py:35), [adapter.py](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/tf_axis_20261010/adapter.py:13)

   실제 **SepTDA**는 전체 시퀀스에 접근하는 학습 query로 attractor를 만들고, FiLM으로 성분축을 생성한 뒤 구간 내부·구간 사이·성분 사이 처리를 반복합니다. **TF-GridNet**은 TF 표현에서 주파수·시간·프레임 간 attention을 결합합니다. **SepNetEDCI**는 매 단계에 원혼합 특징과 이전 *프레임별 표현*을 전달하고, 원혼합과 초기 추정을 이용해 정제합니다. 작은 출력 attention이나 잔여 재입력과는 정보 흐름이 다릅니다. [SepTDA §2](https://arxiv.org/html/2401.12473v1), [TF-GridNet §III](https://arxiv.org/html/2211.12433v2), [SepNetEDCI §2](https://www.isca-archive.org/interspeech_2025/yang25_interspeech.pdf)

3. **잔여의 문제는 오차 결합과 추정 순서이지, 셋째 출력의 학습 부재가 아닙니다.** 현재 구조는
   \[
   a=f_1(x),\quad b=f_2(x-a),\quad c=x-a-b
   \]
   이므로, 배경 정답이 0인 현재 합성에서
   \[
   e_3=-(e_1+e_2),\qquad
   \|e_3\|^2=\|e_1\|^2+\|e_2\|^2+2\Re\langle e_1,e_2\rangle .
   \]
   오차는 상쇄할 수도 있어 “단계마다 반드시 누적된다”도 부정확합니다. 약한 정답의 작은 전력으로 나누면 상쇄 후에도 NMSE가 클 수 있습니다. 관측된 **TRAIN 14삼중 혼합의 마지막 출력 NMSE 1.549152**는 미학습 구조의 결과이지 달성 가능한 성능 하한이 아닙니다. [출력별 관측](/home/pyj/문서/GitHub/Drone_RF_Estimation/reports/2026-10-10/SUCCESSIVE_INITIAL_ROLES_KO.md:13)

   PIT와 선택 슬롯이 고정된 구간에서, RI 좌표의 출력 gradient를 \(g_i\)라 하면
   \[
   \frac{dL}{da}=g_1-g_3-J_{f_2}^{\mathsf T}(g_2-g_3).
   \]
   `detach`가 없어 마지막 손실도 첫 추출까지 전달됩니다. 문제는 전력 `argmax`의 선택 전환이 미분되지 않고, 나머지 추정들을 최종 조립에서 버리며, 셋째에 별도 추정 경로가 없다는 비대칭입니다. **잔여 표현 자체는 합이 맞는 임의의 세 파형을 표현할 수 있습니다.** [successive.py](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/recursive_20261010/successive.py:15)

4. **원혼합 conditioning은 정보 접근을 복원하지만 잔여 설계를 해결하지는 않습니다.** 기존 정규화 잔여 \(R/\sigma_R\)와 새 특징 \(X/q,\rho=\sigma_R/q\)를 함께 보면, 바닥값 밖에서 \(R/q=\rho R/\sigma_R\), 첫 추정도 \((X-R)/q\)로 표현할 수 있습니다. 따라서 두 번째 단계에 원혼합과 상대 잔여 규모를 돌려주는 의미가 있습니다. 하지만 새 관측을 추가한 것은 아니며, 셋째 잔여·배경 0·강도순 선택은 그대로입니다. 또한 출력은 여전히 잔여 RMS로 스케일됩니다. 영초기화 convolution은 첫 backward부터 가중치 gradient를 받을 수 있고, CPU 검사 통과는 이 연결의 확인입니다. **RF 개선 증거는 아닙니다.** [conditioning.py](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/recursive_conditioning_20261010/conditioning.py:11)

5. **합 일치와 PIT가 식별 정보를 만들지는 않습니다.** 기반 투영은
   \[
   \hat S_i=U_i+\frac{X-\sum_{j=1}^{4}U_j}{4}
   \]
   로 공통 오차 방향만 제거합니다. 성분 간 잘못된 재배분은 그대로 허용합니다. 현재 손실은 이미 전체창 PIT와 성분별 전력 정규화 NMSE를 사용하므로, 단순히 “약신호가 작아서 손실에서 무시된다”는 설명도 부족합니다. coherence는 상수 복소 배율에 불변이고, 절대 진폭·위상 오차는 NMSE가 담당합니다. [투영](/home/pyj/문서/GitHub/Drone_RF_Estimation/src/drone_rf/model.py:64), [손실](/home/pyj/문서/GitHub/Drone_RF_Estimation/src/drone_rf/losses.py:33)

구체적인 구조 후보는 다음 두 가지입니다. **둘 다 전체 32.14M 본체를 보존하는 설계 가설**입니다.

| 후보 | 명시적 forward | 기존 시도와의 차이 |
|---|---|---|
| **A. SepTDA형 성분별 시퀀스 디코더** | 기존 혼합 \(X\)·문맥·창 위치 → 전체 U-Net 특징 \(H\). 시간 순서를 유지한 `[H, RI(X)]`에 세 학습 query가 접근 → attractor → FiLM으로 \(H_k[F,T,D]\) 생성 → 구간 내부·사이 및 성분축 처리 → 세 복소 출력 보정과 별도 배경 보정 → 기존 출력에 더하고 합 투영·iSTFT. | 성분축을 **시퀀스 처리 전에 생성하여 디코딩까지 유지**합니다. 마지막 TF점의 attention과 다릅니다. 국소 입력 길이를 유지해도 성립하므로 문맥 확대를 전제로 하지 않습니다. |
| **B. 원혼합을 참조하는 전체 출력 공동 정제** | 전체 U-Net의 초기 네 출력 \(Y^0\)와 원혼합 \(X\) → 각 성분의 `[RI(X), RI(Y_k^0)]` 표현 → 세밀한 TF격자의 주파수·시간·cross-frame 처리와 성분 교환 → 네 복소 보정 \(\Delta_k\) → \(\operatorname{iSTFT}P_4(Y^0+\Delta)\). | 강한 출력 선택이나 잔여 전용 재입력 없이 **세 추정 모두를 직접 수정**합니다. 현재 conditioning과 달리 셋째도 학습된 보정을 받습니다. SepNetEDCI의 정제 원리를 적용하는 후보이며 전체 재현은 아닙니다. |

두 후보 모두 추론 출력은 **세 파형+배경, 1–3 개수 logits**이고 정답 개수·기종은 입력하지 않습니다. RI와 양쪽 주파수를 유지하며, 음성의 비음수 마스크를 복소 STFT에 그대로 곱해 혼합 위상에 묶지 않습니다.

마지막으로, 추가 75업데이트에서 TRAIN 적합도는 좋아지고 DEV가 나빠진 관측은 “구조가 정보를 전혀 이용하지 못한다”는 설명에 반합니다. 원기록·분포 차이와 반복 적합도 함께 고려해야 합니다. 기존 native RFUAV 비교는 TRAIN 8/DEV 5 원기록 묶음, 반복 사용한 DEV 630혼합과 제한된 삼중 조합에 기반합니다. [적합도 진단](/home/pyj/문서/GitHub/Drone_RF_Estimation/reports/2026-10-10/COUNT_TRAIN_FIT_KO.md:20), [비교 범위](/home/pyj/문서/GitHub/Drone_RF_Estimation/reports/2026-10-10/SOURCE_INTERACTION_FINAL.md:37) **schedule3 대 replay1 결과 전에는 후속 학습 판단을 유보해야 합니다.** 그 결과 역시 같은 원기록의 새 합성 효과이며, 구조적 원인을 단독으로 확정하지는 못합니다.
