# 복원 개선 후보의 근거와 실험 결정

2026-10-09. 사용자의 자율 연구 지시에 따라 실험설계와 문헌 확인을 병행했다.
목표는 같은 데이터셋·같은 원 RF 대역의 1–3개 기록 기여 파형을 새 기록에서도
복원하는 것이다. 현재 학습은 합성 전 I/Q를 정답으로 사용하는 **지도학습**이다.
추론에 정답이 필요하지 않다는 사실을 비지도 학습과 혼동하지 않는다.

## 현재 근거가 가리키는 우선순위

기존 전체·국소 FiLM 결합은 동일 예산에서 두 지표 동시 개선 기준을 충족하지
못했다. 현재 마지막 디코더 결합을 같은 부모·동일 규모 두 군으로 비교 중이다.
이 결과가 나오기 전에 긴 문맥, Transformer 또는 Mamba가 해결책이라고 결론짓지
않는다. 원파형 모델이 기존 STFT 선택 모델보다 크게 나쁜 상태도 함께 공개한다.
두 계열의 사전학습 이력이 다르므로 이 차이를 공정한 구조 순위로 사용하지 않는다.

학습 네 혼합의 CPU 진단에서 마지막 디코더 결합 e1 모델은 혼합 전체의 위상만
바꾸고 되돌려도 예측이 달라졌다. 네 위상 예측 평균은 이 네 예제의 case-average
NMSE를 0.604353→0.584527로 낮췄다. 기존 STFT 모델의 두 성분 예제에서는 오히려
0.285732→0.287023으로 증가했다. 따라서 세 모델의 전체 개발 검증을 사전에 고정해
확인한다. 좋은 학습 예제만으로 이 방법을 채택하지 않는다.

또 기존 STFT 모델이 정답 보조 상수 배율 진단보다 평균 NMSE에서 약 1% 정도만
앞섰다는 [기존 결과](../2026-10-08/RECOVERY_SOLUTIONS_KO.md)를 고려한다.
시간·주파수별 전력 배분에 개선 여지가 있는지 CPU 진단을 추가했다. 정답의 STFT
전력을 쓰는 두 고정 마스크(주파수별 시간 평균 / 시간·주파수별 전력비)를 같은
420개 두·세 성분 검증 혼합에 적용한다. 이것은 실제 분리기나 비선형 성능 한계가
아니다. 정답을 사용하는 진단값을 논문의 제안법 결과로 대신 쓰지 않는다.

## 직접 확인한 방법론

아래는 실험 후보를 결정하기 위한 제한된 원문 조사다. 체계적 전수 검색이나
신규성 확정, 문헌의 자체 결과 재현은 아니다.

| 연구·자료 상태 | 본문에서 확인한 내용 | 우리 실험에 연결하는 가설과 경계 |
|---|---|---|
| Gansekoele 외, **Relative Phase Equivariant Deep Neural Systems**, TMLR 2025 | 전역 위상 회전을 군 작용으로 다루며 이산 회전·순환 합성곱을 설계한다. [게재 기록](https://ir.cwi.nl/pub/36105/), [원문 §§3.2–3.4](https://arxiv.org/html/2501.04730v2) | 전역 위상에 일관적인 RF 처리를 시험할 근거. 원 연구는 수신·복조이고, 이번 네 위상 추론 평균은 해당 구조 재현도, 정확한 순열 집합 등변성 증명도 아니다. |
| Gao 외, **IQUMamba-1D**, JKSUCIS 38:63, **2026** | U-Net에 선택적 상태 공간 처리, skip 처리, 다중 해상도 출력을 결합한다. Huber 손실과 MSE/SI-SDR 비교도 있다. [출판사 원문 §§4–6](https://link.springer.com/article/10.1007/s44443-025-00440-5) | 긴 문맥과 세부 복원을 결합하는 선행 구조. 결론에서 실제 OTA 및 3개 이상 분리의 한계를 밝힌다. DJI 실측 3대 해결 근거가 아니다. DOI의 2025를 출판연도로 잘못 쓰지 않는다. |
| Lifar 외, **The Radio-Frequency Transformer**, 2026 프리프린트 | 단독 SOI tokenizer를 먼저 학습하고 Transformer가 토큰을 추정한다. QPSK SOI 중심이며 통합 모델은 일부 간섭에서 전문 모델보다 나쁘다. [원문 §§3–4](https://arxiv.org/html/2603.09201v1) | 단독 파형 사전표현의 후보 근거. 우리 드론의 알려진 비트·변조 정답을 가정할 수 없다. §4.2는 원 RF Challenge의 CommSignal2 시험 누출 때문에 KU-TII 해당 결과를 제외한다고 명시하므로 그 수치를 목표값으로 복사하지 않는다. |
| **Score-based Source Separation with Applications to Digital Communication Signals**, NeurIPS **2023** | 두 성분의 score 모델을 이용해 합 일치 제약 아래 α-posterior/Randomized Gaussian Smoothing으로 반복 추정한다. [원문 §3·Algorithm 1](https://arxiv.org/html/2306.14411v3), [저자 코드](https://github.com/tkj516/score_based_source_separation) | RGS는 일반 잔차 보정층의 이름이 아니다. 단독 성분 분포 학습, 스케일/간섭 가정, 반복 추론 비용을 포함해야 한다. 최대 세 개의 미지 드론 성분에 바로 검증된 방법이라고 부르지 않는다. |
| Wang 외, **On the Compensation Between Magnitude and Phase**, IEEE SPL 2021 | 위상 오차가 있을 때 파형 손실이 크기 축소를 유도할 수 있으며 크기 손실과 파형 지표 사이의 절충을 분석한다. [원문 §§II–III](https://arxiv.org/html/2108.05470v2) | 크기 손실이 STFT를 선명하게 만들더라도 I/Q 복원이 좋아진다고 보장하지 않는다. 현재 RF 잔차가 같은 원인이라는 직접 증거도 아니다. 추가 손실은 원인 진단과 같은 예산 대조가 필요하다. |
| Lancho 외, **RF Challenge**, IEEE OJCOMS 2025 | RF 상관 길이에 맞춘 초기 커널과 U-Net·WaveNet을 설명한다. [원문 III.B](https://arxiv.org/html/2409.08839v3), [IEEE 기록](https://ieeexplore.ieee.org/abstract/document/10945811) | 시간 구조 후보의 근거다. IEEE 기록에 정정이 표시돼 있으며 정정 상세 원문은 이번에 확보하지 못했다. RFT의 특정 결과 제외 설명 이상으로 모든 결과의 무효를 주장하지 않는다. |
| Rodrigez 외, **Learning to Separate RF Signals Under Uncertainty**, 2026 프리프린트 | 여러 간섭 유형에 대한 Detect-Then-Separate와 통합 모델을 비교한다. [원문 §§II–IV](https://arxiv.org/html/2602.04650v1) | 모델 용량을 통제한 통합/전문 비교의 근거. 실험의 K=2를 동시 드론 2대로 읽으면 안 된다. 주요 설정의 K는 선택 가능한 간섭 유형 수다. |

앞서 확인한 Henneke의 SOI-matched autoencoder는 ICASSPW **2024** 자료다.
[저자 원문](https://rfchallenge.mit.edu/wp-content/uploads/2024/02/Lhen_final_paper.pdf)의
이전 검토는 [기존 보고](../2026-10-08/RECOVERY_SOLUTIONS_KO.md)에 있다.
이번 재접속은 시간 초과였으므로 이번에 새로 원문 검토를 완료했다고 표시하지 않는다.

## 실험에 따른 다음 결정

1. 현재 디코더 결합 두 군을 정해 둔 5 epoch까지 완료하고 모든 epoch를 검산한다.
   같은 NMSE 체크포인트 선택 규칙으로 2·3성분 각각의 NMSE와 SI-SDR을 비교한다.
2. 그 선택 가중치 두 개와 기존 STFT 가중치 하나에서 네 위상 추론 평균을 평가한다.
   양쪽 신호 수에서 두 지표가 함께 좋아져야 개발 후보로 채택한다. 계산량은 4배의
   순전파이며 추가 학습이 아니다. [고정 규약](../../experiments/rfuav_phase_average_20261009/README.md).
3. 위상 평균이 원파형 모델에서만 개선되더라도 기존 STFT보다 나쁘면 최선법으로
   승격하지 않는다. 무효 또는 혼합 결과라면 각도/평균 가중치를 검증셋에 맞춰 반복
   조정하지 않고 원인을 기록한다.
4. 전력 배분 진단에서 큰 개선 여지가 보이면 기존 강한 STFT 모델에서 전력 마스크
   학습/복소 보정 경로를 분리해 비교할 후보를 정한다. 전력 진단도 별 도움이 없다면
   단독 파형 사전표현, 프로토콜 구조, 기록 조건 변화를 우선 검토한다. 진단값을
   근거로 실제 성능을 미리 보장하지 않는다.
5. 새 학습마다 같은 부모·자료·업데이트·초기화와 한 가지 핵심 변경을 고정한다.
   개발 검증으로 반복 선택한 결과는 독립 시험 성과로 쓰지 않는다. Autel/예약 확인
   파일은 방법 확정 전까지 열지 않는다.

## 검색·검증 기록

- 확인일: 2026-10-09 KST. 공식 출판사·arXiv·기관 저장소·저자 저장소 사용.
- 주요 질의: `RF signal separation 2025 2026 complex waveform time frequency
  U-Net transformer weak signal RF Challenge`, `IQU Mamba RF signal separation
  2025 waveform RGS WaveNet source separation`, `relative phase equivariant deep
  2025`, `site:rfchallenge.mit.edu SOI matched autoencoder Henneke`,
  `Correction to RF Challenge Lancho Weiss 2025`, `RGS score diffusion RF`.
- research-lookup 기본 Parallel CLI/API 인증은 준비돼 있지 않아 기존 웹 검색으로
  원문을 직접 확인했다. 60편 문헌 패킷/완전 검색을 수행했다고 표시하지 않는다.
- 절차 지원: experimental-design, research-lookup. Kassis, T., Agarwal, V.,
  He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library
  of Procedural Knowledge for Research Agents*.
  [확인한 현행 기록](https://doi.org/10.48550/arXiv.2609.00065).
  이는 RF 성능의 근거 인용이 아닌 연구 절차·소프트웨어 인용이다.
