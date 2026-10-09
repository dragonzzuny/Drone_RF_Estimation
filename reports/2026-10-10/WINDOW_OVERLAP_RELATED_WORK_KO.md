# 창 병합의 선행 근거와 이번 진단의 범위

2026-10-10 공식 문서·코드를 확인했다. [Asteroid LambdaOverlapAdd](https://asteroid-team.github.io/asteroid/package_reference/dsp.html)는 겹친 구간의 상관을 이용해 출력 순서를 맞추고 창을 합친다. [구현](https://github.com/asteroid-team/asteroid/blob/master/asteroid/dsp/overlap_add.py)은 순서 정렬 뒤 합성 창을 적용한다. 창마다 출력 순서가 달라질 수 있는 분리기를 긴 신호에 적용하는 기존 방식이다.

[Demucs의 apply_model](https://github.com/facebookresearch/demucs/blob/main/demucs/apply.py)은 여러 시간 이동의 예측 평균과 창 중앙에 큰 가중치를 주는 병합을 구현한다. 이는 음악 분리 코드이며 드론 I/Q에서 개선을 보증하는 실험은 아니다. 유한 개 이동 평균이 모든 이동에 대한 정확한 등변성을 보장한다고 확대하지 않는다.

현재 RF 진단은 같은 원기록 혼합을 다른 시작의 창으로 자르고, 공통 표본의 복소 예측 차이로 세 슬롯을 정렬한다. 원기록의 RF 간격·위상·이득을 보존한다. 세 창에서 같은 정답 표본을 평가하므로 활동량 차이에 따른 별개 창의 비교를 피한다. 창 위치에 따라 문맥 정렬과 국소 정규화도 달라져 padding만의 인과 실험은 아니다.

이 절차 자체를 새로운 분리 이론으로 주장하지 않는다. 기존 방법이 현재 모델의 잔차를 줄이는지 검증하고, 이후 겹침 일관성 학습을 고려할 근거를 마련하는 진단이다. 실제 18회 결과는 별도 보고하며 선행의 수치를 우리 성능으로 사용하지 않는다.
