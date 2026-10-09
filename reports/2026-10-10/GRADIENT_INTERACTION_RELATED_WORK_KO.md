# 손실 간 학습 방향: 선행 근거와 현재 범위

## 확인한 원문

[Yu 외, Gradient Surgery for Multi-Task Learning, NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), §2.2–2.4·알고리즘1은 음의 내적을 갖는 작업 기울기를 투영하는 PCGrad를 설명한다. 충돌만으로 성능 저하가 확정되는 것은 아니며 크기 불균형·곡률·학습률 조건도 다룬다. 여러 작업을 순회하는 순서는 무작위다. 두 작업의 볼록 조건에서의 수렴 논의도 모든 비볼록 U-Net·AdamW 설정의 개선 보장은 아니다.

[Liu 외, Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf)는 평균 손실의 개선과 각 작업의 국소 개선을 함께 고려하는 CAGrad를 제시한다. PCGrad와 같은 알고리즘이 아니며, 수렴 주장을 현재 RF 복원기에 자동 적용하지 않는다.

두 원문은 드론 세 기체의 실제 I/Q 분리 효과를 검증한 근거가 아니다. 학습 방향 조정 자체도 신규 이론으로 주장하지 않는다. 현재 GPU에는 PCGrad/CAGrad를 구현·적용하지 않았다.

## 현재 관찰에서 도출할 수 있는 가설

[소수 학습6사례의 전체 파라미터 진단](MAGNITUDE_PARAMETER_GRADIENT_KO.md)에서 단일 성분 크기 손실 기울기와 다중 성분 원 손실 기울기의 내적은 음수였다. 시작점에서 단일 성분의 보조 목표를 낮추는 작은 일반gradient-descent 이동이 해당 다중 성분 손실을 높일 수 있는 방향이다. 그러나 전체자료·학습 경로·AdamW의 실제 이동을 측정한 결과는 아니다.

현재는 [단일 성분 보조항 제거 대조](SELECTIVE_MAGNITUDE_PLAN_KO.md)로 이 감독 범위가 실제 개발 성능에 미치는 영향을 확인한다. 순수 기울기 투영과 다른 실험이다. 결과가 좋더라도 기울기 간섭이 유일한 원인이라고 확대하지 않는다.

추후 투영을 검토한다면 실제 학습 batch의 개수별 기울기 빈도·크기·내적을 먼저 기록하고, 투영 없는 같은 자료·예산의 대조를 둬야 한다. 작업별 backward 비용과 업데이트 횟수를 모두 보고해야 한다. 단순히 음의 cosine을 발견한 것을 새 분리 정보의 확보나 일반화 성공으로 해석하지 않는다.
