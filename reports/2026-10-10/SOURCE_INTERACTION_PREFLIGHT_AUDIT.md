# 성분 상호작용 출력층: 원 규모 GPU TRAIN4 검사

같은 보존 U-Net e2·학습 혼합4개·각32업데이트·원 손실·AdamW1e-4·동일 초기 예측이다. 본체는 양쪽 모두 학습했다. 아래 수치는 고정 TRAIN 적합도이며 개발검증 성능이 아니다. 검사 가중치는 폐기했다. 모든 기록 단계와 성분 수를 함께 보고한다.

| 군 | 업데이트 | 성분 수 | TRAIN NMSE ↓ | TRAIN 복소 SI-SDR ↑ dB |
|---|---:|---:|---:|---:|
|retained_unet|0|2|0.186290|11.866|
|retained_unet|0|3|0.595590|-5.363|
|retained_unet|1|2|0.189786|11.966|
|retained_unet|1|3|0.539302|-0.278|
|retained_unet|8|2|0.173953|13.277|
|retained_unet|8|3|0.476320|2.194|
|retained_unet|16|2|0.121604|15.668|
|retained_unet|16|3|0.447555|3.724|
|retained_unet|32|2|0.073123|16.257|
|retained_unet|32|3|0.368011|5.440|
|source_interaction|0|2|0.186290|11.866|
|source_interaction|0|3|0.595590|-5.363|
|source_interaction|1|2|0.189815|11.965|
|source_interaction|1|3|0.539228|-0.267|
|source_interaction|8|2|0.173935|13.270|
|source_interaction|8|3|0.476264|2.194|
|source_interaction|16|2|0.121111|15.677|
|source_interaction|16|3|0.447239|3.728|
|source_interaction|32|2|0.072147|16.289|
|source_interaction|32|3|0.367325|5.450|

두 군의 초기 GPU 파형·개수 logits가 정확히 같았고, 추가층의 내부 attention에도 첫 단계 이후 기울기가 전달됐다. 통과 여부는 실행·최적화 조건에 한정하며 새 기록 일반화의 증거가 아니다.

- retained_unet: 32,142,859파라미터, 40.0초, 최대 할당 2.293GiB, 두 성분 수의 초기 대비 NMSE 감소=True.
- source_interaction: 32,180,747파라미터, 77.3초, 최대 할당 2.375GiB, 두 성분 수의 초기 대비 NMSE 감소=True.
