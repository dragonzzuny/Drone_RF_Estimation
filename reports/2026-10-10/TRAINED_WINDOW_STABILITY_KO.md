# 일관성 학습이 실제 창 의존성을 줄였는가

원 부모와 두 창 대조/일관성 후보의 실제e1을 비교했다. 각 개수의 앞2혼합씩 TRAIN6+DEV6, 총72회 CPU 추론이다. 원래 합성 일정과 ±163.84μs의 동일한 두 창 규칙을 사용했다. 공통474.88μs의 예측을 정답 순서로 대응한 값은 진단용이며 정답을 추론에 주지 않는다.

| 자료 | 성분 수 | 모델 | 일관성 항 ↓ | 창 간 차이/혼합 에너지 ↓ | 원창 NMSE ↓ | 이동창 NMSE ↓ |
|---|---:|---|---:|---:|---:|---:|
|train_pack|1|parent|0.118834|0.348936|0.058007|0.012982|
|train_pack|2|parent|0.428618|0.348182|0.186290|0.544278|
|train_pack|3|parent|0.031765|0.038411|0.595590|0.602558|
|validation_pack|1|parent|0.000062|0.000175|0.005102|0.004869|
|validation_pack|2|parent|0.044032|0.000104|0.650259|0.725576|
|validation_pack|3|parent|0.206287|0.012997|0.559944|0.553243|
|train_pack|1|paired_supervision|0.123414|0.361817|0.061553|0.005643|
|train_pack|2|paired_supervision|0.410675|0.345884|0.177330|0.548062|
|train_pack|3|paired_supervision|0.204262|0.061054|0.614725|0.568995|
|validation_pack|1|paired_supervision|0.000029|0.000079|0.002176|0.002056|
|validation_pack|2|paired_supervision|0.054223|0.000272|0.765269|0.875604|
|validation_pack|3|paired_supervision|0.187050|0.011544|0.556185|0.547302|
|train_pack|1|paired_consistency|0.133545|0.390928|0.066930|0.005922|
|train_pack|2|paired_consistency|0.396342|0.337718|0.171069|0.547879|
|train_pack|3|paired_consistency|0.204344|0.065617|0.616477|0.582273|
|validation_pack|1|paired_consistency|0.000028|0.000077|0.002524|0.002402|
|validation_pack|2|paired_consistency|0.059956|0.000293|0.754418|0.837077|
|validation_pack|3|paired_consistency|0.155219|0.009644|0.554776|0.553379|

창 간 예측이 비슷해지는 것과 정답에 가까워지는 것은 별개다. 개수마다2사례뿐이며 반복 사용한 DEV의 기전 진단이다. 실제 모델 채택은 이미 완료한630개 전체 DEV 파형 지표를 기준으로 하며, 이 사후 진단으로 바꾸지 않는다.

[모든36행](TRAINED_WINDOW_STABILITY_RESULT.json) · [전체 DEV 결과](PAIRED_WINDOW_KO.md)
