# 친화도 보조 목표와 실제 파형의 종점 진단

원규모 실제e1 세 군에서 개수별 앞2혼합씩TRAIN4+DEV4,총24회CPU추론이다. 정답은 친화도 진단과파형평가에만 사용했다. 개발 자료의 결과로 추가 하이퍼파라미터를 선택하지 않는다.

| 자료 | 성분 수 | 군 | hard친화도 손실 ↓ | soft친화도 손실 ↓ | NMSE ↓ |
|---|---:|---|---:|---:|---:|
|train_pack|2|affinity_control|0.121118|0.106619|0.177624|
|train_pack|3|affinity_control|0.228051|0.105472|0.600235|
|validation_pack|2|affinity_control|0.205123|0.134845|0.759393|
|validation_pack|3|affinity_control|0.179338|0.104441|0.555170|
|train_pack|2|hard_affinity|0.091850|0.081036|0.177530|
|train_pack|3|hard_affinity|0.217578|0.090993|0.601680|
|validation_pack|2|hard_affinity|0.172972|0.088497|0.761916|
|validation_pack|3|hard_affinity|0.166276|0.081285|0.555607|
|train_pack|2|soft_affinity|0.133872|0.105457|0.177997|
|train_pack|3|soft_affinity|0.275929|0.091532|0.600770|
|validation_pack|2|soft_affinity|0.171738|0.074570|0.751540|
|validation_pack|3|soft_affinity|0.226729|0.060781|0.555693|

hard와soft는 서로 다른 목표이므로 각각 같은 열 안에서 군을 비교한다. 보조 손실 감소만으로 복원 개선을 주장하지 않는다. 전체630개DEV의채택판정을 이 소수 사례로 대체하지 않는다.

[전체24행](SOURCE_AFFINITY_ENDPOINT_RESULT.json) · [전체개발비교](SOURCE_AFFINITY_KO.md)
