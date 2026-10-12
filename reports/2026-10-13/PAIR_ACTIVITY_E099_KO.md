# 조합·국소 활동별 부모99의 복원 오차

기존630개 평가 수치와 학습 전용 PSD 템플릿을 재분석했다. 추가 추론·학습·임계값 탐색은 없다.

|조합|혼합 수|U-Net99 NMSE↓|기존 스펙트럼 기준 NMSE↓|U-Net 최약 NMSE↓|
|---|---:|---:|---:|---:|
|DJI AVATA2 + DJI FPV COMBO|54|0.201649|0.254494|0.348236|
|DJI AVATA2 + DJI FPV COMBO + DJI MAVIC3 PRO|210|0.381557|2.540365|0.722353|
|DJI AVATA2 + DJI MAVIC3 PRO|54|0.281927|0.566822|0.501188|
|DJI FPV COMBO + DJI MAVIC3 PRO|51|0.151918|0.330006|0.239671|
|DJI MINI3 + DJI MINI4 PRO|51|0.395999|0.325550|0.673159|

|성분 수|기체|국소 상대 전력<0.1|성분 수|U-Net NMSE|스펙트럼 기준 NMSE|
|---|---|---|---:|---:|---:|
|2|DJI AVATA2|False|108|0.161916|0.263274|
|2|DJI FPV COMBO|False|93|0.133882|0.217777|
|2|DJI FPV COMBO|True|12|0.573173|1.092836|
|2|DJI MAVIC3 PRO|False|69|0.145778|0.343556|
|2|DJI MAVIC3 PRO|True|36|0.579087|1.023791|
|2|DJI MINI3|False|18|0.373035|0.107078|
|2|DJI MINI3|True|33|0.666550|0.703775|
|2|DJI MINI4 PRO|False|37|0.154978|0.046793|
|2|DJI MINI4 PRO|True|14|0.424781|0.451630|
|3|DJI AVATA2|False|210|0.300298|0.602291|
|3|DJI FPV COMBO|False|179|0.209145|0.389799|
|3|DJI FPV COMBO|True|31|0.906359|35.677398|
|3|DJI MAVIC3 PRO|False|134|0.327729|0.577406|
|3|DJI MAVIC3 PRO|True|76|0.893011|2.905299|

|조합|학습 평균PSD 겹침 질량|
|---|---:|
|DJI MINI3 + DJI MINI4 PRO|0.009370|
|DJI AVATA2 + DJI FPV COMBO|0.140647|
|DJI AVATA2 + DJI MAVIC3 PRO|0.388267|
|DJI FPV COMBO + DJI MAVIC3 PRO|0.080316|

겹침 질량은 두 정규화된 학습 평균PSD의 bin별 최솟값을 합한 기술통계다.
평균PSD가 떨어져 있어도 순간 패킷의 중첩 여부와 저활동 구간 잡음 분리의 난도는 별도다.
수신 중심 주파수는 송신 신호의 실제 중심 또는 점유 대역폭과 같다고 가정하지 않았다.
국소 상대 전력0.1은 기존 사후 설명 기준이다. 신호/잡음 정답이나 실패의 원인을 확정하지 않는다.
기체·기록·전력·활동 조건이 함께 달라지며, 표의 창 수는 독립 녹음 수가 아니다.
