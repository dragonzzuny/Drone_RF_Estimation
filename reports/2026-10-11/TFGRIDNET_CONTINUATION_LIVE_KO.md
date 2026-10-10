# TF-GridNet RF 고정 TRAIN4 진행 보고

같은 네 학습 혼합의 적합도 검사다. DEV·보류 I/Q는 읽지 않았고, 기존 최선 모델을 대체하지 않는다.

최종 확인 상태: TRAINING. 실제 worker STATE와 대조한다.

|업데이트|두 성분 NMSE|세 성분 NMSE|두 성분 SI-SDR dB|세 성분 SI-SDR dB|세 성분 최약NMSE|
|---:|---:|---:|---:|---:|---:|
|32|0.202298|0.468279|7.227|1.291|0.904962|
|40|0.156362|0.411629|8.422|2.636|0.793328|

새 초기화6블록/D128/LSTM192/4heads. 전체512×500 격자. 전체최대64업데이트/추가40분의 진단이며 수렴 완료·새 기록 일반화·물리 드론 대수 추정의 증거가 아니다.

[고정 규약](TFGRIDNET_CONTINUATION_PLAN_KO.md) · [초기화 추가 명세](TFGRIDNET_INITIALIZATION_NOTE_KO.md) · [전체 수치](TFGRIDNET_CONTINUATION_PROGRESS.json)
