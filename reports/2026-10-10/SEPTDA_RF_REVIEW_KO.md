**정적 검토와 JSON 대조에서 확정적인 구현 차단 요인은 발견하지 못했습니다.** 다만 실제 학습 경로의 검증 공백이 하나 있습니다.

- **검증 공백:** 실제 학습은 전체 `predict` checkpoint 안에 블록별 checkpoint를 중첩합니다([train.py:117](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/train.py:117)). CPU 검사와 GPU preflight는 직접 `predict`를 호출하므로, 중첩 경로의 출력·기울기 동등성까지 확인한 것은 아닙니다([check.py:62](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/check.py:62), [train.py:107](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/train.py:107)). **구현 오류의 증거는 없지만, preflight를 실제 학습 경로와 맞추길 권합니다.**

- **축·미분은 일관됩니다.** chunk `[B,S,K,D]`의 fold 순서와 중첩 횟수 정규화가 맞습니다. intra는 K축, inter는 S축, source attention은 C축을 처리합니다. overlap-add∘segment의 미분도 항등입니다([architecture.py:15](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/architecture.py:15), [architecture.py:77](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/architecture.py:77), [architecture.py:108](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/architecture.py:108)).

- **query→FiLM→8블록→복소 출력 경로가 연결됩니다.** 네 보정의 평균 제거는 \(I-\mathbf1\mathbf1^\top/4\) 투영으로, 합 일치를 보존하며 세 번째 신호에도 직접 학습 경로가 있습니다. 영 readout의 초기 부모 동등성과 내부 초기 영기울기는 설계와 일치합니다([architecture.py:136](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/architecture.py:136), [architecture.py:161](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/architecture.py:161)). CPU JSON의 코드 지문도 현재 파일과 일치합니다.

- **학습·감사 로직은 규약과 맞습니다.** schedule3·2,400예제·75업데이트, 부모/분리부 lr, 원 손실을 유지합니다. 감사는 optimizer 상태·75단계·실제/최종/선택 tensor를 검사하며, 최소 NMSE 선택과 공동 채택 기준을 구분합니다([train.py:88](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/train.py:88), [audit.py:22](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/audit.py:22)). FRESH 대조의 관련 JSON 지문도 감사 기록과 일치했습니다.

75업데이트의 초기 적응, 반복 DEV 선택, 다른 모델 크기·계산량, 기존 count head 유지와 오디오 원형과의 차이는 **과학적 한계이며 구현 결함은 아닙니다**([계획서:15](/home/pyj/문서/GitHub/Drone_RF_Estimation/reports/2026-10-10/SEPTDA_RF_PLAN_KO.md:15)). ordered-context 완료·감사 PASS 선행 조건도 코드에 있습니다([train.py:34](/home/pyj/문서/GitHub/Drone_RF_Estimation/experiments/septda_rf_20261010/train.py:34)).

파일 수정·원시 I/Q/체크포인트 접근·학습·GPU 실행은 하지 않았습니다.
