# RFUAV 확대 자료의 U-Net / 게이트 U-Net 비교

실행 중인 코드의 검토·재현용 스냅샷이다. 활성 작업 위치는 `/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008`이며, 저장소로 복사하면서 학습 소스나 실행 경로를 바꾸지 않았다. 실제 실행에는 별도 자료 manifest·특징 캐시·동일 초기 체크포인트가 필요하다. 이 스냅샷에는 원자료와 가중치가 포함되지 않는다.

- RFUAV 단일 자료, 같은 원 RF 대역 안의 중심 정렬 합성. 원 수신 중심주파수 간격은 보존하지 않는다.
- 학습 83개 기존 원파일의 문맥 249→3,902개 확대. 새 독립 기록이나 새 기종의 추가가 아니다.
- 기존 5범주, 검증 원기록 묶음 분리 유지. 보류 Autel I/Q는 열지 않는다.
- 1–3개 기록 기여 파형과 배경을 분리한다. 합성 성분 개수는 실제 드론 대수와 구분한다.
- 같은 e22 기준 체크포인트, seed 0, LR 1e-5, 각 5 epoch / 375업데이트 / 12,000혼합. 검증 630혼합.
- 기준 U-Net 32,142,859파라미터, 게이트 U-Net 34,769,675파라미터. 자료·업데이트 예산을 맞추며 파라미터 수는 같지 않다.
- 게이트는 U-Net 병목에 시간축 dilated gated residual 블록을 추가한다. 출력 투영 0 초기화로 두 모델의 초기 출력이 일치한다. 전체 복소 입력 관측 길이는 0.63872ms로 유지하며, 20.97152ms 전력 특징은 기존 평균 문맥을 사용한다.
- WaveNet 구현은 후보로 보관하며 이번 두 군 비교에는 포함하지 않는다.

`ARCHIVE_MANIFEST.json`은 활성 소스와 복사본을 연결한다. `SOURCE_SNAPSHOT.json`과 `vendor/drone_rf/`는 사용한 기존 연구 모듈 버전을 고정한다. 모델·스케줄·검증 선택 조건은 `DENSE_GATED_CURRENT_KO.md`, 실제 실행 상태와 GPU 검사 결과는 저장소의 `reports/2026-10-08/DENSE_GATED_GPU_LAUNCH_KO.md`를 확인한다.

CPU 검사: 해당 폴더에서 기존 PyTorch 환경으로 `python -m unittest test_models -v`; 전환 검사: `python -m unittest discover -s launch -p test_host_switch.py -v`. `launch/` 도구는 이 연구 PC의 정확한 기존 프로세스와 자료를 전제로 하며 일반 설치 도구가 아니다. 이미 전환한 큐에 최초 전환 명령을 반복 실행하지 않는다.
