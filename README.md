# Drone RF Estimation

드론의 혼합 복소 I/Q에서 각 신호를 복원하고, 학습에 사용하지 않은 기록·기종으로 일반화하는지 검증하는 연구입니다. 2026-10-06부터 이 저장소를 코드·계획·결과의 공유 위치로 사용합니다.

**현재 상태:** 기존 두 신호 복원 실험은 진행 중입니다. 다기종·가변 개수 모델의 성능은 아직 검증되지 않았습니다. 이번 첫 커밋에는 외장하드 자료 조사, 조종기 제외 정책, 새 실험계획, 원 규모 U-Net 및 순열 불변 손실의 준비 코드를 포함합니다.

- [목표와 실험계획](docs/PLAN_KO.md)
- [자료와 조종기 제외 기준](docs/DATA_POLICY_KO.md)
- [확인된 결과와 진행 상태](reports/2026-10-06/STATUS_KO.md)
- [데이터 조사 요약](reports/2026-10-06/inventory_summary.json)

## 준비 코드 실행

Python 3.10 이상과 NumPy·PyTorch 환경에서:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/inventory_rfuav.py --data-root /path/to/uav_rf_research_20260914 --output local/inventory
python scripts/check_raw_regions.py --inventory local/inventory/INVENTORY.json --output local/raw_qc.json
```

자료 조사는 `extracted/rfuav`의 XML·파일 목록과 `archives`의 ZIP 목차를 읽습니다. 두 번째 명령은 학습·검증 후보 파일의 세 위치를 제한적으로 읽어 수치 이상과 부분 중복을 검사합니다. 미지 기종 보류 파일은 읽지 않습니다. 전체 파일 무결성이나 수집 세션 독립성을 대신하는 검사가 아닙니다.

`src/drone_rf/model.py`는 기존 N의 64–128–256–512–1024 채널 본체를 유지합니다. 개별 신호 출력과 배경 출력의 합이 입력과 일치하도록 구성했습니다. `losses.py`는 두 개 또는 네 개 출력의 순서를 전체 파형 단위로 맞춰 학습합니다. **아직 새 자료용 학습 실행기와 GPU 학습 결과는 포함하지 않습니다.** CPU 검사는 모델 축소 실험이나 RF 성능 평가가 아닙니다.

실행 중인 이전 실험은 기존 작업 디렉터리에 보존하고, 검증이 끝난 결과와 새 코드를 이 저장소로 옮깁니다. 원시 데이터·체크포인트·자격증명은 Git에 넣지 않습니다. 과거 연구의 결과를 새 다기종 모델의 성능으로 표시하지 않습니다.
