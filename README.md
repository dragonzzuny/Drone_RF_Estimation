# Drone RF Estimation

드론의 혼합 복소 I/Q에서 각 신호를 복원하고, 학습에 사용하지 않은 기록·기종으로 일반화하는지 검증하는 연구입니다. 2026-10-06부터 이 저장소를 코드·계획·결과의 공유 위치로 사용합니다.

**현재 상태:** 기존 두 신호 복원 실험은 진행 중입니다. 새 다기종 실험은 원자료 검사·393개 캐시·혼합 목록과 실제 I/Q 합성 검산까지 마쳤습니다. 다기종·가변 개수 모델의 GPU 학습과 성능 평가는 아직 시작하지 않았습니다.

- [목표와 실험계획](docs/PLAN_KO.md)
- [자료와 조종기 제외 기준](docs/DATA_POLICY_KO.md)
- [확인된 결과와 진행 상태](reports/2026-10-06/STATUS_KO.md)
- [데이터 조사 요약](reports/2026-10-06/inventory_summary.json)
- [CPU 병행 처리와 산출물 정리](reports/2026-10-06/CPU_AND_CLEANUP_KO.md)
- [CPU 원자료 검사 완료와 후속 혼합 구성](reports/2026-10-06/CPU_COMPLETED_KO.md)
- [실제 혼합 입력 검산과 창별 전력차](reports/2026-10-06/MIXTURE_INPUTS_KO.md)
- [전력 처리 대조 50 epoch 최종 결과](reports/2026-10-06/POWER_COMPARISON_KO.md)
- [반복 간격 측정과 시간 문맥 모델 후보](reports/2026-10-06/TEMPORAL_STRUCTURE_KO.md)
- [최대 세 신호와 TCN·Transformer·LSTM 문맥 구현](reports/2026-10-06/CONTEXT_THREE_KO.md)

## 준비 코드 실행

Python 3.10 이상과 NumPy·PyTorch 환경에서:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/inventory_rfuav.py --data-root /path/to/uav_rf_research_20260914 --output local/inventory
python scripts/check_raw_regions.py --inventory local/inventory/INVENTORY.json --output local/raw_qc.json
```

전체 해시·긴 구간 캐시·학습/검증 이동 유사도 검사는 GPU 학습과 독립된 CPU 프로세스로 실행할 수 있습니다. 아래 코어 번호와 경로는 해당 기계에 맞춥니다. 출력 폴더는 새 실행 전용으로 지정합니다.

```bash
ionice -c 2 -n 7 python scripts/prepare_cpu_corpus.py \
  --inventory local/inventory/INVENTORY.json \
  --output local/cpu_intake --cache /path/to/ssd/iq_cache --cpus 14,15
```

이번 실제 실행은 실행 중 코드가 바뀌지 않도록 로컬 스냅샷에서 시작했습니다. 작업별 상태는 `PROGRESS.json`, 파일별 해시는 `SOURCE_RECORDS.jsonl`, 캐시는 `CACHE_MANIFEST.json`, 최종 진단은 `COMPLETE.json`에 남습니다. 완성 전 캐시를 학습에 자동 편입하지 않습니다.

자료 조사는 `extracted/rfuav`의 XML·파일 목록과 `archives`의 ZIP 목차를 읽습니다. 두 번째 명령은 학습·검증 후보 파일의 세 위치를 제한적으로 읽어 수치 이상과 부분 중복을 검사합니다. 미지 기종 보류 파일은 읽지 않습니다. 전체 파일 무결성이나 수집 세션 독립성을 대신하는 검사가 아닙니다.

`src/drone_rf/model.py`는 기존 N의 64–128–256–512–1024 채널 본체를 유지합니다. 개별 신호 출력과 배경 출력의 합이 입력과 일치하도록 구성했습니다. `losses.py`는 두 개·세 개·네 개 출력의 순서를 전체 파형 단위로 맞춰 학습합니다. **아직 새 자료용 학습 실행기와 GPU 학습 결과는 포함하지 않습니다.** CPU 검사는 모델 축소 실험이나 RF 성능 평가가 아닙니다.

`src/drone_rf/data.py`는 고정한 혼합 목록을 실제 복소 I/Q로 재현합니다. 긴 문맥의 전력과 독립 위상으로 합성하고 짧은 창의 전력을 다시 맞추지 않습니다. 캐시 무결성과 역할 분리를 검사합니다. `scripts/audit_mixture_inputs.py`는 첫 학습 epoch와 검증 혼합을 CPU로 생성해 국소 전력차와 정답 합을 검산합니다. 정답 파형은 감독·검산용이며 모델 입력은 혼합 STFT입니다.

실행 중인 이전 실험은 기존 작업 디렉터리에 보존하고, 검증이 끝난 결과와 새 코드를 이 저장소로 옮깁니다. 원시 데이터·체크포인트·자격증명은 Git에 넣지 않습니다. 과거 연구의 결과를 새 다기종 모델의 성능으로 표시하지 않습니다.

이번 우선 범위는 최대 **세 신호**입니다. `context_model.py`는 전체 U-Net에 긴 혼합 문맥의 TCN·Transformer·양방향 LSTM을 연결하고, 별도 1/2/3개 신호 수 추정 출력을 제공합니다. `context_data.py`는 같은 복소 혼합에서 긴 문맥과 짧은 복원 창을 만듭니다. `scripts/audit_context_inputs.py`로 학습 자료의 105조건을 검산했습니다. 해당 모델들은 아직 학습하지 않았으며 실제 드론 대수 정답은 활동·수집 근거를 별도 확인해야 합니다.
