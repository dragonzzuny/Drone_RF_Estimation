# Drone RF Estimation

드론의 혼합 복소 I/Q에서 각 신호를 복원하고, 학습에 사용하지 않은 기록·기종으로 일반화하는지 검증하는 연구입니다. 2026-10-06부터 이 저장소를 코드·계획·결과의 공유 위치로 사용합니다.

**2026-10-09 갱신:** RFUAV 같은 RF 대역 안에서 원 수신 중심주파수 간격을 보존한 실험의 추가 5 epoch/375업데이트를 완료했습니다. 공통 관측 대역으로 제한한 기여 파형을 복원하며, 이전 중심 정렬 수치와 동일 관측의 개선으로 비교하지 않습니다. 선택된 원 규모 U-Net(e2)의 두·세 성분 NMSE는 **0.466896/0.690253**, 같은 모델의 네 위상 평균은 **0.446896/0.689089**입니다. 약신호 정밀 복원·새 기종 일반화·물리 드론 대수 추정 목표는 아직 달성하지 못했습니다.

개수 손실의 공통 인코더 기울기만 차단하는 동일 부모·자료·5 epoch 대조와 네 위상 평가도 완료·검산했습니다. 세 성분은 개선됐지만 두 성분 NMSE는 악화해 전체 우월성으로 채택하지 않습니다. 구조 검토에서 긴 문맥의 시간 평균과 원 규모 WaveNet의 짧은 수용영역을 확인했습니다. native 전체 입력의 WaveNet/U-Net GPU 적합도 진단을 시작했고, 통과 후 각 새 초기화·같은5epoch/375업데이트를 적용하는 세 구조 비교를 등록했습니다. 실행 상태는 기록 시점이며 해당 worker의 상태 파일로 확인합니다.

- [원 주파수 배치 실험 전체 결과](reports/2026-10-09/NATIVE_RF_FINAL.md)
- [현재 구조 적합성·WaveNet·최신 RF U-Net 검토](reports/2026-10-09/ARCHITECTURE_REVIEW_KO.md)
- [개수 기울기 차단 대조·네 위상 평가 완료](reports/2026-10-09/COUNT_DETACH_FINAL.md)
- [학습 곡선 PDF](reports/2026-10-09/NATIVE_RF_EPOCHS.pdf)
- [위상 평균과 학습/새 혼합 적합도 진단](reports/2026-10-09/NATIVE_RF_INFERENCE_DIAGNOSIS.md)
- [개수 손실 기울기 진단과 후속 대조](reports/2026-10-09/NATIVE_RF_GRADIENT_DIAGNOSIS.md)
- [개수 분류·정답 보조 전력 분배 진단](reports/2026-10-09/NATIVE_RF_CPU_DIAGNOSTICS.md)
- [원 주파수 합성 및 실행 규약](experiments/rfuav_native_frequency_20261009/README.md)
- [이전 공통 6 epoch 결과](reports/2026-10-07/PROGRESS_1300_KO.md)
- [지도 파형 복원용 데이터 범위 확대와 조합 후보](reports/2026-10-07/DATA_VOLUME_AND_COMBINATIONS_KO.md)
- [첫 동일 예산 결과와 검증 보완](reports/2026-10-07/SAME_BAND_EVIDENCE_REVIEW_KO.md)
- [같은 대역 GPU 실행과 이전 25 epoch 결과](reports/2026-10-07/SAME_BAND_LAUNCH_KO.md)
- [같은 대역 주 실험 실행 규약](docs/SAME_BAND_EXECUTION_KO.md)
- [실제 U-Net+TCN 구조와 관측 시간](docs/TEMPORAL_MODEL_KO.md)
- [목표와 실험계획](docs/PLAN_KO.md)
- [최대 세 성분 대조 실행 규약](docs/CONTEXT_EXPERIMENT_KO.md)
- [CPU 준비 완료·GPU 실행 연결](reports/2026-10-06/CONTEXT_EXECUTION_KO.md)
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

`src/drone_rf/model.py`는 기존 N의 64–128–256–512–1024 채널 본체를 유지합니다. 개별 신호 출력과 배경 출력의 합이 입력과 일치하도록 구성했습니다. `losses.py`는 두 개·세 개·네 개 출력의 순서를 전체 파형 단위로 맞춰 학습합니다. 현재 실행은 최대 세 성분입니다. CPU 구조 검사는 모델 축소 실험이나 RF 성능 평가가 아닙니다.

`src/drone_rf/data.py`는 고정한 혼합 목록을 실제 복소 I/Q로 재현합니다. 긴 문맥의 전력과 독립 위상으로 합성하고 짧은 창의 전력을 다시 맞추지 않습니다. 캐시 무결성과 역할 분리를 검사합니다. `scripts/audit_mixture_inputs.py`는 첫 학습 epoch와 검증 혼합을 CPU로 생성해 국소 전력차와 정답 합을 검산합니다. 정답 파형은 감독·검산용이며 모델 입력은 혼합 STFT입니다.

이전 실험 자료는 기존 작업 디렉터리에 보존하고, 검증이 끝난 결과와 새 코드를 이 저장소로 옮깁니다. 원시 데이터·체크포인트·자격증명은 Git에 넣지 않습니다. 과거 연구의 결과를 새 다기종 모델의 성능으로 표시하지 않습니다.

이번 우선 범위는 최대 **세 신호**입니다. `context_model.py`에는 TCN·Transformer·양방향 LSTM 후보가 구현되어 있으며, 초기 실행에서는 **TCN 두 군**을 비교했습니다. 최신 실행은 위 갱신 항목과 개별 실험 규약을 따릅니다. `context_data.py`는 같은 복소 혼합에서 긴 문맥과 짧은 복원 창을 만듭니다. 별도 1/2/3개 추정 head의 정답은 합성 기록 성분 수입니다. 실제 드론 대수 정답은 활동·수집 근거를 별도 확인해야 합니다. 과거 105조건 CPU 입력 검산과 새 같은 대역 비교의 범위를 구분합니다.

`audit_context_evidence.py`는 저장된 같은 epoch의 두 군을 대조해 2/3성분만의 복원 성능과 국소 전력별 결과를 추가로 계산합니다. `watch_context_evidence.py`는 이를 CPU에서 단계별로 생성합니다. 정답 I/Q를 새로 열거나 학습의 체크포인트 선택 규칙을 바꾸지 않습니다.

`prepare_context_experiment.py`는 1/2/3개 구성 수를 균등 배정하고 공통 문맥 특징을 CPU에서 준비합니다. `train_context_experiment.py`는 전체 입력 GPU 검사와 고정 혼합 적합 진단을 거친 뒤 ordered/mean 두 군을 같은 초기화·예산으로 학습합니다. `context_execution_queue.py`는 기존 GPU worker를 유지하면서 새 비교를 이어 실행합니다. 수집 한계와 합성 성분 수/물리 드론 대수의 구분은 실행 규약을 따릅니다.
