#!/usr/bin/env bash
# Run ONCE in a normal host terminal. Preserve old checkpoints, then train the
# already prepared baseline/gated U-Net pair. Exact identities are verified.
set -euo pipefail
rf_switch_base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rf_switch_python='/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/.venv_temporal_v65/bin/python'
rf_switch_runtime='/home/pyj/문서/GitHub/uav_analysis/scripts/with_uav_cuda.sh'
rf_switch_probe='import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
rf_switch_command=("$rf_switch_python")
# Prefer the host's working CUDA runtime. The older compatibility libraries
# are valid only for the exact kernel version they were verified against.
if ! compgen -G '/dev/nvidia[0-9]*' > /dev/null; then
  echo 'GPU_DEVICE_NOT_VISIBLE: 이 실행 환경에는 NVIDIA GPU 장치가 보이지 않습니다. 기존 작업은 변경하지 않았습니다.' >&2
  exit 1
fi
if "$rf_switch_python" -c "$rf_switch_probe"; then
  echo 'CUDA 확인: 현재 환경의 기본 런타임을 사용합니다.'
elif [[ -r /proc/driver/nvidia/version ]] && \
     rg -q 'Kernel Module[[:space:]]+580\.173\.02' /proc/driver/nvidia/version && \
     bash "$rf_switch_runtime" "$rf_switch_python" -c "$rf_switch_probe"; then
  rf_switch_command=(bash "$rf_switch_runtime" "$rf_switch_python")
  echo 'CUDA 확인: 커널 버전과 일치하는 검증된 호환 런타임을 사용합니다.'
else
  echo 'CUDA_UNAVAILABLE: 현재 런타임에서 CUDA 확인에 실패했습니다. 기존 작업은 변경하지 않았습니다.' >&2
  exit 1
fi
mkdir -p "$rf_switch_base/dense_gpu_run"
nohup "${rf_switch_command[@]}" -u \
  "$rf_switch_base/launch/host_switch.py" --apply \
  >> "$rf_switch_base/dense_gpu_run/host_transition.log" 2>&1 < /dev/null &
rf_switch_pid=$!
printf '전환 실행기 PID: %s\n로그: %s\n이 메시지는 GPU 학습 시작 확인이 아닙니다.\n' \
  "$rf_switch_pid" "$rf_switch_base/dense_gpu_run/host_transition.log"
