"""User-run host transition: preserve the old experiment, start the dense pair.

This cannot run through an isolated Codex session. Default is read-only check;
--apply is the explicit host-terminal action. No SSH/daemon or privilege bypass.
Only the two exact process identities from the known old queue may be signalled.
"""
import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch


BASE = Path(__file__).resolve().parents[1]
LOCAL = Path('/home/pyj/문서/GitHub/Drone_RF_Estimation/local')
OLD = LOCAL / 'context_experiment_20261006_v1'
QUEUE = LOCAL / 'within_dataset_context_20261007_v1/queue'
RUN = BASE / 'dense_gpu_run'
SHARED_LOCK = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024**2):
            h.update(data)
    return h.hexdigest()


def status(stage, **kwargs):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **kwargs)
    path = RUN / 'HOST_TRANSITION.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def identity(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
            argv=[v.decode() for v in (root / 'cmdline').read_bytes().split(b'\0') if v],
            exe=str((root / 'exe').resolve(strict=True)))
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(actual, expected):
    return bool(actual and actual['state'] not in ('Z', 'X') and
                all(actual[k] == expected[k] for k in ('pid', 'start_ticks', 'argv', 'exe')))


def check_queue(state):
    if state.get('status') != 'OLD_CONTEXT_RESUME_RUNNING':
        raise RuntimeError('Old queue is in a different phase; no processes will be changed')
    controller, worker = state['controller'], state['child']
    controller_script = str(QUEUE / 'resource_revision_001/resume_same_band_resource.py')
    worker_script = str(OLD / 'training_snapshot/train_context_experiment.py')
    if (controller_script not in controller['argv'] or worker_script not in worker['argv']
            or str(OLD / 'run') not in worker['argv'] or controller['pid'] == worker['pid']):
        raise RuntimeError('Unexpected controller/worker command; no processes will be changed')
    for expected in (controller, worker):
        if not matches(identity(expected['pid']), expected):
            raise RuntimeError('Host process unavailable or identity changed; use a normal host terminal')
    return controller, worker


def checked_signal(expected, sig):
    # pidfd prevents PID-reuse races; no broad kill/pkill and no process groups.
    if not matches(identity(expected['pid']), expected):
        raise RuntimeError('Refusing to signal a changed process identity')
    fd = pidfd_open(expected['pid'])
    try:
        if not matches(identity(expected['pid']), expected):
            raise RuntimeError('Process changed while acquiring its pidfd')
        pidfd_signal(fd, sig)
    finally:
        os.close(fd)


def linux_syscall(number, *args):
    # The conda Python build lacks os.pidfd_open and signal.pidfd_send_signal.
    # Use the same kernel interfaces, preserving normal namespace/UID checks.
    # Constants verified against this x86_64 host's asm/unistd_64.h.
    # https://man7.org/linux/man-pages/man2/pidfd_open.2.html
    # https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
    if platform.machine() != 'x86_64':
        raise RuntimeError('This host launcher supports the verified x86_64 ABI only')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    return linux_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(fd, sig):
    return linux_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def stop(expected):
    checked_signal(expected, signal.SIGSTOP)
    for _ in range(100):
        actual = identity(expected['pid'])
        if matches(actual, expected) and actual['state'] in ('T', 't'):
            return
        time.sleep(.05)
    raise RuntimeError('Exact process did not stop')


def retire(expected):
    actual = identity(expected['pid'])
    if not matches(actual, expected) or actual['state'] not in ('T', 't'):
        raise RuntimeError('Retirement requires a verified stopped process')
    checked_signal(expected, signal.SIGTERM)
    # Deliver the pending termination to a stopped process.
    checked_signal(expected, signal.SIGCONT)
    for _ in range(100):
        if not matches(identity(expected['pid']), expected):
            return
        time.sleep(.1)
    raise RuntimeError('Old process did not exit; checkpoint files remain preserved')


def check_inputs():
    frozen = read(RUN / 'FREEZE.json')
    for name, expected in frozen['source_files'].items():
        if digest(BASE / name) != expected:
            raise RuntimeError('Frozen training source changed: ' + name)
    for name, expected in frozen['dataset_files'].items():
        p = Path(name)
        if digest(p if p.is_absolute() else BASE / p) != expected:
            raise RuntimeError('Frozen input/checkpoint changed: ' + p.name)
    prep = BASE / 'dense_preparation'
    if read(prep / 'FEATURES_COMPLETE.json')['status'] != 'ALL_DENSE_FEATURES_READY':
        raise RuntimeError('Dense inputs incomplete')
    for role, epochs in [('train_pack', range(1, 6)), ('validation_pack', (1,))]:
        for epoch in epochs:
            stem = prep / 'features' / f'{role}_{epoch:03d}'
            receipt = read(stem.with_suffix('.json'))
            if receipt['status'] != 'FEATURES_READY' or digest(stem.with_suffix('.npy')) != receipt['feature_sha256']:
                raise RuntimeError('Feature cache mismatch')
    if frozen['arms'] != ['unet_mean', 'unet_gated'] or not frozen['dense_corpus_used']:
        raise RuntimeError('Not the authorized two-arm dense comparison')


def gpu_pids():
    value = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           check=True, capture_output=True, text=True, timeout=15)
    return {int(line.strip()) for line in value.stdout.splitlines() if line.strip()}


def unexpected_gpu_pids(allowed, display=None):
    remaining = gpu_pids() - set(allowed)
    if display and display['identity']['pid'] in remaining:
        expected = display['identity']
        limit = float(display['maximum_memory_mib'])
        if (expected['exe'] != '/usr/share/rustdesk/rustdesk'
                or expected['argv'] != ['/usr/share/rustdesk/rustdesk', '--server']
                or not 0 < limit <= 256
                or not matches(identity(expected['pid']), expected)
                or (Path('/proc') / str(expected['pid'])).stat().st_uid != os.getuid()):
            raise RuntimeError('Pinned display identity changed; no unrelated process will be signalled')
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader,nounits'], check=True,
                                capture_output=True, text=True, timeout=15)
        memory = {int(row.split(',')[0]): float(row.split(',')[1])
                  for row in result.stdout.splitlines() if row.strip()}
        value = memory.get(expected['pid'], float('nan'))
        if not math.isfinite(value) or not 0 <= value <= limit or not matches(identity(expected['pid']), expected):
            raise RuntimeError('Pinned display memory/identity check failed')
        remaining.remove(expected['pid'])
    return remaining


def preserve_checkpoints(backup):
    completed = read(OLD / 'run/COMPLETED_EPOCHS.json')
    expected_freeze = digest(OLD / 'run/FREEZE.json')
    records = {}
    for arm in ('ordered', 'mean'):
        path = OLD / 'run' / arm / 'LAST.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['epoch'] != completed[arm] or saved['updates'] != 75 * completed[arm]
                or saved['freeze_sha256'] != expected_freeze
                or not (path.parent / f'EPOCH_{completed[arm]:03d}.json').exists()):
            raise RuntimeError('Checkpoint/epoch receipt update in progress; original processes will resume')
        if not all(k in saved for k in ('optimizer', 'torch_rng', 'cuda_rng')):
            raise RuntimeError('Missing resume state')
        del saved
        gc.collect()
        target = backup / f'{arm}_LAST.pt'
        shutil.copyfile(path, target)
        original_hash = digest(path)
        if digest(target) != original_hash:
            raise RuntimeError('Checkpoint backup mismatch')
        records[arm] = dict(epoch=completed[arm], updates=75 * completed[arm],
                            original=str(path), backup=str(target), sha256=original_hash)
    for name in ('COMPLETED_EPOCHS.json', 'PROGRESS.json', 'FREEZE.json'):
        shutil.copyfile(OLD / 'run' / name, backup / name)
    (backup / 'PRESERVED.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def transition(display=None):
    initial = read(QUEUE / 'STATE.json')
    controller, worker = check_queue(initial)
    if unexpected_gpu_pids({worker['pid'], os.getpid()}, display):
        raise RuntimeError('Other GPU compute work is active; it will not be touched')
    backup = RUN / 'legacy_preserved' / time.strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    paused, retired = [], False
    try:
        paused.append(controller)
        stop(controller)
        current = read(QUEUE / 'STATE.json')
        other_controller, other_worker = check_queue(current)
        if other_controller['pid'] != controller['pid'] or other_worker['pid'] != worker['pid']:
            raise RuntimeError('Queue changed while pausing; abort transition')
        paused.append(worker)
        stop(worker)
        checkpoints = preserve_checkpoints(backup)
        status('OLD_CHECKPOINTS_PRESERVED', checkpoints=checkpoints,
               untouched_display_admission=display,
               old_progress_may_be_rolled_back_to_saved_epoch=True)
        # Retire the queue first so it cannot start another legacy GPU task.
        retire(controller)
        retired = True
        retire(worker)
        paused.clear()
        status('OLD_QUEUE_RETIRED_NEW_TRAINING_PENDING', checkpoints=checkpoints,
               untouched_original_checkpoints=True)
    except BaseException:
        # Before retiring the controller, failed checks roll back the pause.
        # After retirement, resume the worker if needed, but do not pretend that
        # the queue was restored. The saved checkpoint receipt remains explicit.
        for expected in reversed(paused):
            if matches(identity(expected['pid']), expected):
                checked_signal(expected, signal.SIGCONT)
        status('TRANSITION_FAILED', old_controller_retired=retired, backup=str(backup))
        raise
    if not SHARED_LOCK.exists():
        raise RuntimeError('Existing shared GPU lock not found')
    with SHARED_LOCK.open('r') as shared:
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(30):
            if not unexpected_gpu_pids({os.getpid()}, display):
                break
            time.sleep(1)
        else:
            raise RuntimeError('GPU compute processes remain; no new training started')
        log = RUN / 'training.log'
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        argv = [sys.executable, '-u', 'study.py', '--preparation', './dense_preparation',
                '--run', './dense_gpu_run', '--epochs', '5']
        with log.open('a') as output:
            process = subprocess.Popen(argv, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            status('DENSE_COMPARISON_PROCESS_STARTED', child_pid=process.pid, log=str(log),
                   gpu_preflight_not_yet_confirmed=True)
            code = process.wait()
        status('DENSE_COMPARISON_EXITED', returncode=code,
               milestone_complete=(RUN / 'MILESTONE_005.json').exists())
        if code:
            raise RuntimeError('New comparison exited with an error; see training.log; no automatic restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.chdir(BASE)
    RUN.mkdir(exist_ok=True)
    with (RUN / 'HOST_SWITCH.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not list(Path('/dev').glob('nvidia*')) or not torch.cuda.is_available():
            raise SystemExit('HOST_GPU_UNAVAILABLE: no process was signalled. Run from a normal GPU host terminal.')
        probe = pidfd_open(os.getpid())
        try:
            pidfd_signal(probe, 0)  # Permission probe only; does not deliver a signal.
        finally:
            os.close(probe)
        check_inputs()
        check_queue(read(QUEUE / 'STATE.json'))
        if args.apply:
            display_path = BASE / 'launch/GPU_DISPLAY_ADMISSION.json'
            transition(read(display_path) if display_path.exists() else None)
        else:
            print('READ_ONLY_CHECK_PASSED: no process was signalled; --apply performs the transition.')
