"""Keep the shared experiment lock without misclassifying remote display CUDA.

Only an explicitly pinned local display executable is admitted. Unknown CUDA
processes still prevent a new experiment; no process is terminated here.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import time

DISPLAY = Path('/usr/share/rustdesk/rustdesk')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(f)


def _digest(f):
    value = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024*1024), b''):
        value.update(chunk)
    return value.hexdigest()


def display_policy():
    return dict(executable=str(DISPLAY.resolve()), sha256=digest(DISPLAY),
                uid=Path('/proc/self').stat().st_uid, maximum_gpu_memory_mib=512,
                rationale='Observed RustDesk remote display CUDA context, not a research worker')


def process_identity(pid):
    folder = Path('/proc')/str(pid)
    executable = (folder/'exe').resolve(strict=True)
    return dict(executable=str(executable), sha256=digest(executable), uid=folder.stat().st_uid)


def classify(entries, own, predecessor, policy, identity=process_identity):
    allowed, waiting, unknown = [], [], []
    for entry in entries:
        pid = entry['pid']
        if pid == own:
            continue
        if pid == predecessor:
            waiting.append(entry)
            continue
        try:
            actual = identity(pid)
        except FileNotFoundError:
            continue  # Process exited after the nvidia-smi snapshot.
        except (PermissionError, OSError):
            unknown.append(entry)
            continue
        same = all(actual[k] == policy[k] for k in ('executable', 'sha256', 'uid'))
        if same and 0 <= entry['memory_mib'] <= policy['maximum_gpu_memory_mib']:
            allowed.append(dict(entry, **actual))
        else:
            unknown.append(dict(entry, **actual))
    return dict(allowed_display=allowed, waiting_predecessor=waiting, unknown=unknown)


def inspect(predecessor, policy):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    entries = []
    for line in raw.splitlines():
        if line.strip():
            pid, memory = line.split(',')
            entries.append(dict(pid=int(pid.strip()), memory_mib=int(memory.strip())))
    return classify(entries, os.getpid(), predecessor, policy)


def wait_for_predecessor(predecessor, policy):
    deadline = time.time()+60
    while True:
        state = inspect(predecessor, policy)
        if state['unknown']:
            raise RuntimeError('Unrecognized GPU compute process: '+repr(state['unknown']))
        if not state['waiting_predecessor']:
            return dict(state, checked_at=time.time(), shared_lock_required=True)
        if time.time() > deadline:
            raise RuntimeError('Predecessor CUDA context has not exited')
        time.sleep(1)
