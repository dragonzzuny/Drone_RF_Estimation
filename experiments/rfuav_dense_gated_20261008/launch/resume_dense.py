"""Continue the frozen dense comparison after the old queue was preserved.

Foreground supervisor: holds both existing locks; never signals a process.
Default checks only. --apply starts the reviewed five-epoch comparison.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import host_switch as h


def main(apply=False):
    os.chdir(h.BASE)
    with (h.RUN / 'HOST_SWITCH.lock').open('a') as own, h.SHARED_LOCK.open('r') as shared:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not h.torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        h.check_inputs()
        previous = h.read(h.RUN / 'HOST_TRANSITION.json')
        if previous['stage'] not in ('DENSE_COMPARISON_PROCESS_STARTED', 'DENSE_COMPARISON_EXITED'):
            raise RuntimeError('Unreviewed transition state')
        for key in ('pid', 'child_pid'):
            pid = previous.get(key)
            if pid and pid != os.getpid() and Path('/proc', str(pid)).exists():
                raise RuntimeError('Previous supervisor/worker PID still exists; review its identity')
        old = h.read(h.QUEUE / 'STATE.json')
        for expected in (old['controller'], old['child']):
            if h.matches(h.identity(expected['pid']), expected):
                raise RuntimeError('Old queue/worker still active')
        receipts = sorted((h.RUN / 'legacy_preserved').glob('*/PRESERVED.json'))
        if not receipts:
            raise RuntimeError('No saved old experiment receipt')
        receipt = receipts[-1]
        for record in h.read(receipt).values():
            for key in ('original', 'backup'):
                if h.digest(Path(record[key])) != record['sha256']:
                    raise RuntimeError('Old saved checkpoint changed')
        display = h.read(h.BASE / 'launch/GPU_DISPLAY_ADMISSION.json')
        if h.unexpected_gpu_pids({os.getpid()}, display):
            raise RuntimeError('Other compute work is active')
        if not apply:
            print('PRESERVED_COMPARISON_CHECK_PASSED: no process launched', flush=True)
            return 0
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        argv = [sys.executable, '-u', 'study.py', '--preparation', './dense_preparation',
                '--run', './dense_gpu_run', '--epochs', '5']
        log = h.RUN / 'training.log'
        with log.open('a') as output:
            process = subprocess.Popen(argv, cwd=h.BASE, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            h.status('DENSE_COMPARISON_PROCESS_STARTED', child_pid=process.pid, log=str(log),
                     foreground_supervisor=True, preserved_receipt=str(receipt),
                     previous_supervisor_disappeared_without_exit_receipt=True)
            code = process.wait()
        h.status('DENSE_COMPARISON_EXITED', returncode=code,
                 milestone_complete=(h.RUN / 'MILESTONE_005.json').exists())
        return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    raise SystemExit(main(parser.parse_args().apply))
