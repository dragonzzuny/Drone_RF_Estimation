"""Preserve a published old stage, prioritize same-band training, then resume.

Never change frozen trainers/configurations. Signals are restricted to pinned
process identities. The old trial remains resumable with its original budget.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from drone_rf.context_training_data import write_json
from drone_rf.data import sha256


def read(path):
    return json.loads(Path(path).read_text())


def check_freeze(path):
    for name, digest in read(path)['files'].items():
        if sha256(name) != digest:
            raise RuntimeError('Frozen file changed: ' + name)


class PriorityQueue:
    def __init__(self, root):
        self.root = root
        self.q = root / 'queue'
        self.request = read(self.q / 'REQUEST.json')
        self.previous = Path(self.request['previous_experiment'])
        sys.path.insert(0, self.request['legacy_controller'])
        import controller140
        self.legacy = controller140
        self.common = controller140.C
        self.identity, self.same = controller140.identity, controller140.same

    def check(self):
        check_freeze(self.q / 'FREEZE.json')
        check_freeze(self.root / 'run/FREEZE.json')
        check_freeze(self.previous / 'run/FREEZE.json')
        self.legacy.checks()

    def state(self, status, **kw):
        result = dict(status=status, time=time.time(), controller=self.identity(os.getpid()),
                      execution_amendment=str(self.q / 'AMENDMENT_KO.md'), **kw)
        write_json(self.q / 'STATE.json', result)
        for parent in [self.previous / 'queue', self.legacy.P, self.legacy.PREV]:
            write_json(parent / 'STATE.json', dict(result, superseded_by=str(self.q / 'STATE.json')))
        print(json.dumps(result, ensure_ascii=False), flush=True)

    def require_identity(self, expected):
        if not self.same(self.identity(expected['pid']), expected):
            raise RuntimeError('Pinned process changed: ' + str(expected['pid']))

    def stop_exact(self, expected):
        self.require_identity(expected)
        os.kill(expected['pid'], signal.SIGSTOP)
        for _ in range(100):
            self.require_identity(expected)
            if self.identity(expected['pid'])['state'] in ('T', 't'):
                return
            time.sleep(.05)
        raise RuntimeError('Pinned process did not stop')

    def retire_stopped(self, expected):
        self.require_identity(expected)
        if self.identity(expected['pid'])['state'] not in ('T', 't'):
            raise RuntimeError('Retirement requires stopped process')
        os.kill(expected['pid'], signal.SIGTERM)
        os.kill(expected['pid'], signal.SIGCONT)
        for _ in range(100):
            if not self.same(self.identity(expected['pid']), expected):
                return
            time.sleep(.1)
        raise RuntimeError('Pinned process did not retire')

    def adopt(self):
        self.check()
        if (self.q / 'ADOPTION.json').exists():
            raise RuntimeError('Repeated adoption requires explicit recovery')
        old = self.request['controller']
        worker = self.request['worker']
        self.require_identity(old)
        self.require_identity(worker)
        self.common.suspended_ok()
        committed = False
        try:
            self.stop_exact(old)
            state = read(self.previous / 'queue/STATE.json')
            if state['status'] != 'CONTEXT_THREE_RUNNING' or not self.same(state['child'], worker):
                raise RuntimeError('Previous queue phase changed')
            self.require_identity(worker)
            self.retire_stopped(old)
            committed = True
            self.state('WAITING_OLD_STAGE_025', child=self.identity(worker['pid']))
            self.common.suspended_ok()
            write_json(self.q / 'ADOPTION.json', dict(time=time.time(), retired_controller=old,
                       unchanged_gpu_worker=worker, gpu_worker_signalled=False))
        finally:
            if not committed and self.same(self.identity(old['pid']), old):
                os.kill(old['pid'], signal.SIGCONT)
        return worker

    def stage_ready(self):
        run = self.previous / 'run'
        completed = read(run / 'COMPLETED_EPOCHS.json')
        for arm in ('ordered', 'mean'):
            if completed.get(arm, 0) < 25:
                return False
            for name in ['STAGE_025.json', 'SELECTED_025.json', 'SELECTED_025.pt']:
                if not (run / arm / name).exists():
                    return False
        return True

    def park_old(self, worker):
        import torch
        while not self.stage_ready():
            self.require_identity(worker)
            if (self.previous / 'run/FAILURE.json').exists():
                raise RuntimeError('Old worker failed before the stage boundary')
            self.state('WAITING_OLD_STAGE_025', child=self.identity(worker['pid']),
                       completed=read(self.previous / 'run/COMPLETED_EPOCHS.json'))
            time.sleep(5)
        run = self.previous / 'run'
        for _ in range(10):
            retired = False
            try:
                self.stop_exact(worker)
                completed = read(run / 'COMPLETED_EPOCHS.json')
                snapshots = {}
                coherent = True
                for arm in ('ordered', 'mean'):
                    p = run / arm / 'LAST.pt'
                    saved = torch.load(p, map_location='cpu', weights_only=False)
                    coherent &= (saved['epoch'] == completed[arm]
                        and saved['updates'] == completed[arm] * 75
                        and saved['freeze_sha256'] == sha256(run / 'FREEZE.json')
                        and (run / arm / f'EPOCH_{completed[arm]:03d}.json').exists())
                    del saved
                    stage = read(run / arm / 'STAGE_025.json')
                    if stage['checkpoint_sha256'] != sha256(run / arm / 'SELECTED_025.pt'):
                        raise RuntimeError('Published stage checkpoint hash mismatch')
                    snapshots[arm] = dict(epoch=completed[arm], original=str(p), sha256=sha256(p),
                        selected_25_sha256=stage['checkpoint_sha256'])
                if not coherent:
                    continue  # Let an in-flight atomic checkpoint transaction finish.
                archive = self.q / 'old_resume_checkpoints'
                archive.mkdir(exist_ok=True)
                for arm, meta in snapshots.items():
                    target = archive / (arm + '_LAST.pt')
                    shutil.copy2(meta['original'], target)
                    if sha256(target) != meta['sha256']:
                        raise RuntimeError('Resume checkpoint copy mismatch')
                    meta['backup'] = str(target)
                self.retire_stopped(worker)
                retired = True
                receipt = dict(status='OLD_STAGE_PRESERVED_FOR_RESUME', time=time.time(),
                    worker=worker, completed=completed, checkpoints=snapshots,
                    original_training_budget_unchanged=True, remaining_epochs_not_cancelled=True,
                    discarded_uncommitted_work='Any next-epoch work after the saved boundary is replayed on resume')
                write_json(self.q / 'OLD_PARKED.json', receipt)
                write_json(run / 'PARKED_FOR_SAME_BAND.json', receipt)
                self.state('OLD_STAGE_PRESERVED_FOR_RESUME', completed=completed)
                return
            finally:
                if not retired and self.same(self.identity(worker['pid']), worker):
                    os.kill(worker['pid'], signal.SIGCONT)
                    time.sleep(1)
        raise RuntimeError('No coherent checkpoint boundary found; old worker left running')

    def resource(self):
        while True:
            result = self.common.gate()
            if result['allowed']:
                return
            self.state('WAITING_GPU_RESOURCE', resource=result)
            time.sleep(20)

    def child(self, argv, label, env=None):
        self.check()
        with (self.q / (label.lower() + '.log')).open('a') as log:
            process = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=env)
            while True:
                self.state(label + '_RUNNING', child=self.identity(process.pid), argv=argv)
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    pass
        self.state(label + '_EXITED', returncode=code)
        if code:
            raise RuntimeError(f'{label} failed with code {code}; no automatic restart')

    def run(self):
        with (self.q / 'CONTROLLER.lock').open('a') as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = self.adopt()
            with (self.legacy.N / 'neural_queue.lock').open('a') as shared:
                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.park_old(worker)
                self.resource()
                snap = self.root / 'training_snapshot'
                env = {**os.environ, 'PYTHONPATH': str(snap), 'OMP_NUM_THREADS': '2',
                       'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
                argv = [sys.executable, '-u', str(snap / 'train_context_experiment.py'),
                        '--preparation', str(self.root / 'preparation'), '--run', str(self.root / 'run')]
                self.child(argv + ['--preflight-only'], 'SAME_BAND_GPU_PREFLIGHT', env)
                self.child(argv, 'SAME_BAND_TRAINING', env)
                if read(self.root / 'run/COMPLETE.json')['status'] != 'MATCHED_CONTEXT_DEVELOPMENT_COMPLETE':
                    raise RuntimeError('New comparison incomplete')
                # Original CPU worker has stayed bounded by old training progress.
                self.require_identity(self.request['old_feature_worker'])
                oldsnap = self.previous / 'training_snapshot'
                self.child([sys.executable, '-u', str(oldsnap / 'train_context_experiment.py'),
                    '--preparation', str(self.previous / 'preparation'), '--run', str(self.previous / 'run')],
                    'OLD_CONTEXT_RESUME', {**env, 'PYTHONPATH': str(oldsnap)})
                self.common.suspended_ok()
                self.resource()
                self.child([sys.executable, '-u', str(self.legacy.N / 'train_neural91.py'),
                            'train', 'D1', 'rf_transformer'], 'NATIVE_RF_RESUME')
                self.child([sys.executable, '-u', str(self.legacy.PREV / 'wrappers136.py'),
                            'native_evaluate'], 'NATIVE_RF_EVALUATION')
            self.state('COMPLETE_SAME_BAND_AND_PRESERVED_TRIALS', research_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    queue = PriorityQueue(args.experiment)
    try:
        queue.run()
    except Exception:
        failure = dict(status='FAILED', time=time.time(), pid=os.getpid(), traceback=traceback.format_exc())
        write_json(queue.q / 'FAILURE.json', failure)
        if (queue.q / 'ADOPTION.json').exists():
            queue.state('FAILED_REQUIRES_REVIEW', failure=failure)
        raise
