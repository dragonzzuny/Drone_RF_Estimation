"""Adopt the live legacy magnitude worker and insert the new context comparison.

Only retire the exact former scheduler. Never signal its GPU worker or edit
frozen legacy source. Preserve and later resume the native RF comparison.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from drone_rf.context_training_data import write_json
from drone_rf.data import sha256


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = args.experiment / 'queue'
        self.root.mkdir(exist_ok=True)
        self.legacy = args.legacy_controller
        sys.path.insert(0, str(self.legacy))
        import controller140
        self.old = controller140
        self.common = controller140.C
        self.identity, self.same = controller140.identity, controller140.same

    def check(self):
        self.old.checks()
        recipe = json.loads((self.args.experiment / 'run/FREEZE.json').read_text())
        for path, digest in recipe['files'].items():
            if sha256(path) != digest:
                raise RuntimeError(f'New frozen source changed: {Path(path).name}')
        frozen = json.loads((self.root / 'FREEZE.json').read_text())
        for path, digest in frozen['files'].items():
            if sha256(path) != digest:
                raise RuntimeError(f'Queue freeze changed: {Path(path).name}')
        return frozen

    def state(self, status, **kw):
        result = dict(status=status, time=time.time(), controller=self.identity(os.getpid()),
                      amendment=str(self.root / 'AMENDMENT_KO.md'), **kw)
        write_json(self.root / 'STATE.json', result)
        # Legacy suspended guards and status readers continue to see the real
        # controller identity. Only mutable status files are updated.
        for folder in (self.legacy, self.old.PREV):
            write_json(folder / 'STATE.json', dict(result, superseded_by=str(self.root / 'STATE.json')))
        if status.startswith('FULL_N_MAGNITUDE'):
            write_json(self.old.Q139 / 'QUEUE_STATE.json', result)
        elif status.startswith('WINDOW_'):
            write_json(self.old.Q130 / 'QUEUE_STATE.json', result)
        elif status.startswith('NATIVE_RF'):
            write_json(self.old.D / 'STATE.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    def adopt(self):
        frozen = self.check()
        expected = frozen['previous_state']
        old, worker = expected['controller'], expected['child']
        if (self.root / 'ADOPTION.json').exists():
            raise RuntimeError('Repeated adoption requires explicit state recovery')
        self.common.suspended_ok()
        if not self.same(self.identity(old['pid']), old) or not self.same(self.identity(worker['pid']), worker):
            raise RuntimeError('Expected scheduler/worker changed before adoption')
        if expected['status'] != 'FULL_N_MAGNITUDE_RUNNING':
            raise RuntimeError('Only the live magnitude phase can be adopted')
        stopped, retired = False, False
        try:
            os.kill(old['pid'], signal.SIGSTOP); stopped = True
            for _ in range(100):
                current = self.identity(old['pid'])
                if current and current['state'] in ('T', 't'):
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('Old scheduler did not stop')
            state = json.loads((self.legacy / 'STATE.json').read_text())
            if state['status'] != expected['status'] or not self.same(state['child'], worker):
                raise RuntimeError('Legacy phase changed during adoption')
            if not self.same(self.identity(worker['pid']), worker):
                raise RuntimeError('Live worker changed during adoption')
            os.kill(old['pid'], signal.SIGTERM)
            os.kill(old['pid'], signal.SIGCONT)
            for _ in range(100):
                if not self.same(self.identity(old['pid']), old):
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Old scheduler did not retire')
            retired = True
            self.state('FULL_N_MAGNITUDE_RUNNING', child=self.identity(worker['pid']), adopted=True,
                       context_model_training='QUEUED_AFTER_MAGNITUDE_AND_WINDOW_AUDIT')
            self.common.suspended_ok()
            write_json(self.root / 'ADOPTION.json', dict(status='SAME_MAGNITUDE_WORKER_CONTINUES',
                time=time.time(), retired_scheduler=old, worker_before=worker,
                worker_after=self.identity(worker['pid']), worker_signalled=False,
                worker_restarted=False, exit_code_available=False))
        except BaseException:
            if stopped and not retired and self.same(self.identity(old['pid']), old):
                os.kill(old['pid'], signal.SIGCONT)
            raise
        return worker

    def resource(self):
        while True:
            result = self.common.gate()
            if result['allowed']:
                return
            self.state('WAITING_GPU_RESOURCE', resource=result)
            time.sleep(20)

    def child(self, argv, label, *, env=None):
        self.check()
        with (self.root / f'{label.lower()}.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=self.args.experiment, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while True:
                self.state(label + '_RUNNING', child=self.identity(process.pid), argv=argv)
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    pass
        self.state(label + '_EXITED', returncode=code, child_pid=process.pid)
        if code != 0:
            raise RuntimeError(f'{label} exited {code}; automatic restart is disabled')

    def run(self):
        with (self.root / 'CONTROLLER.lock').open('a') as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = self.adopt()
            while self.same(self.identity(worker['pid']), worker):
                self.state('FULL_N_MAGNITUDE_RUNNING', child=self.identity(worker['pid']), adopted=True,
                           context_model_training='QUEUED_AFTER_MAGNITUDE_AND_WINDOW_AUDIT')
                time.sleep(20)
            if (self.old.Q139 / 'FAILURE.json').exists():
                raise RuntimeError('Adopted magnitude worker failed')
            if self.old.read(self.old.Q139 / 'GPU_COMPLETE.json')['status'] != 'FULL_TRAINING_AND_SAVED_EVALUATION_COMPLETE':
                raise RuntimeError('Magnitude completion receipt missing')
            self.child([sys.executable, '-u', str(self.old.Q139 / 'audit139.py')], 'FULL_N_MAGNITUDE_AUDIT')
            if self.old.read(self.old.Q139 / 'AMPLITUDE_AUDIT.json')['status'] != 'PASS_ALL_STAGED_AMPLITUDE_METRICS':
                raise RuntimeError('Magnitude saved outputs did not pass audit')
            write_json(self.old.Q139 / 'QUEUE_STATE.json', dict(status='COMPLETE', time=time.time(), epochs=50, updates=3750))
            self.resource()
            self.child([sys.executable, '-u', str(self.old.Q130 / 'run130.py'), 'capture'], 'WINDOW_CAPTURE')
            if self.old.read(self.old.Q130 / 'GPU_COMPLETE.json')['status'] != 'PASS_ALL_FULL_MODEL_WINDOW_OUTPUTS_SAVED':
                raise RuntimeError('Window capture incomplete')
            self.child([sys.executable, '-u', str(self.old.Q130 / 'analyze130.py')], 'WINDOW_NUMPY_ANALYSIS')
            if self.old.read(self.old.Q130 / 'RESULT_AUDIT.json')['status'] != 'PASS_ALL_270_MERGES_AND_90_WINDOW_DIAGNOSTICS':
                raise RuntimeError('Window analysis incomplete')
            write_json(self.old.Q130 / 'QUEUE_STATE.json', dict(status='COMPLETE', time=time.time(), full_outputs=270))
            self.common.suspended_ok()
            self.resource()
            with (self.old.N / 'neural_queue.lock').open('a') as shared:
                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                snapshot = self.args.experiment / 'training_snapshot'
                self.child([sys.executable, '-u', str(snapshot / 'train_context_experiment.py'),
                    '--preparation', str(self.args.experiment / 'preparation'),
                    '--run', str(self.args.experiment / 'run')], 'CONTEXT_THREE',
                    env={**os.environ, 'PYTHONPATH': str(snapshot), 'OMP_NUM_THREADS': '2',
                         'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'})
            if self.old.read(self.args.experiment / 'run/COMPLETE.json')['status'] != 'MATCHED_CONTEXT_DEVELOPMENT_COMPLETE':
                raise RuntimeError('Context comparison incomplete')
            self.common.suspended_ok()
            self.resource()
            with (self.old.N / 'neural_queue.lock').open('a') as shared:
                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.child([sys.executable, '-u', str(self.old.N / 'train_neural91.py'),
                            'train', 'D1', 'rf_transformer'], 'NATIVE_RF_RESUME')
                if not (self.old.N / 'neural_run/D1/rf_transformer/COMPLETE.json').exists():
                    raise RuntimeError('Native RF training incomplete')
                self.child([sys.executable, '-u', str(self.old.PREV / 'wrappers136.py'),
                            'native_evaluate'], 'NATIVE_RF_EVALUATION')
            if self.old.read(self.old.PREV / 'NATIVE_ARCHIVE_COMPLETE.json')['status'] != 'PASS_ALL_20_NATIVE_OUTPUTS_SAVED':
                raise RuntimeError('Native output archival incomplete')
            self.check()
            write_json(self.old.D / 'STATE.json', dict(status='COMPLETE', time=time.time(),
                execution_amendment=str(self.root / 'AMENDMENT_KO.md'), D2_unchanged=True,
                all_native_outputs_archived=True))
            self.state('COMPLETE_CONTEXT_AND_LEGACY_EXECUTION', research_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--legacy-controller', type=Path, required=True)
    args = parser.parse_args()
    queue = Queue(args)
    try:
        queue.run()
    except Exception:
        failure = dict(status='FAILED', time=time.time(), pid=os.getpid(), traceback=traceback.format_exc())
        write_json(queue.root / 'FAILURE.json', failure)
        write_json(args.experiment / 'run/FAILURE.json', dict(failure, upstream_execution_failure=True))
        # Before adoption, do not overwrite a still-live legacy scheduler state.
        if (queue.root / 'ADOPTION.json').exists():
            queue.state('FAILED', failure=failure)
        raise
