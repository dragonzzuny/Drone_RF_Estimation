"""Resume a safely parked queue with one pinned, bounded display process."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from within_band_priority_queue import PriorityQueue, check_freeze, read
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256


def admit_display(result, expected, maximum_mib):
    if result['allowed']:
        return result
    if result.get('reasons') != ['UNRECOGNIZED_OR_OVERSIZED_GPU_PROCESS']:
        return result
    blocked=result.get('blocking_processes',[])
    if not blocked:
        return result
    for row in blocked:
        actual=row.get('identity')
        if (not actual or actual.get('state') in ('Z','X') or
            any(actual.get(k)!=expected[k] for k in ('pid','start_ticks','argv','exe')) or
            not math.isfinite(row['memory_mib']) or not 0 <= row['memory_mib'] <= maximum_mib):
            return result
    amended=copy.deepcopy(result)
    amended.update(allowed=True,reasons=[],blocking_processes=[],
                   pinned_display_admission=blocked,display_memory_limit_mib=maximum_mib)
    return amended


class ResourceResume(PriorityQueue):
    def __init__(self,root,revision):
        super().__init__(root)
        self.revision=revision
        self.policy=read(revision/'REQUEST.json')
        original=self.common.gate
        self.common.gate=lambda:admit_display(original(),self.policy['display_process'],self.policy['maximum_display_mib'])

    def check(self):
        super().check()
        check_freeze(self.revision/'FREEZE.json')

    def adopt(self):
        if getattr(self, '_adopted_worker', None) is not None:
            return self._adopted_worker
        self.check()
        if (self.revision/'ADOPTION.json').exists():
            raise RuntimeError('Repeated recovery requires review')
        old=self.policy['previous_controller']
        self.require_identity(old)
        self.common.suspended_ok()
        parked=read(self.q/'OLD_PARKED.json')
        if self.same(self.identity(parked['worker']['pid']),parked['worker']):
            raise RuntimeError('Old GPU worker unexpectedly alive')
        committed=False
        try:
            self.stop_exact(old)
            state=read(self.q/'STATE.json')
            if state['status']!='WAITING_GPU_RESOURCE' or not self.same(state['controller'],old):
                raise RuntimeError('Resource wait changed before recovery')
            if state.get('child'):
                raise RuntimeError('Unexpected child during resource-only recovery')
            self.retire_stopped(old)
            committed=True
            self.state('SAME_BAND_RESOURCE_REVISION_APPLIED',revision=str(self.revision),
                       display_process_untouched=True)
            self.common.suspended_ok()
            write_json(self.revision/'ADOPTION.json',dict(time=time.time(),retired_controller=old,
                display_process_untouched=True,old_gpu_worker_already_parked=True))
        finally:
            if not committed and self.same(self.identity(old['pid']),old):
                os.kill(old['pid'],signal.SIGCONT)
        return parked['worker']

    def run(self):
        # The waiting controller owns the base lock. Retire it under a separate
        # recovery lock before the inherited runner acquires that base lock.
        with (self.revision/'RECOVERY.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._adopted_worker=self.adopt()
            super().run()

    def park_old(self,worker):
        parked=read(self.q/'OLD_PARKED.json')
        if read(self.previous/'run/COMPLETED_EPOCHS.json')!=parked['completed']:
            raise RuntimeError('Old progress changed while parked')
        for meta in parked['checkpoints'].values():
            for key in ('original','backup'):
                if sha256(meta[key])!=meta['sha256']:
                    raise RuntimeError('Parked resume checkpoint changed')
        self.state('PARKED_CHECKPOINTS_REVERIFIED',completed=parked['completed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--revision',type=Path,required=True)
    args=parser.parse_args()
    queue=ResourceResume(args.experiment,args.revision)
    try:
        queue.run()
    except Exception:
        failure=dict(time=time.time(),status='FAILED',traceback=traceback.format_exc())
        write_json(args.revision/'FAILURE.json',failure)
        if (args.revision/'ADOPTION.json').exists():queue.state('FAILED_REQUIRES_REVIEW',failure=failure)
        raise
