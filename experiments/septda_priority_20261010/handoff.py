"""Wait for the current candidate epoch receipt, then change worker at that boundary."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]


def write(path, value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(start_ticks=fields[19],cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' '))
    except FileNotFoundError:
        return None


def run(root):
    p=json.loads((root/'EXECUTION_PRIORITY.json').read_text())
    for path, sha in p['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    pid=p['predecessor_pid']; boundary=p['boundary_epoch']
    expected=p['predecessor_identity']
    assert 'septda_continuation_20261010/train.py' in expected['cmdline']
    receipt=root/'septda'/f'EPOCH_{boundary:03d}.json'
    deadline=time.monotonic()+7200
    write(root/'PRIORITY_HANDOFF_STATE.json',dict(status='WAITING_SAVED_EPOCH',epoch=boundary,pid=os.getpid(),time=time.time()))
    while not receipt.exists():
        assert process_identity(pid)==expected,'Original worker stopped/changed before the required receipt'
        assert not (root/'FAILURE.json').exists(),'Original worker failed'
        if time.monotonic()>deadline:raise TimeoutError('Epoch boundary wait exceeded two hours')
        time.sleep(.5)
    event=json.loads(receipt.read_text())
    assert event['epoch']==boundary and event['updates']==75*boundary and event['exact_last_roundtrip_checked']
    assert event['protocol_sha256']==p['original_protocol_sha256']
    before=json.loads((root/'STATE.json').read_text())
    if process_identity(pid)==expected:
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if process_identity(pid)!=expected:break
            time.sleep(.1)
        else:raise RuntimeError('Original worker did not exit after SIGTERM; no new worker launched')
    elif process_identity(pid) is not None:
        raise RuntimeError('PID was reused; no signal or new launch')
    record=dict(status='OLD_WORKER_STOPPED_AFTER_SAVED_EPOCH',epoch=boundary,predecessor_pid=pid,
                predecessor_identity=expected,state_before_stop=before,
                receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),time=time.time())
    write(root/'PRIORITY_HANDOFF.json',record)
    log=(root/'priority_worker.log').open('ab',buffering=0)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    child=subprocess.Popen(['taskset','-c','14,15','nice','-n','10',sys.executable,
        str(ROOT/'experiments/septda_priority_20261010/train.py'),'--run',str(root)],
        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'PRIORITY_PID').write_text(str(child.pid)+'\n')
    write(root/'PRIORITY_HANDOFF_STATE.json',dict(status='PRIORITY_WORKER_LAUNCHED',
        pid=child.pid,completed_epoch=boundary,time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve()
    try:
        with (root/'.priority_handoff.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(root)
    except Exception:
        write(root/'PRIORITY_HANDOFF_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
