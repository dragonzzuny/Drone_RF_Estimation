"""Explicit CAGrad callback into the frozen, identical-budget PCGrad worker."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import shutil
import time
import traceback
import cagrad

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('count_training_base',HERE/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ROOT=base.ROOT;w=base.w
ORIGINAL_REGISTER=base.register;ORIGINAL_VERIFY=base.verify
CHECK=ROOT/'reports/2026-10-10/COUNT_CAGRAD_CPU_CHECK.json'
PLAN=ROOT/'reports/2026-10-10/COUNT_CAGRAD_PLAN_KO.md'


def register(root,study,predecessor):
    p=ORIGINAL_REGISTER(root,study,predecessor)
    p.update(status='REGISTERED_COUNT_CAGRAD',c=.5,
        gradient_update='CAGrad Eq3: g0=sum weighted group gradients, c=0.5; unrescaled paper direction, then clip/AdamW',
        cagrad_plan_sha256=w.digest(PLAN),
        pcgrad_comparison_protocol_sha256=w.digest(predecessor/'PROTOCOL.json'))
    for source in (Path(__file__),HERE/'audit_cagrad.py'):
        rel=str(source.relative_to(ROOT));p['source_sha256'][rel]=w.digest(source)
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    # Registration completes before any data/model operation. The base retains
    # its original plan hash as a shared design dependency; this plan is extra.
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    ORIGINAL_VERIFY(root,p)
    assert p['status']=='REGISTERED_COUNT_CAGRAD' and p['c']==.5
    assert w.digest(PLAN)==p['cagrad_plan_sha256']
    assert w.digest(Path(p['predecessor'])/'PROTOCOL.json')==p['pcgrad_comparison_protocol_sha256']


base.CHECK=CHECK
base.verify=verify
base.surgery=cagrad.surgery


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','predecessor','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            predecessor=a.predecessor.resolve()
            w.write(root/'STATE.json',dict(status='WAITING_FOR_PCGRAD',pid=__import__('os').getpid(),time=time.time()))
            while not (predecessor/'COMPLETE.json').exists():
                if (predecessor/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                time.sleep(5)
            p=register(root,a.study.resolve(),predecessor);base.run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
