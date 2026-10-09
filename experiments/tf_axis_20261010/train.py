"""Full-capacity dual-axis U-Net, ordinary original-objective updates."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from adapter import augment

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/count_pcgrad_20261010'))
spec=importlib.util.spec_from_file_location('axis_training_base',ROOT/'experiments/count_pcgrad_20261010/train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
w=base.w;ORIGINAL_REGISTER=base.register;ORIGINAL_VERIFY=base.verify
ORIGINAL_MODEL=base.worker.make_model
CHECK=ROOT/'reports/2026-10-10/TF_AXIS_CPU_CHECK.json'
PLAN=ROOT/'reports/2026-10-10/TF_AXIS_PLAN_KO.md'


def make_model(arm,parent=None):
    net=ORIGINAL_MODEL(arm,parent);torch.manual_seed(0);return augment(net)


def ordinary(groups,rng):
    flat=groups.sum(0)
    gram=torch.stack([torch.stack([torch.dot(a,b) for b in groups]) for a in groups])
    return flat,dict(gram=gram.tolist(),ordinary_norm=float(flat.norm()),projected_norm=float(flat.norm()),change_norm=0.,
        projected_direction_dot_original_tasks=[float(torch.dot(flat,g)) for g in groups])


def register(root,study,predecessor):
    p=ORIGINAL_REGISTER(root,study,predecessor);check=w.read(CHECK)
    p.update(status='REGISTERED_DUAL_AXIS_UNET',parameters=check['full_parameters'],added_parameters=check['added_parameters'],
        architecture='Complete retained U-Net plus two full-frequency then subband-time BLSTM residual blocks at bottleneck',
        inference='Full retained U-Net augmented with dual-axis recurrence; mixture-only inputs unchanged, no count/category/reference input',
        gradient_update='Ordinary sum of sample-weighted count-group gradients; no surgery',
        axis_plan_sha256=w.digest(PLAN),
        limitation='One seed, reused DEV; extra 5.26M parameters and recurrent compute, not equal cost or TF-GridNet reproduction')
    for source in (Path(__file__),HERE/'audit.py',ROOT/'experiments/count_pcgrad_20261010/gradient.py'):
        rel=str(source.relative_to(ROOT));p['source_sha256'][rel]=w.digest(source)
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    ORIGINAL_VERIFY(root,p)
    assert p['status']=='REGISTERED_DUAL_AXIS_UNET' and p['parameters']==37406475
    assert w.digest(PLAN)==p['axis_plan_sha256']


base.CHECK=CHECK;base.verify=verify;base.surgery=ordinary;base.worker.make_model=make_model


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','predecessor','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            predecessor=a.predecessor.resolve()
            w.write(root/'STATE.json',dict(status='WAITING_FOR_WAVE_GUARD',pid=os.getpid(),time=time.time()))
            skip=False
            while not (predecessor/'COMPLETE.json').exists():
                if (predecessor/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                state=predecessor/'STATE.json'
                if state.exists() and w.read(state)['status'].startswith('SKIPPED_'):
                    skip=True;break
                time.sleep(5)
            if skip or w.read(predecessor/'COMPLETE.json')['criterion_met']:
                w.write(root/'STATE.json',dict(status='SKIPPED_PRIOR_METHOD_MET_CRITERION',pid=os.getpid(),time=time.time()))
                print('Prior method met criterion; conditional structure experiment skipped',flush=True)
            else:
                p=register(root,a.study.resolve(),predecessor);base.run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
