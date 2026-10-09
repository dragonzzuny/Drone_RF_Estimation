"""One-factor follow-up: no magnitude auxiliary on one-source TRAIN examples.

Uses the frozen trainer through an explicit, process-local objective callback.
All training examples and waveform/count objectives are retained. True count
is used only to define the training auxiliary, never as inference input.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
import train as base
from objective import spectral_terms

w=base.w
CHECK=base.ROOT/'reports/2026-10-10/SELECTIVE_MAGNITUDE_CPU_CHECK.json'


def objective(estimates,logits,item,weight=.1):
    v=base.worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])
    main=v['loss']+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
    terms=spectral_terms(estimates,item['references'],item['active'],item['mixture'],v['assignment'])
    per_example=(terms['relative_l1']*item['active']).sum(-1)/item['active'].sum(-1).clamp_min(1)
    # Keep the same averaging denominator, including one-source examples.
    magnitude=(per_example*(item['construction_count']>1)).mean()
    return dict(loss=main+weight*magnitude,main=main,magnitude=magnitude,assignment=v['assignment'])


def register(root,previous):
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    p=w.read(previous/'PROTOCOL.json');done=w.read(previous/'COMPLETE.json')
    assert done['status']=='COMPLETE' and done['protocol_sha256']==w.digest(previous/'PROTOCOL.json')
    audit=w.read(base.ROOT/'reports/2026-10-10/MAGNITUDE_FINAL.json')['audit']
    assert audit['status']=='PASS' and audit['study_protocol_sha256']==w.digest(previous/'PROTOCOL.json')
    check=w.read(CHECK);assert check['status']=='PASS'
    sources=dict(p['source_sha256']);sources.update(check['source_sha256'])
    for rel,sha in sources.items():
        assert w.digest(base.ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base.ROOT/rel,dest)
    p.update(status='REGISTERED_MULTISOURCE_ONLY_MAGNITUDE',source_sha256=sources,
        predecessor=str(previous),predecessor_complete_sha256=w.digest(previous/'COMPLETE.json'),
        selective_cpu_check_sha256=w.digest(CHECK),
        initialization='Reload original retained native parent e2; do not resume magnitude e1 weights or optimizer',
        loss='Original waveform/count objectives on ALL examples; add 0.1 relative magnitude L1 only when TRAIN construction_count>1',
        change='Remove magnitude auxiliary on the 800 one-source training examples; keep all 2400 examples and original losses',
        objective_callback='selective.objective installed in frozen train.run inside this process; no model/data/inference edits',
        contrast='Same native parent, 75 updates and 0.1 weight; primary diagnostic comparator is completed all-count magnitude e1, plus original e1 and parent',
        hypothesis='Excess relative auxiliary pressure on well-fit one-source examples may interfere with multi-source adaptation; not a proven cause',
        extra_acceptance='Do not adopt e0 or a candidate that fails the original parent and equal-budget waveform control; also report all-count magnitude comparison',
        adaptive_followup=True,independent_test=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',p);return p


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('previous','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root,a.previous.resolve())
            assert w.digest(CHECK)==p['selective_cpu_check_sha256']
            base.objective=objective
            base.run(root,p,a.public.resolve())
            assert w.digest(CHECK)==p['selective_cpu_check_sha256']
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
