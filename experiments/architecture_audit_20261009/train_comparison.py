"""Prospective native-RF architecture screen, equal data and optimizer updates.

Fresh full STFT U-Net vs full WaveNet cycle10/cycle15; each five epochs/375
updates. Round-robin epochs keep common budgets visible. This bounded screen
does not establish each architecture's converged performance or equal compute.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import fit_diagnostic as fit
from native_wavenet import build_native_wavenet
from cycle15 import build_cycle15
from models import build, predict
from native_data import NativeMixtures, sha256, write_json
from study import atomic_torch
from report import summarize

ROOT=fit.ROOT
ARMS={'wavenet_cycle10':4601867,'wavenet_cycle15':4601867,'unet_mean':32142859}


def model(arm):
    torch.manual_seed(0)
    value=(build_native_wavenet() if arm=='wavenet_cycle10' else
           build_cycle15() if arm=='wavenet_cycle15' else build('unet_mean'))
    if sum(p.numel() for p in value.parameters())!=ARMS[arm]:raise ValueError('Capacity changed')
    return value.cuda()


def register(root, preparation, first, second):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate registration')
    root.mkdir(parents=True,exist_ok=True)
    source=dict(fit.read(first/'PROTOCOL.json')['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('cycle15.py'),fit.NATIVE/'report.py'):
        source[str(path.relative_to(ROOT))]=sha256(path)
    protocol=dict(status='REGISTERED_FRESH_NATIVE_ARCHITECTURE_SCREEN',source_sha256=source,
        arms=ARMS,seed=0,epochs_per_arm=5,updates_per_arm=375,effective_batch=32,microbatch=1,
        optimizer='AdamW lr5e-4 wd1e-4 clip1, fresh, shared hyperparameters; no per-family tuning',
        loss='same PIT NMSE+coherence+inactive/background plus0.1 construction-count CE',
        preparation=str(preparation),preparation_sha256=sha256(preparation/'PREPARATION.json'),
        parent_checkpoint=None,initialization='fresh seed0; cycle10/15 start with identical parameter tensors',
        first_diagnostic=str(first),second_diagnostic=str(second),
        first_protocol_sha256=sha256(first/'PROTOCOL.json'),second_protocol_sha256=sha256(second/'PROTOCOL.json'),
        selection='per arm min mean NMSE counts2/3 including e0',validation_cases=630,
        ordering='round-robin epoch: cycle10,cycle15,full STFT U-Net',
        comparison='same native data and375updates; unequal capacities/compute; bounded development screen, not converged family ranking',
        acceptance='NMSE lower and complex SI-SDR higher in both counts2/3; weak/count separately',
        heldout_read=False,physical_aircraft_count=False)
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Frozen dependency changed')
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest)
    write_json(root/'PROTOCOL.json',protocol)
    return protocol


def verify(root, protocol):
    for rel,digest in protocol['source_sha256'].items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Running source changed')
    if sha256(Path(protocol['preparation'])/'PREPARATION.json')!=protocol['preparation_sha256']:
        raise ValueError('Prepared data changed')


def train_epoch(root,arm,epoch,protocol,train,val):
    folder=root/arm;folder.mkdir(exist_ok=True)
    digest=sha256(root/'PROTOCOL.json')
    net=model(arm)
    opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4,foreach=False)
    last=folder/'LAST.pt';updates=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=digest or saved['epoch']!=epoch-1:raise ValueError('Nonconsecutive state')
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        updates=saved['updates'];best=saved['best']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        if epoch!=1:raise ValueError('Missing earlier epoch')
        val0=fit.base.validate(net,val,folder/'VALIDATION_000.json',0)
        best=dict(epoch=0,metric=val0['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=digest,arm=arm))
    if len(train)!=2400:raise ValueError('Changed update budget')
    net.train();opt.zero_grad(set_to_none=True)
    begin=time.time();total=0.;torch.cuda.reset_peak_memory_stats()
    for index in range(len(train)):
        item=fit.base.batch([train[index]])
        loss=fit.base.objective(net,item)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        (loss/32).backward();total+=float(loss.detach())
        if (index+1)%32==0:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            if updates%5==0:
                write_json(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=epoch,updates=updates,
                    examples=index+1,total=2400,seconds=time.time()-begin,pid=os.getpid(),time=time.time()))
    seconds=time.time()-begin
    value=fit.base.validate(net,val,folder/f'VALIDATION_{epoch:03d}.json',epoch)
    if value['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=digest,arm=arm))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=digest,arm=arm,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    result=dict(arm=arm,epoch=epoch,updates=updates,best=best,train_seconds=seconds,
        peak_bytes=torch.cuda.max_memory_allocated(),mean_training_loss=total/2400,
        validation={k:v for k,v in value.items() if k!='rows'},protocol_sha256=digest)
    write_json(folder/f'EPOCH_{epoch:03d}.json',result)
    print(json.dumps(result),flush=True)
    del net,opt,item,loss
    gc.collect();torch.cuda.empty_cache()


def run(root,protocol):
    begin=time.time()
    for name in ('first_diagnostic','second_diagnostic'):
        dependency=Path(protocol[name])
        while not (dependency/'COMPLETE.json').exists():
            if (dependency/'FAILURE.json').exists():raise RuntimeError('Full-input preflight failed')
            if time.time()-begin>7200:raise TimeoutError('Full-input preflight wait exceeded')
            write_json(root/'STATE.json',dict(status='WAITING_FULL_INPUT_FIT_DIAGNOSTICS',dependency=str(dependency),pid=os.getpid(),time=time.time()))
            time.sleep(10)
        result=fit.read(dependency/'COMPLETE.json')['results']
        if not all(r['improved'] and r['finite_gradients'] for r in result.values()):
            write_json(root/'STATE.json',dict(status='NOT_STARTED_FIT_DIAGNOSTIC_REQUIRES_REVIEW',dependency=str(dependency),pid=os.getpid(),time=time.time()))
            return
    for name,key in (('first_diagnostic','first_protocol_sha256'),('second_diagnostic','second_protocol_sha256')):
        if sha256(Path(protocol[name])/'PROTOCOL.json')!=protocol[key]:raise ValueError('Preflight protocol changed')
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify(root,protocol)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        if any(int(p)!=os.getpid() for p in pids):raise RuntimeError('Other GPU worker active')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        val=NativeMixtures(protocol['preparation'],'validation_pack',1)
        for epoch in range(1,6):
            train=NativeMixtures(protocol['preparation'],'train_pack',epoch)
            for arm in ARMS:
                verify(root,protocol)
                train_epoch(root,arm,epoch,protocol,train,val)
            selected={}
            for arm in ARMS:
                history=[dict(epoch=e,**summarize(root/arm/f'VALIDATION_{e:03d}.json')) for e in range(epoch+1)]
                selected[arm]=min(history,key=lambda x:x['selection_nmse'])
            write_json(root/f'COMMON_EPOCH_{epoch:03d}.json',dict(epochs_per_arm=epoch,updates_per_arm=75*epoch,selected=selected))
        verify(root,protocol)
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',epochs_per_arm=5,updates_per_arm=375,
            protocol_sha256=sha256(root/'PROTOCOL.json'),time=time.time(),heldout_read=False))
        write_json(root/'STATE.json',dict(status='COMPLETED',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','preparation','first','second'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    try:
        plan=register(root,args.preparation.resolve(),args.first.resolve(),args.second.resolve())
        run(root,plan)
    except Exception:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
        raise
