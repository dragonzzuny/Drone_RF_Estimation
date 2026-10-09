"""Queued full-input GPU sanity/fit check for phase-preserving long context.

Wait for the running architecture screen to finish; retain the shared GPU lock.
Two full 4.64M models, identical initial tensors, TRAIN4 only, 32 updates each.
These weights are discarded. This is not a validation performance comparison.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

import phase_packing as pp
import fit_diagnostic as fit
from native_data import NativeMixtures, sha256, write_json
from drone_rf.context_data import component_gains
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics

ROOT=Path(__file__).resolve().parents[2]


def register(root, predecessor, cpu_check):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate registration')
    root.mkdir(parents=True,exist_ok=True)
    previous=fit.read(predecessor/'PROTOCOL.json')
    check=fit.read(cpu_check)
    if check['status']!='PASS' or check['parameters']!=pp.PARAMETERS:
        raise ValueError('CPU preflight not passed')
    source=dict(previous['source_sha256'])
    for path in (Path(__file__),Path(pp.__file__),Path(__file__).with_name('check_phase_packing.py')):
        source[str(path.relative_to(ROOT))]=sha256(path)
    for name,digest in check['source_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('CPU-checked candidate changed')
    plan=dict(status='REGISTERED_FULL_INPUT_PHASE_CONTEXT_FIT',
        predecessor=str(predecessor),predecessor_protocol_sha256=sha256(predecessor/'PROTOCOL.json'),
        preparation=previous['preparation'],preparation_sha256=previous['preparation_sha256'],
        cpu_check=str(cpu_check),cpu_check_sha256=sha256(cpu_check),source_sha256=source,
        arms=['local','long'],parameters_per_arm=pp.PARAMETERS,seed=0,
        train_indices=[4,5,2,11],updates_per_arm=32,effective_batch=4,microbatch=1,
        long_samples=pp.SAMPLES,fine_samples=pp.FINE,packing=pp.PACK,sample_rate_hz=100_000_000,
        change='only visibility of out-of-crop complex IQ; same full-mixture RMS and mean power features',
        optimizer='fresh AdamW lr5e-4 wd1e-4 clip1, float32 TF32 disabled',
        loss='same PIT NMSE+coherence+inactive/background plus0.1 construction-count CE',
        purpose='Full-input GPU forward/backward and optimization preflight on fixed TRAIN4; discard weights',
        full_comparison_registered=False,validation_iq_read=False,heldout_read=False)
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest:
            raise ValueError('Registered source changed')
        dest=root/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest)
    write_json(root/'PROTOCOL.json',plan)
    return plan


def verify(root, plan):
    for rel,digest in plan['source_sha256'].items():
        if sha256(ROOT/rel)!=digest or sha256(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen preflight source changed')
    for path,digest in ((Path(plan['preparation'])/'PREPARATION.json',plan['preparation_sha256']),
                        (Path(plan['cpu_check']),plan['cpu_check_sha256']),
                        (Path(plan['predecessor'])/'PROTOCOL.json',plan['predecessor_protocol_sha256'])):
        if sha256(path)!=digest:
            raise ValueError('Preflight data/protocol changed')


def examples(plan):
    data=NativeMixtures(plan['preparation'],'train_pack',1)
    result=[]
    for index in plan['train_indices']:
        item=data[index]
        row=data.rows[index]
        count=int(row['count'])
        indices=row['indices'][:count]
        gains=component_gains([data.library.clips[int(i)]['mean_power'] for i in indices],
                              row['levels'][:count],row['phases'][:count])
        mixture=np.zeros(pp.SAMPLES,np.complex64)
        for i,gain in zip(indices,gains):
            mixture+=(data.library._array(int(i))*gain).astype(np.complex64)
        start=int(item['crop_start'])
        np.testing.assert_array_equal(mixture[start:start+pp.FINE],item['mixture'])
        values={k:torch.as_tensor(np.asarray(item[k])[None],device='cuda') for k in
                ('mixture','references','active','context_features','crop_start','construction_count')}
        values['long_mixture']=torch.from_numpy(mixture[None]).cuda()
        result.append(values)
    if [int(x['construction_count']) for x in result]!=[2,2,3,3]:
        raise ValueError('TRAIN4 composition changed')
    return result


def predict(net,item):
    # References, activity labels and the true count never enter the model.
    return net(item['long_mixture'],item['context_features'],item['crop_start'])


def score(net,items):
    net.eval()
    rows=[]
    with torch.no_grad():
        for item in items:
            predicted=predict(net,item)
            metrics=waveform_metrics(predicted['estimates'],item['references'],item['active'],item['mixture'])
            values=metrics['nmse'][item['active']]
            if not torch.isfinite(values).all() or torch.any(metrics['sum_relative_error']>1e-9):
                raise ValueError('Invalid preflight waveform output')
            rows.append(dict(count=int(item['construction_count']),nmse=values.cpu().tolist(),
                             mean_nmse=float(values.mean())))
    return dict(mean_nmse=float(np.mean([r['mean_nmse'] for r in rows])),rows=rows)


def run(root,plan):
    predecessor=Path(plan['predecessor'])
    while not (predecessor/'COMPLETE.json').exists():
        if (predecessor/'FAILURE.json').exists():
            raise RuntimeError('Architecture screen failed; do not bypass it')
        write_json(root/'STATE.json',dict(status='WAITING_ARCHITECTURE_SCREEN',pid=os.getpid(),time=time.time()))
        time.sleep(30)
    write_json(root/'STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify(root,plan)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        predecessor_pid=fit.read(predecessor/'STATE.json')['pid']
        deadline=time.time()+60
        while True:
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
            others={int(p) for p in pids}-{os.getpid()}
            if not others:
                break
            if others-{predecessor_pid} or time.time()>deadline:
                raise RuntimeError('Another GPU worker is active')
            time.sleep(1)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        items=examples(plan)
        results={}
        for arm in plan['arms']:
            torch.manual_seed(0)
            net=pp.PhasePackedWaveNet(arm).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4,foreach=False)
            initial=score(net,items)
            history=[]
            torch.cuda.reset_peak_memory_stats()
            began=time.time()
            for step in range(1,33):
                net.train();opt.zero_grad(set_to_none=True)
                for item in items:
                    predicted=predict(net,item)
                    loss=pit_waveform_loss(predicted['estimates'],item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*F.cross_entropy(predicted['count_logits'],item['construction_count']-1)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite preflight loss')
                    (loss/4).backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                if step==1:
                    for parameter in (net.net.input_projection.weight,net.net.blocks[-1].filter_gate.weight):
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum()==0:
                            raise ValueError('Disconnected/nonfinite backbone gradient')
                opt.step()
                if step in (1,8,16,32):
                    value=dict(step=step,gradient_norm=float(norm),**score(net,items))
                    history.append(value)
                    write_json(root/f'{arm}_STEP_{step:03d}.json',value)
                    write_json(root/'STATE.json',dict(status='TRAIN_FIT_DIAGNOSTIC',arm=arm,step=step,pid=os.getpid(),time=time.time()))
                    print(json.dumps(dict(arm=arm,**value)),flush=True)
            final=history[-1]['mean_nmse']
            results[arm]=dict(initial=initial,history=history,final_nmse=final,
                improved=final<initial['mean_nmse'],finite_gradients=True,
                seconds=time.time()-began,peak_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in net.parameters()),weights_discarded=True)
            del net,opt,predicted,loss
            gc.collect();torch.cuda.empty_cache()
        verify(root,plan)
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',results=results,
            protocol_sha256=sha256(root/'PROTOCOL.json'),validation_iq_read=False,heldout_read=False,time=time.time()))
        write_json(root/'STATE.json',dict(status='COMPLETED',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','predecessor','cpu-check'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve()
    try:
        plan=register(root,args.predecessor.resolve(),args.cpu_check.resolve())
        run(root,plan)
    except Exception:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
        raise
