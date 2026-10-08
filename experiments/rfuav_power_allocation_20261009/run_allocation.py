"""Matched full STFT U-Net fine-tuning with/without power allocation supervision.

Wait for the separately queued phase evaluation. Discard fixed-training-case
diagnostic weights, then restart both arms at the identical incumbent. No
automatic extra hyperparameter search, new data admission, or heldout access.
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
from allocation import ROOT,LEGACY,allocation_kl
from models import build,predict
from study import admitted_dataset,atomic_torch,evaluate
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256

HERE=Path(__file__).resolve().parent
PREPARATION=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')
PARENT=PREPARATION.parent/'dense_gpu_run/unet_mean/SELECTED_005.pt'
PARENT_SHA='86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356'
LOCK=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')
ARMS=('waveform_only','waveform_allocation')
WEIGHTS=dict(waveform_only=0.,waveform_allocation=.1)
EPOCHS=5
EFFECTIVE_BATCH=32
MICROBATCH=2


def freeze(root):
    config=json.loads((PREPARATION/'PREPARATION.json').read_text())
    if sha256(PARENT)!=PARENT_SHA:
        raise RuntimeError('Incumbent checkpoint changed')
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [*HERE.glob('*.py'),*LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]}
    data={str(p):sha256(p) for p in [PREPARATION/'PREPARATION.json',PREPARATION/'TRAIN.npy',
        PREPARATION/'VALIDATION.npy',Path(config['manifest']),PARENT]}
    protocol=dict(status='FROZEN_MATCHED_POWER_ALLOCATION',source_sha256=sources,data_sha256=data,
        arms=list(ARMS),weights=WEIGHTS,seed=0,parameters=32142859,epochs=EPOCHS,
        examples_per_epoch=2400,effective_batch=EFFECTIVE_BATCH,microbatch=MICROBATCH,
        updates_per_epoch=75,total_updates_per_arm=375,optimizer='fresh AdamW',lr=1e-5,
        weight_decay=1e-4,gradient_clip=1.,precision='float32 TF32 disabled',
        parent=str(PARENT),parent_sha256=PARENT_SHA,parent_selected_additional_epoch=1,
        source_prediction='unchanged full STFT U-Net; three unordered outputs and background',
        main_loss='existing whole-window PIT NMSE + coherence + background; 0.1 count CE',
        auxiliary='reference-energy-weighted KL of relative STFT source powers, existing waveform PIT assignment, background target zero',
        eps='relative 1e-8 smoothing of predicted power only; no phase/gain fitting',
        selection='minimum mean NMSE over counts2/3 including epoch0',
        acceptance='NMSE down AND complex SI-SDR up separately for counts2/3',
        diagnostic='same four TRAIN mixtures, each arm32 updates; discard weights, not generalized performance',
        train_indices=[4,5,2,11],cpu_gpu_parallel=True,validation_cases=630,
        heldout_access=False,physical_aircraft_count=False,native_center_offsets_preserved=False,
        data_scope='same frozen RFUAV records, same native band per center-aligned synthetic mixture')
    path=root/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:
            raise RuntimeError('Frozen allocation study changed')
    else:
        for name in sources:
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        write_json(path,protocol)
    return sha256(path)


def model():
    net=build('unet_mean')
    saved=torch.load(PARENT,map_location='cpu',weights_only=False)
    if saved['best']['epoch']!=1:
        raise RuntimeError('Unexpected incumbent selection')
    net.load_state_dict(saved['model'])
    if sum(p.numel() for p in net.parameters())!=32142859:
        raise RuntimeError('Full model capacity changed')
    return net.cuda()


def batch(items):
    keys=('mixture','references','active','context_features','crop_start','construction_count')
    return {k:torch.as_tensor(np.stack([item[k] for item in items]),device='cuda') for k in keys}


def objective(net,item,weight):
    estimates,logits=predict(net,item)
    wave=pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])
    primary=wave['loss']+.1*F.cross_entropy(logits,item['construction_count']-1)
    # The control computes the same diagnostic scalar, without its gradient.
    with torch.set_grad_enabled(torch.is_grad_enabled() and weight>0):
        auxiliary=allocation_kl(estimates,item['references'],wave['assignment'])
    return primary+weight*auxiliary,primary,auxiliary


def optimizer(net):
    return torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)


@torch.no_grad()
def fit_scores(net,items,weight):
    net.eval();nmse=[];si=[];primary=[];aux=[]
    for item in items:
        loss,p,a=objective(net,item,weight)
        estimates,_=predict(net,item)
        m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
        nmse+=m['nmse'][item['active']].tolist();si+=m['si_sdr'][item['active']].tolist()
        primary.append(float(p));aux.append(float(a))
    return dict(mean_nmse=float(np.mean(nmse)),mean_si_sdr=float(np.mean(si)),
        primary=float(np.mean(primary)),allocation=float(np.mean(aux)),component_nmse=nmse)


def diagnostic(root,frozen):
    if (root/'GPU_PREFLIGHT.json').exists():
        previous=json.loads((root/'GPU_PREFLIGHT.json').read_text())
        if previous['protocol_sha256']!=frozen or previous['status']!='PASS':
            raise RuntimeError('Existing diagnostic is incompatible')
        return
    train=admitted_dataset(PREPARATION,'train_pack',1)
    indices=[int(i) for k in (2,3) for i in np.flatnonzero(train.rows['count']==k)[:2]]
    if indices!=[4,5,2,11]:
        raise RuntimeError('Fixed diagnostic cases changed')
    source=[train[i] for i in indices]
    items=[batch(source[i:i+MICROBATCH]) for i in range(0,4,MICROBATCH)]
    results={}
    for arm in ARMS:
        net=model();opt=optimizer(net);weight=WEIGHTS[arm]
        before=fit_scores(net,items,weight);torch.cuda.reset_peak_memory_stats()
        # GroupNorm should not change predictions when microbatching.
        with torch.no_grad():
            whole,_=predict(net,items[0]);single,_=predict(net,batch(source[:1]))
            error=float((whole[:1]-single).abs().square().sum()/single.abs().square().sum())
            if error>1e-9:
                raise RuntimeError('Microbatch prediction differs')
        curve=[];start=time.time();net.train()
        for step in range(1,33):
            opt.zero_grad(set_to_none=True)
            for item in items:
                loss,_,_=objective(net,item,weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite diagnostic loss')
                (loss/len(items)).backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            if not float(norm)>0:
                raise RuntimeError('Missing model gradient')
            opt.step()
            if step in (1,16,32):
                curve.append(dict(step=step,**fit_scores(net,items,weight)));net.train()
        after=curve[-1]
        if not after['mean_nmse']<before['mean_nmse']:
            raise RuntimeError(f'{arm} fixed-case reconstruction failed to improve; inspect before main training')
        results[arm]=dict(before=before,curve=curve,microbatch_relative_error=error,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start)
        folder=root/arm;folder.mkdir(exist_ok=True)
        write_json(folder/'GPU_PREFLIGHT.json',dict(status='PASS',parameters=32142859,
            protocol_sha256=frozen,**results[arm]))
        write_json(root/'DIAGNOSTIC_PROGRESS.json',results)
        print(json.dumps(dict(stage='TRAIN_DIAGNOSTIC',arm=arm,**results[arm])),flush=True)
        del net,opt,loss;gc.collect();torch.cuda.empty_cache()
    write_json(root/'GPU_PREFLIGHT.json',dict(status='PASS',protocol_sha256=frozen,results=results,
        discarded_updates_per_arm=32,training_indices=indices,validation_or_heldout_access=False))


def validation(net,data,path,epoch):
    metrics=evaluate(net,data,path,epoch)
    # Explicit new selection rule; the original evaluator's all-count mean is
    # preserved separately instead of silently being relabeled.
    metrics['legacy_all_count_nmse']=metrics['selection_nmse']
    metrics['selection_nmse']=metrics['two_three_nmse']
    if len(metrics['rows'])!=630 or any(r['sum_relative_error']>1e-9
            or any(v is None for v in r['si_sdr']) for r in metrics['rows']):
        raise RuntimeError('Incomplete or invalid validation')
    if epoch==0:
        prior=json.loads((PARENT.parent/'VALIDATION_001.json').read_text())
        for new,old in zip(metrics['rows'],prior['rows']):
            if any(new[k]!=old[k] for k in ('index','count','categories','pack_ids','nominal_levels_db')):
                raise RuntimeError('Incumbent validation identity differs')
            for k in ('nmse','si_sdr','reference_power'):
                if not np.allclose(new[k],old[k],rtol=1e-5,atol=1e-7):
                    raise RuntimeError('Initial incumbent scores do not reproduce')
    write_json(path,metrics)
    return metrics


def train_epoch(arm,epoch,train,val,root,frozen):
    folder=root/arm;folder.mkdir(exist_ok=True)
    receipt=folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['protocol_sha256']!=frozen or prior['updates']!=75*epoch or not (folder/'LAST.pt').exists():
            raise RuntimeError('Unverified saved epoch')
        if epoch in (1,5) and not (folder/f'SELECTED_{epoch:03d}.pt').exists():
            raise RuntimeError('Incomplete checkpoint transaction')
        return
    net=model();opt=optimizer(net)
    last=folder/'LAST.pt';updates=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=frozen or saved['epoch']!=epoch-1:
            raise RuntimeError('Nonconsecutive or altered resume')
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        updates=saved['updates'];best=saved['best'];del saved
        chosen=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        if chosen['best']!=best or chosen['protocol_sha256']!=frozen:
            raise RuntimeError('Partial best/last checkpoint transaction')
        del chosen
    else:
        if epoch!=1:
            raise RuntimeError('Missing earlier epoch')
        score=validation(net,val,folder/'VALIDATION_000.json',0)
        best=dict(epoch=0,metric=score['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=frozen))
    net.train();opt.zero_grad(set_to_none=True);start=time.time();totals=np.zeros(3)
    torch.cuda.reset_peak_memory_stats()
    if len(train)!=2400:
        raise RuntimeError('Training schedule changed')
    for i in range(0,len(train),MICROBATCH):
        item=batch([train[j] for j in range(i,i+MICROBATCH)])
        values=objective(net,item,WEIGHTS[arm]);loss=values[0]
        if not all(torch.isfinite(v) for v in values):
            raise RuntimeError('Nonfinite training objective')
        (loss*(MICROBATCH/EFFECTIVE_BATCH)).backward()
        totals+=np.array([float(v.detach()) for v in values])*MICROBATCH
        if (i+MICROBATCH)%EFFECTIVE_BATCH==0:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            if updates%5==0:
                write_json(root/'PROGRESS.json',dict(stage='TRAIN',arm=arm,epoch=epoch,updates=updates,
                    examples=i+MICROBATCH,total=2400,seconds=time.time()-start,time=time.time(),pid=os.getpid()))
    torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()
    metrics=validation(net,val,folder/f'VALIDATION_{epoch:03d}.json',epoch)
    if metrics['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=metrics['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=frozen))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=frozen,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    write_json(folder/f'EPOCH_{epoch:03d}.json',dict(epoch=epoch,updates=updates,best=best,
        train_seconds=elapsed,peak_allocated_bytes=peak,average_loss=float(totals[0]/2400),
        primary_loss=float(totals[1]/2400),allocation_loss=float(totals[2]/2400),protocol_sha256=frozen,
        metrics={k:v for k,v in metrics.items() if k!='rows'}))
    if epoch in (1,5):
        shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    print(json.dumps(dict(stage='EPOCH_COMPLETE',arm=arm,epoch=epoch,best=best,
        by_count=metrics['by_count'])),flush=True)
    del net,opt,loss,values;gc.collect();torch.cuda.empty_cache()


def run(root,after):
    if (root/'COMPLETE.json').exists():
        raise RuntimeError('Study complete; do not silently rerun')
    frozen=freeze(root)
    write_json(root/'STATE.json',dict(status='WAITING_FOR_PHASE_EVALUATION',dependency=str(after),pid=os.getpid(),time=time.time()))
    while not (after/'COMPLETE.json').exists():
        if (after/'FAILURE.json').exists():
            raise RuntimeError('Dependent phase evaluation failed; investigate before next study')
        time.sleep(15)
    with LOCK.open('r') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                time.sleep(15)
        time.sleep(2)
        if not torch.cuda.is_available():
            raise RuntimeError('Native CUDA unavailable')
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
        for line in lines:
            pid,memory=line.split(',');pid=int(pid)
            p=Path('/proc',str(pid))
            if pid!=os.getpid() and ((p/'exe').resolve()!=Path('/usr/share/rustdesk/rustdesk')
                    or p.stat().st_uid!=os.getuid() or not 0<=float(memory)<=256):
                raise RuntimeError(f'Another CUDA task remains active: {pid}')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if freeze(root)!=frozen:
            raise RuntimeError('Queue source changed')
        write_json(root/'STATE.json',dict(status='GPU_DIAGNOSTIC_RUNNING',time=time.time(),pid=os.getpid()))
        diagnostic(root,frozen)
        write_json(root/'STATE.json',dict(status='GPU_TRAINING_RUNNING',time=time.time(),pid=os.getpid()))
        val=admitted_dataset(PREPARATION,'validation_pack',1)
        for epoch in range(1,EPOCHS+1):
            if freeze(root)!=frozen:
                raise RuntimeError('Frozen experiment changed')
            train=admitted_dataset(PREPARATION,'train_pack',epoch)
            for arm in ARMS if epoch%2 else reversed(ARMS):
                train_epoch(arm,epoch,train,val,root,frozen)
            write_json(root/'MATCHED_PROGRESS.json',dict(completed_epochs=epoch,updates_per_arm=epoch*75,time=time.time()))
        freeze(root)
        write_json(root/'COMPLETE.json',dict(status='MATCHED_COMPARISON_COMPLETE',epochs=EPOCHS,
            updates_per_arm=375,protocol_sha256=frozen,time=time.time()))
        write_json(root/'STATE.json',dict(status='COMPLETED',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True);args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]));os.nice(10)
    try:
        run(args.run,args.after)
    except Exception:
        write_json(args.run/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        write_json(args.run/'STATE.json',dict(status='FAILED',time=time.time()))
        raise
