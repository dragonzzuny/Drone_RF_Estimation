"""Full-capacity native-RF observation baseline; prospective five-epoch budget.

This changes the observation geometry. It is not a new-architecture comparison
against historical center-aligned scores. Heldout recordings remain unopened.
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
from native_data import ROOT,LEGACY,HERE,NativeMixtures,write_json,sha256
from models import build,predict
from study import atomic_torch,evaluate
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics

PARENT=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_gpu_run/unet_mean/SELECTED_005.pt')
PARENT_SHA='86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356'
LOCK=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')
EPOCHS=5


def freeze(root):
    if sha256(PARENT)!=PARENT_SHA:raise RuntimeError('Parent changed')
    prep=json.loads((root/'PREP_PROTOCOL.json').read_text())
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [HERE/'run_native.py',HERE/'native.py',
        HERE/'native_data.py',*LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]}
    protocol=dict(status='REGISTERED_NATIVE_RF_BASELINE',source_sha256=sources,
        prep_protocol_sha256=sha256(root/'PREP_PROTOCOL.json'),parent=str(PARENT),parent_sha256=PARENT_SHA,
        seed=0,parameters=32142859,epochs=5,examples_per_epoch=2400,effective_batch=32,microbatch=2,
        updates_per_epoch=75,total_updates=375,optimizer='fresh AdamW lr1e-5 wd1e-4 clip1',
        precision='float32 TF32 disabled',main_loss='whole-window PIT NMSE + coherence + inactive/background; 0.1 count CE',
        auxiliary_loss=False,diagnostic='TRAIN indices4,5,2,11; 32updates; discard and reload parent',
        selection='min mean NMSE counts2/3 including e0',validation_cases=630,
        inference_inputs='mixture IQ, its own long power context, crop coordinate only',
        comparison='same native observations: frozen parent versus adapted model; different update budget explicitly stated',
        acceptance='NMSE lower and complex SI-SDR higher for both counts2/3; inspect weakest and count too',
        heldout_access=False,physical_aircraft_count=False,geometry=prep['bands'])
    path=root/'GPU_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise RuntimeError('GPU protocol changed')
    else:
        for name in sources:
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha256(target)!=sources[name]:raise RuntimeError('Snapshot conflict')
            if not target.exists():shutil.copyfile(ROOT/name,target)
        write_json(path,protocol)
    return sha256(path)


def model():
    net=build('unet_mean')
    saved=torch.load(PARENT,map_location='cpu',weights_only=False)
    if saved['best']['epoch']!=1:raise RuntimeError('Wrong parent epoch')
    net.load_state_dict(saved['model'])
    if sum(p.numel() for p in net.parameters())!=32142859:raise RuntimeError('Capacity changed')
    return net.cuda()


def batch(items):
    keys=('mixture','references','active','context_features','crop_start','construction_count')
    return {k:torch.as_tensor(np.stack([x[k] for x in items]),device='cuda') for k in keys}


def objective(net,item):
    estimates,logits=predict(net,item)
    return pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']+.1*F.cross_entropy(logits,item['construction_count']-1)


def optimizer(net):
    return torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)


@torch.no_grad()
def fit_score(net,items):
    net.eval();nmse=[]
    for item in items:
        output,_=predict(net,item)
        score=waveform_metrics(output,item['references'],item['active'],item['mixture'])
        if float(score['sum_relative_error'].max())>1e-9:raise RuntimeError('Model mixture sum failed')
        nmse+=score['nmse'][item['active']].tolist()
    return float(np.mean(nmse))


def diagnostic(root,protocol,train):
    path=root/'GPU_PREFLIGHT.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['protocol_sha256']!=protocol or result['status']!='PASS':raise RuntimeError('Invalid old diagnostic')
        return
    source=[train[i] for i in (4,5,2,11)];items=[batch(source[:2]),batch(source[2:])]
    net=model();opt=optimizer(net);before=fit_score(net,items)
    with torch.no_grad():
        grouped,_=predict(net,items[0]);single,_=predict(net,batch(source[:1]))
        error=float((grouped[:1]-single).abs().square().sum()/single.abs().square().sum())
        if error>1e-9:raise RuntimeError('GroupNorm microbatch mismatch')
    torch.cuda.reset_peak_memory_stats();curve=[];start=time.time()
    for step in range(1,33):
        net.train();opt.zero_grad(set_to_none=True)
        for item in items:
            loss=objective(net,item)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite diagnostic loss')
            (loss/2).backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        if not float(grad)>0:raise RuntimeError('No diagnostic gradient')
        opt.step()
        if step in (1,16,32):
            score=fit_score(net,items);curve.append(dict(step=step,nmse=score))
            print(json.dumps(dict(stage='GPU_DIAGNOSTIC',step=step,nmse=score,before=before)),flush=True)
    if not curve[-1]['nmse']<before:raise RuntimeError('Cannot fit native TRAIN cases; inspect before full training')
    write_json(path,dict(status='PASS',protocol_sha256=protocol,before_nmse=before,curve=curve,
        microbatch_relative_error=error,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.time()-start,discarded_updates=32,training_indices=[4,5,2,11],heldout_read=False))
    del net,opt,items,loss;gc.collect();torch.cuda.empty_cache()


def validate(net,data,path,epoch):
    value=evaluate(net,data,path,epoch)
    value['legacy_all_count_nmse']=value['selection_nmse'];value['selection_nmse']=value['two_three_nmse']
    value['native_center_offsets_preserved']=True
    value['reconstruction_target']='common-RF-band-limited recorded contributions, not complete raw records'
    if len(value['rows'])!=630 or any(r['sum_relative_error']>1e-9 for r in value['rows']):
        raise RuntimeError('Incomplete or inconsistent validation')
    write_json(path,value);return value


def wait_ready(root,role,epoch):
    path=root/'preparation/features'/f'{role}_{epoch:03d}.json'
    while not path.exists():
        if (root/'PREP_FAILURE.json').exists():raise RuntimeError('CPU preparation failed')
        write_json(root/'GPU_STATE.json',dict(status='WAITING_FOR_CPU_DATA',role=role,epoch=epoch,time=time.time(),pid=os.getpid()))
        time.sleep(10)
    return NativeMixtures(root/'preparation',role,epoch)


def train_epoch(root,epoch,protocol,train,val):
    folder=root/'gpu';folder.mkdir(exist_ok=True)
    receipt=folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['protocol_sha256']!=protocol or prior['updates']!=epoch*75:raise RuntimeError('Resume mismatch')
        return
    net=model();opt=optimizer(net);last=folder/'LAST.pt';updates=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=protocol or saved['epoch']!=epoch-1:raise RuntimeError('Nonconsecutive resume')
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);best=saved['best'];updates=saved['updates']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
    else:
        if epoch!=1:raise RuntimeError('Missing earlier epoch')
        value=validate(net,val,folder/'VALIDATION_000.json',0)
        best=dict(epoch=0,metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=protocol))
        print(json.dumps(dict(stage='INITIAL_VALIDATION',by_count=value['by_count'])),flush=True)
    if len(train)!=2400:raise RuntimeError('Native schedule budget changed')
    net.train();opt.zero_grad(set_to_none=True);started=time.time();total=0.
    torch.cuda.reset_peak_memory_stats()
    for i in range(0,len(train),2):
        item=batch([train[i],train[i+1]]);loss=objective(net,item)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        (loss/16).backward();total+=float(loss.detach())*2
        if (i+2)%32==0:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            if updates%5==0:
                write_json(root/'GPU_PROGRESS.json',dict(stage='TRAIN',epoch=epoch,updates=updates,
                    examples=i+2,total=2400,seconds=time.time()-started,time=time.time(),pid=os.getpid()))
    torch.cuda.synchronize();elapsed=time.time()-started;peak=torch.cuda.max_memory_allocated()
    value=validate(net,val,folder/f'VALIDATION_{epoch:03d}.json',epoch)
    if value['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=protocol))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=protocol,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    if epoch in (1,5):shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    write_json(receipt,dict(epoch=epoch,updates=updates,best=best,protocol_sha256=protocol,
        train_seconds=elapsed,peak_allocated_bytes=peak,average_loss=total/2400,
        metrics={k:v for k,v in value.items() if k!='rows'}))
    print(json.dumps(dict(stage='EPOCH_COMPLETE',epoch=epoch,best=best,by_count=value['by_count'])),flush=True)
    del net,opt,loss;gc.collect();torch.cuda.empty_cache()


def run(root):
    if (root/'GPU_COMPLETE.json').exists():raise RuntimeError('Already complete')
    protocol=freeze(root)
    val=wait_ready(root,'validation_pack',1);train=wait_ready(root,'train_pack',1)
    with LOCK.open('r') as lock:
        write_json(root/'GPU_STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not torch.cuda.is_available():raise RuntimeError('Native CUDA unavailable')
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        if any(int(p)!=os.getpid() for p in pids):raise RuntimeError('Another compute worker is active')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if freeze(root)!=protocol:raise RuntimeError('Frozen code changed')
        write_json(root/'GPU_STATE.json',dict(status='DIAGNOSTIC_RUNNING',pid=os.getpid(),time=time.time()))
        diagnostic(root,protocol,train)
        for epoch in range(1,EPOCHS+1):
            train=wait_ready(root,'train_pack',epoch)
            if freeze(root)!=protocol:raise RuntimeError('Source changed during training')
            write_json(root/'GPU_STATE.json',dict(status='TRAINING',epoch=epoch,pid=os.getpid(),time=time.time()))
            train_epoch(root,epoch,protocol,train,val)
        write_json(root/'GPU_COMPLETE.json',dict(status='COMPLETE',epochs=5,updates=375,
            protocol_sha256=protocol,time=time.time(),heldout_read=False))
        write_json(root/'GPU_STATE.json',dict(status='COMPLETED',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();os.sched_setaffinity(0,{14,15});os.nice(10)
    try:run(args.run.resolve())
    except Exception:
        write_json(args.run/'GPU_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(args.run/'GPU_STATE.json',dict(status='FAILED',time=time.time()))
        raise
