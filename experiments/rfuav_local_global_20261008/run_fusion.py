"""Matched local/global complex-context fusion, common parent and fresh optimizers."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from fusion_models import ARMS, LEGACY, PARENT_DIR, PARENT, PARENT_SHA256, build, predict, stack_batch
from fusion_evaluation import evaluate
from fusion_checks import preflight
sys.path.insert(0,str(LEGACY))
from study import atomic_torch
from long_data import admitted_dataset, batch
from drone_rf.data import sha256
from drone_rf.losses import pit_waveform_loss
from drone_rf.context_training_data import write_json

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PREPARATION=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')
LOCK=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')
EPOCHS=5
BATCH=8
MICROBATCH=4
UPDATES_PER_EPOCH=300
TOTAL_UPDATES=EPOCHS*UPDATES_PER_EPOCH


def lr_for_update(index):
    if not 0<=index<TOTAL_UPDATES:
        raise ValueError('Optimizer update outside fixed budget')
    return 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*index/(TOTAL_UPDATES-1)))


def objective(net,item):
    estimate,logits=predict(net,item)
    return (pit_waveform_loss(estimate,item['references'],item['active'],item['mixture'])['loss']
            +.1*F.cross_entropy(logits,item['construction_count']-1))


def optimizer_for(net):
    names=('iq_context.','fine_film.','coarse_film.')
    local=[p for n,p in net.named_parameters() if not n.startswith(names)]
    added=[p for n,p in net.named_parameters() if n.startswith(names)]
    return torch.optim.AdamW([dict(params=local,lr=lr_for_update(0),lr_multiplier=1.),
                             dict(params=added,lr=10*lr_for_update(0),lr_multiplier=10.)],
                            weight_decay=1e-4,foreach=False)


def freeze(root):
    config=json.loads((PREPARATION/'PREPARATION.json').read_text())
    files=sorted(PARENT_DIR.glob('*.py'))+sorted(HERE.glob('*.py'))+sorted(LEGACY.glob('*.py'))+sorted((LEGACY/'vendor/drone_rf').glob('*.py'))
    sources={str(p.relative_to(REPO)):sha256(p) for p in files}
    data_files=[PREPARATION/name for name in ('PREPARATION.json','TRAIN.npy','VALIDATION.npy')]
    data_files.append(Path(config['manifest']))
    if sha256(PARENT)!=PARENT_SHA256:
        raise RuntimeError('Selected common parent changed')
    data_files.append(PARENT)
    protocol=dict(status='FROZEN_LOCAL_GLOBAL_FUSION_DEVELOPMENT_COMPARISON',seed=0,arms=list(ARMS),
        epochs=EPOCHS,examples_per_epoch=2400,examples_per_arm=12000,effective_batch=BATCH,microbatch=MICROBATCH,
        updates_per_epoch=UPDATES_PER_EPOCH,updates_per_arm=TOTAL_UPDATES,
        optimizer='AdamW; fresh both arms',learning_rate='local cosine 1e-5 to 1e-6; added branch 10x; 1500 new updates',
        weight_decay=1e-4,gradient_clip_norm=1.,count_loss_weight=.1,precision='float32',
        initialization='identical selected short-context e5 parent (1500 prior updates); fresh seed0 new branch',
        parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA256,parent_prior_updates=1500,
        exposure='repeat the same admitted 5-epoch training schedule for both arms; no new recordings',
        preparation_used_for='data and feature schedule only; old initialization/optimizer fields ignored',
        selection='minimum validation mean NMSE across counts 2 and 3, include epoch0; report count1 separately',
        acceptance='both counts 2/3 NMSE lower AND complex SI-SDR higher; one-seed development only',
        dataset='RFUAV',same_dataset=True,same_native_band=True,native_center_offsets_preserved=False,
        scored_window_samples=63872,long_iq_samples=1048576,power_context_samples=2097152,fs_hz=100000000,
        heldout_iq_access=False,controller_classes_excluded=True,full_capacity=True,
        source_sha256=sources,data_sha256={str(p):sha256(p) for p in data_files},
        context='same local waveform path and mean power features; added complex branch sees masked/local or full 10.48576ms IQ',
        packing='lossless pack16 before learned strided context encoding; subsequent features are lossy',
        main_contrast='identical dual-path model; added context branch alone changes observed IQ; context normalization response also changes',
        fusion='bounded multiplicative gain plus additive bias at 64-channel fine and 1024-channel coarse features; scored-area centers only',
        complex_context='four learned stride4 convs after lossless pack16; 256 ordered tokens; seven two-conv dilated TCN blocks',
        interpretation='learned context compression is not guaranteed lossless; no claim of pure periodicity effect',
        milestones=[1,5],validation_cases=630)
    path=root/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:
            raise RuntimeError('Frozen source/data/protocol changed: new run directory required')
    else:
        for name in sources:
            destination=root/'source_snapshot'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO/name,destination)
        write_json(path,protocol)
    return sha256(path)


def gpu_check():
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA unavailable; no CPU training fallback')
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'],text=True).splitlines()
    for line in lines:
        pid,memory=line.split(','); pid=int(pid)
        if pid==os.getpid():
            continue
        p=Path('/proc',str(pid))
        if (p.joinpath('exe').resolve()!=Path('/usr/share/rustdesk/rustdesk')
                or p.stat().st_uid!=os.getuid() or not 0<=float(memory)<=256):
            raise RuntimeError(f'Another CUDA task is active: {pid}')


def train_epoch(arm,epoch,train,validation,root,frozen):
    folder=root/arm; last=folder/'LAST.pt'; receipt=folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        if not last.exists() or json.loads(receipt.read_text())['protocol_sha256']!=frozen:
            raise RuntimeError('Unverified completed epoch')
        return
    net=build(arm).cuda()
    optimizer=optimizer_for(net)
    updates=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=frozen or saved['epoch']!=epoch-1:
            raise RuntimeError('Nonconsecutive or incompatible resume')
        net.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        updates=saved['updates']; best=saved['best']; del saved
        best_saved=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        if best_saved['best']!=best or best_saved['protocol_sha256']!=frozen:
            raise RuntimeError('Incomplete checkpoint transaction; recovery required')
        del best_saved
    elif epoch==1:
        metrics=evaluate(net,validation,folder/'VALIDATION_000.json',0)
        best=dict(epoch=0,metric=metrics['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=frozen))
    else:
        raise RuntimeError('Missing previous epoch')
    if len(train)!=2400 or updates!=(epoch-1)*UPDATES_PER_EPOCH:
        raise RuntimeError('Matched training exposure changed')
    net.train(); optimizer.zero_grad(set_to_none=True)
    start=time.time(); total=0.; torch.cuda.reset_peak_memory_stats()
    for start_index in range(0,len(train),MICROBATCH):
        index=start_index+MICROBATCH-1
        item=stack_batch([train[i] for i in range(start_index,index+1)],'cuda')
        loss=objective(net,item)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is not finite')
        (loss*(MICROBATCH/BATCH)).backward(); total+=float(loss.detach())*MICROBATCH
        if (index+1)%BATCH==0:
            lr=lr_for_update(updates)
            for group in optimizer.param_groups: group['lr']=lr*group['lr_multiplier']
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); updates+=1
            if updates%10==0:
                progress=dict(stage='TRAIN',arm=arm,epoch=epoch,updates=updates,total_updates=TOTAL_UPDATES,
                    examples=index+1,epoch_examples=len(train),learning_rate=lr,new_branch_learning_rate=10*lr,
                    average_loss=total/(index+1),seconds=time.time()-start,time=time.time(),pid=os.getpid())
                write_json(root/'PROGRESS.json',progress); print(json.dumps(progress),flush=True)
    torch.cuda.synchronize(); train_seconds=time.time()-start
    peak=torch.cuda.max_memory_allocated()
    metrics=evaluate(net,validation,folder/f'VALIDATION_{epoch:03d}.json',epoch)
    if metrics['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=metrics['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=frozen))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=frozen,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    summary=dict(epoch=epoch,updates=updates,best=best,train_seconds=train_seconds,
        peak_allocated_bytes=peak,average_loss=total/len(train),protocol_sha256=frozen,
        metrics={k:v for k,v in metrics.items() if k!='rows'})
    write_json(receipt,summary)
    print(json.dumps(dict(stage='EPOCH_COMPLETE',arm=arm,**summary)),flush=True)
    if epoch in (1,5):
        shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    del net,optimizer,loss
    gc.collect(); torch.cuda.empty_cache()


def run(root,preflight_only=False):
    gpu_check()
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    frozen=freeze(root)
    write_json(root/'RUN_STATE.json',dict(status='PREFLIGHT_RUNNING',pid=os.getpid(),time=time.time(),protocol_sha256=frozen))
    train=admitted_dataset(PREPARATION,'train_pack',1)
    for arm in ARMS:
        folder=root/arm; folder.mkdir(exist_ok=True)
        preflight(arm,train,folder,frozen,optimizer_for,objective,MICROBATCH)
    del train
    if preflight_only:
        write_json(root/'RUN_STATE.json',dict(status='PREFLIGHT_COMPLETE_TRAINING_NOT_STARTED',time=time.time()))
        return
    validation=admitted_dataset(PREPARATION,'validation_pack',1)
    write_json(root/'RUN_STATE.json',dict(status='TRAINING_RUNNING',pid=os.getpid(),time=time.time(),protocol_sha256=frozen))
    for epoch in range(1,EPOCHS+1):
        freeze(root)
        train=admitted_dataset(PREPARATION,'train_pack',epoch)
        # Start the full-context arm first; reverse order each epoch.
        for arm in reversed(ARMS) if epoch%2 else ARMS:
            train_epoch(arm,epoch,train,validation,root,frozen)
        del train
        write_json(root/'MATCHED_PROGRESS.json',dict(completed_epochs=epoch,updates_per_arm=epoch*UPDATES_PER_EPOCH,
            time=time.time(),arms=list(ARMS),protocol_sha256=frozen))
    write_json(root/'COMPLETE.json',dict(status='MATCHED_COMPARISON_COMPLETE',epochs=EPOCHS,
        updates_per_arm=TOTAL_UPDATES,protocol_sha256=frozen,time=time.time()))
    write_json(root/'RUN_STATE.json',dict(status='COMPLETED',time=time.time()))
    print('MATCHED_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:])); os.nice(10)
    args.run.mkdir(parents=True,exist_ok=True)
    try:
        with LOCK.open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.run,args.preflight_only)
    except Exception:
        write_json(args.run/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        write_json(args.run/'RUN_STATE.json',dict(status='FAILED',time=time.time()))
        raise
