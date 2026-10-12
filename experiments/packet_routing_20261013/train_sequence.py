"""One matched three-epoch comparison after the immutable 100-epoch parent.

Not a restart of the old queue. Whole model capacity is retained. Model and
optimizer states live on the external disk; public receipts contain no I/Q.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch

from sequence_data import OUT, ROOT, Sequences, Mixtures, ROLES, read, sha256, write_json
from sequence_objective import assert_replay_safe, replay_bundle
from sequence_eval import short_eval, long_eval
from models import build, predict
from study import batch, atomic_torch

HERE=Path(__file__).resolve().parent
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
PUBLIC=ROOT/'reports/2026-10-13'
PLAN=PUBLIC/'SEQUENCE_ASSIGNMENT_PLAN_KO.md'
LOCK=ROOT.parent/'uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock'
STOP=False


def state(status,**kw):
    write_json(OUT/'STATE.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw))


def register():
    assert not (OUT/'PROTOCOL.json').exists(),'Protocol already registered'
    parent=read(PARENT/'PROTOCOL.json')
    sources=set(parent['source_sha256'])
    sources.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py') if p.name in
        ('sequence_data.py','sequence_objective.py','sequence_eval.py','test_sequence.py','train_sequence.py'))
    sources.add(str(PLAN.relative_to(ROOT)))
    hashes={name:sha256(ROOT/name) for name in sorted(sources)}
    p=dict(registered_time=time.time(),source_sha256=hashes,data_plan_sha256=sha256(OUT/'DATA_PLAN.json'),
        parent_protocol_sha256=sha256(PARENT/'PROTOCOL.json'),
        initialization='After audited100, select existing best by original short DEV630 macroNMSE rule; bind before any new update',
        arms=['shared','independent'],epochs_per_arm=3,updates_per_epoch=96,bundles_per_epoch=768,
        effective_bundles=8,chunks=3,parameters=32142859,learning_rate=1e-5,weight_decay=1e-4,
        seed=0,precision='FP32, TF32 disabled',max_gradient_norm=1.,
        inference='unchanged C8, no reference input',control_count=1,
        comparison='matched final additional epoch3; intermediate results exploratory',
        primary='two/three source waveform reconstruction plus full-sequence output consistency',
        no_heldout_access=True,independent_recording_validation=False)
    write_json(OUT/'PROTOCOL.json',p)
    for rel,h in hashes.items():
        dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        assert sha256(dst)==h
    print('REGISTERED_SEQUENCE_PROTOCOL',sha256(OUT/'PROTOCOL.json'),flush=True)


def verify(p):
    assert sha256(OUT/'DATA_PLAN.json')==p['data_plan_sha256']
    assert sha256(PARENT/'PROTOCOL.json')==p['parent_protocol_sha256']
    for rel,h in p['source_sha256'].items():
        assert sha256(ROOT/rel)==h,rel
        assert sha256(OUT/'source_snapshot'/rel)==h,rel


def wait_parent():
    while True:
        s=read(PARENT/'STATE.json')
        if s['status']=='COMPLETE_AUDITED' and s['epoch']==100:
            event=read(PARENT/'EPOCH_100.json');audit=read(PARENT/'AUDIT_100.json')
            assert audit['status']=='PASS' and audit['event_sha256']==sha256(PARENT/'EPOCH_100.json')
            return event
        if s['status'] in ('FAILED','STOPPED_WITH_RECOVERY'):
            raise RuntimeError(f'Parent requires attention: {s["status"]}')
        state('WAITING_PARENT_100',parent_state=s)
        if STOP:return None
        time.sleep(30)


def bind_initial(event):
    selected=int(event['best']['epoch'])
    run=PARENT if selected>50 else ROOT/'local/bandwidth_timeblock_20261011_v1'
    e=read(run/f'EPOCH_{selected:03d}.json');a=read(run/f'AUDIT_{selected:03d}.json')
    assert a['status']=='PASS' and a['event_sha256']==sha256(run/f'EPOCH_{selected:03d}.json')
    path=run/f'ACTUAL_{selected:03d}.pt';digest=sha256(path)
    assert digest==e['checkpoint_sha256']
    init=dict(epoch=selected,checkpoint=str(path),checkpoint_sha256=digest,
        parent_final_event_sha256=sha256(PARENT/'EPOCH_100.json'),
        selected_event_sha256=sha256(run/f'EPOCH_{selected:03d}.json'),
        selected_validation=e['validation'],created_time=time.time())
    if (OUT/'INITIALIZATION.json').exists():
        saved=read(OUT/'INITIALIZATION.json')
        assert all(saved[k]==init[k] for k in init if k!='created_time')
        return saved
    write_json(OUT/'INITIALIZATION.json',init)
    return init


def model_from(path):
    model=build('unet_mean').cuda()
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(saved['model'])
    assert sum(v.numel() for v in model.parameters())==32142859
    assert_replay_safe(model)
    return model,saved


def preflight(initial):
    if (OUT/'PREFLIGHT.json').exists():
        assert read(OUT/'PREFLIGHT.json')['status']=='PASS';return
    net,_=model_from(initial['checkpoint']);net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)
    data=Sequences(ROLES[0],1);rows=[]
    for count in (1,2,3):
        i=next(i for i,r in enumerate(data.rows) if r['count']==count)
        opt.zero_grad(set_to_none=True)
        before=net.output.weight.detach().clone()
        losses,diag=replay_bundle(net,predict,[batch(x,'cuda') for x in data[i]],'shared',1)
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
        assert any(q.grad is not None and bool(q.grad.abs().max()>0) for q in net.down.parameters())
        opt.step()
        assert bool((before-net.output.weight).abs().max()>0)
        rows.append(dict(count=count,gradient_norm=norm,mean_loss=np.mean(losses,axis=0).tolist(),
            max_replay_error=diag['max_replay_error'],output_changed=True))
    write_json(OUT/'PREFLIGHT.json',dict(status='PASS',rows=rows,weights_discarded=True,
        initial_checkpoint_sha256=initial['checkpoint_sha256'],max_gpu_allocated=torch.cuda.max_memory_allocated()))
    del net,opt,data;gc.collect();torch.cuda.empty_cache()
    print('GPU_PREFLIGHT_PASS discarded weights',flush=True)


def save(path,net,opt,epoch,updates,next_bundle,losses,diagnostics,seconds,evaluated):
    atomic_torch(path,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        next_bundle=next_bundle,losses=losses,diagnostics=diagnostics,seconds=seconds,epoch_evaluated=evaluated,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        protocol_sha256=sha256(OUT/'PROTOCOL.json')))


def epoch_run(arm,epoch,initial):
    directory=OUT/arm;directory.mkdir(exist_ok=True)
    if (directory/f'AUDIT_{epoch:03d}.json').exists():
        assert read(directory/f'AUDIT_{epoch:03d}.json')['status']=='PASS';return
    path=directory/'RECOVERY.pt'
    if not path.exists():path=Path(initial['checkpoint'])
    net,saved=model_from(path)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)
    start,updates,losses,diagnostics,offset=0,(epoch-1)*96,[],[],0.
    if 'optimizer' in saved:
        assert saved['protocol_sha256']==sha256(OUT/'PROTOCOL.json')
        opt.load_state_dict(saved['optimizer']);updates=saved['updates']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if saved['epoch']==epoch and not saved['epoch_evaluated']:
            start=saved['next_bundle'];losses=saved['losses'];diagnostics=saved['diagnostics'];offset=saved['seconds']
        else:assert saved['epoch']==epoch-1 and saved['epoch_evaluated']
    else:assert epoch==1
    del saved
    train=Sequences(ROLES[0],epoch);net.train();opt.zero_grad(set_to_none=True)
    assert start%8==0 and updates==(epoch-1)*96+start//8
    began=time.time()
    for i in range(start,len(train)):
        values,diag=replay_bundle(net,predict,[batch(x,'cuda') for x in train[i]],arm,8)
        losses.extend(values);diagnostics.append(diag)
        if (i+1)%8==0:
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True))
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            elapsed=offset+time.time()-began
            if updates%12==0 or STOP:
                save(directory/'RECOVERY.pt',net,opt,epoch,updates,i+1,losses,diagnostics,elapsed,False)
            state('TRAINING',arm=arm,epoch=epoch,updates=updates,bundles=i+1,total=len(train),seconds=elapsed,
                recent_loss=np.mean(losses[-24:],axis=0).tolist(),gradient_norm=norm)
            if STOP:
                state('STOPPED_WITH_RECOVERY',arm=arm,epoch=epoch,updates=updates,bundles=i+1)
                return False
    train_seconds=offset+time.time()-began
    assert updates==epoch*96 and len(losses)==2304 and len(diagnostics)==768
    def progress(stage,**kw):state(stage,arm=arm,epoch=epoch,updates=updates,**kw)
    t=time.time();short=short_eval(net,'cuda',progress);long=long_eval(net,'cuda',progress)
    write_json(directory/f'SHORT_{epoch:03d}.json',short);write_json(directory/f'LONG_{epoch:03d}.json',long)
    model_path=directory/f'ACTUAL_{epoch:03d}.pt'
    atomic_torch(model_path,dict(model=net.state_dict(),epoch=epoch,updates=updates,
        arm=arm,protocol_sha256=sha256(OUT/'PROTOCOL.json')))
    event=dict(arm=arm,epoch=epoch,updates=updates,train_seconds=train_seconds,evaluation_seconds=time.time()-t,
        short={k:v for k,v in short.items() if k!='rows'},
        long_merged={k:v for k,v in long['merged'].items() if k!='rows'},
        mean_training_loss=np.mean(losses,axis=0).tolist(),
        training_local_disagreement_fraction=float(np.mean([d['local_disagreement'] for d in diagnostics])),
        training_shared_minus_local_loss=float(np.mean([d['shared_min_loss']-d['local_min_loss'] for d in diagnostics])),
        checkpoint_sha256=sha256(model_path),short_sha256=sha256(directory/f'SHORT_{epoch:03d}.json'),
        long_sha256=sha256(directory/f'LONG_{epoch:03d}.json'),protocol_sha256=sha256(OUT/'PROTOCOL.json'),
        initial_epoch=initial['epoch'],initial_checkpoint_sha256=initial['checkpoint_sha256'],
        independent_recording_validation=False)
    write_json(directory/f'EPOCH_{epoch:03d}.json',event)
    save(directory/'RECOVERY.pt',net,opt,epoch,updates,len(train),losses,diagnostics,train_seconds,True)
    del net,opt,train;gc.collect();torch.cuda.empty_cache()
    audit_epoch(arm,epoch)
    write_json(PUBLIC/f'SEQUENCE_{arm.upper()}_E{epoch:03d}.json',event)
    print('EPOCH_COMPLETE',arm,epoch,json.dumps(event['short']),flush=True)
    return True


def audit_epoch(arm,epoch):
    from sequence_eval import summarize
    d=OUT/arm;e=read(d/f'EPOCH_{epoch:03d}.json')
    assert sha256(d/f'ACTUAL_{epoch:03d}.pt')==e['checkpoint_sha256']
    assert sha256(d/f'SHORT_{epoch:03d}.json')==e['short_sha256']
    assert sha256(d/f'LONG_{epoch:03d}.json')==e['long_sha256']
    short=read(d/f'SHORT_{epoch:03d}.json');long=read(d/f'LONG_{epoch:03d}.json')
    assert summarize(short['rows'])=={k:short[k] for k in ('by_count','selection_nmse')}
    assert summarize(long['merged']['rows'])=={k:long['merged'][k] for k in ('by_count','selection_nmse')}
    assert len(short['rows'])==630 and len(long['merged']['rows'])==24
    saved=torch.load(d/'RECOVERY.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['updates']==epoch*96 and saved['next_bundle']==768 and saved['epoch_evaluated']
    steps=[int(v['step']) for v in saved['optimizer']['state'].values()]
    assert steps and min(steps)==max(steps)==epoch*96
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    write_json(d/f'AUDIT_{epoch:03d}.json',dict(status='PASS',arm=arm,epoch=epoch,
        event_sha256=sha256(d/f'EPOCH_{epoch:03d}.json'),optimizer_steps=epoch*96,
        checks=['checkpoint and row hashes','saved-row summaries','optimizer updates','finite weights'],time=time.time()))


def run():
    p=read(OUT/'PROTOCOL.json');verify(p)
    parent=wait_parent()
    if parent is None:return
    for epoch in (1,2,3):Sequences(ROLES[0],epoch)
    Sequences(ROLES[1])
    initial=bind_initial(parent)
    assert sha256(initial['checkpoint'])==initial['checkpoint_sha256']
    with LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        verify(p);assert torch.cuda.is_available()
        torch.set_num_threads(2);torch.manual_seed(0)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        preflight(initial)
        if not (OUT/'INITIAL_LONG.json').exists():
            net,_=model_from(initial['checkpoint'])
            write_json(OUT/'INITIAL_LONG.json',long_eval(net,'cuda',lambda s,**kw:state(s,arm='initial',epoch=0,**kw)))
            del net;gc.collect();torch.cuda.empty_cache()
        for epoch in (1,2,3):
            for arm in ('shared','independent'):
                verify(p)
                if STOP:return
                if epoch_run(arm,epoch,initial) is False:return
        state('COMPLETE_AUDITED',arms=p['arms'],epochs_per_arm=3,total_updates=576)


def stop(*_):
    global STOP
    STOP=True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run'])
    action=parser.parse_args().action
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (OUT/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            register() if action=='register' else run()
    except BaseException:
        write_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        state('FAILED');raise
