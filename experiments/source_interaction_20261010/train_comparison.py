"""Matched full U-Net source-head comparison, registered AFTER GPU preflight.

The unmodified retained backbone is trained as well. Neither preflight weights
nor true counts enter inference. A completed preflight is a prerequisite, not
evidence of validation superiority. All epochs, including failures, are saved.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/architecture_audit_20261009'))
import fit_diagnostic as fit
import gpu_start_guard as guard
import watch_epochs as watch
from models import build,predict
from native_data import NativeMixtures
from study import atomic_torch
from drone_rf.losses import pit_waveform_loss
from source_head import augment

ARMS=('retained_unet','source_interaction')
CONTROL_CHECK=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAINING_CHECK.json'


def make_model(arm,parent=None):
    net=build('unet_mean')
    if parent is not None:
        value=torch.load(parent,map_location='cpu',weights_only=False)
        net.load_state_dict(value['model'])
    if arm=='source_interaction':augment(net)
    elif arm!='retained_unet':raise ValueError(arm)
    return net


def register(root,dependency):
    if (root/'PROTOCOL.json').exists():raise ValueError('Use --resume for a registered study')
    previous=watch.read(dependency/'PROTOCOL.json')
    finished=watch.read(dependency/'COMPLETE.json')
    control=watch.read(CONTROL_CHECK)
    if (control['status']!='PASS' or control['worker_sha256']!=watch.digest(Path(__file__))
            or control['parent_checkpoint_sha256']!=previous['parent_checkpoint_sha256']
            or control['checker_sha256']!=watch.digest(HERE/'check_training_control.py')):
        raise ValueError('Training control/parent-state check missing')
    if (finished['protocol_sha256']!=watch.digest(dependency/'PROTOCOL.json')
            or not finished['initial_gpu_predictions_exactly_equal']
            or any(not r['finite_gradients'] or not r['both_counts_improved'] for r in finished['results'])):
        raise ValueError('GPU preflight has not met its preregistered checks')
    sources=dict(previous['source_sha256'])
    for source in (Path(__file__),HERE/'check_training_control.py',HERE/'audit_comparison.py'):
        sources[str(source.relative_to(ROOT))]=watch.digest(source)
    template=ROOT/'local/native_frequency_20261009_v1/phase/native_selected/BASELINE.json'
    watch.validate(template,None)
    plan=dict(status='REGISTERED_MATCHED_SOURCE_INTERACTION',source_sha256=sources,
        dependency=str(dependency),dependency_protocol_sha256=watch.digest(dependency/'PROTOCOL.json'),
        dependency_complete_sha256=watch.digest(dependency/'COMPLETE.json'),
        training_control_check_sha256=watch.digest(CONTROL_CHECK),
        parent_checkpoint=previous['parent_checkpoint'],parent_checkpoint_sha256=previous['parent_checkpoint_sha256'],
        parent_selected_epoch=2,parameters=previous['parameters'],gpu_display_policy=previous['gpu_display_policy'],
        preparation=previous['preparation'],preparation_sha256=previous['preparation_sha256'],
        validation_identity_template=str(template),validation_identity_sha256=watch.digest(template),
        seed=0,arms=list(ARMS),epochs_per_arm=3,updates_per_arm=225,
        train_schedule_epochs=[1,2,3],examples_per_epoch=2400,validation_cases=630,
        effective_batch=32,microbatch=1,learning_rate=1e-5,weight_decay=1e-4,clip_norm=1.,
        optimizer='fresh AdamW1e-5 in BOTH arms; use the retained native parent study learning rate',
        initialization='identical retained native e2 weights; added readout zero; preflight weights discarded',
        loss='same original PIT NMSE+coherence+inactive/background and0.1 count CE in BOTH arms',
        change='source-interaction output head only; all parent parameters remain trainable',
        unmodified='RF geometry, data, schedule, input, context, loss, count head, seed and update budget',
        selection='minimum mean raw NMSE counts2/3 including e0; same original rule for both arms',
        acceptance='counts2/3 raw NMSE lower AND absolute complex SI-SDR higher than matched control and parent; weakest/1source/count failures retained',
        inference='single pass; no references, categories, true count or oracle calibration',
        ordering='e1/e3 retained then interaction; e2 reversed',heldout_read=False,
        limitation='one seed, repeated development validation, five validation recording groups and one three-category composition; no architecture superiority before results',
        registered_at=time.time())
    for rel,digest in sources.items():
        if watch.digest(ROOT/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    watch.write(root/'PROTOCOL.json',plan)
    return plan


def verify(root,p):
    for rel,digest in p['source_sha256'].items():
        if watch.digest(ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen source changed: '+rel)
    for path,key in ((Path(p['dependency'])/'PROTOCOL.json','dependency_protocol_sha256'),
        (Path(p['dependency'])/'COMPLETE.json','dependency_complete_sha256'),
        (Path(p['parent_checkpoint']),'parent_checkpoint_sha256'),
        (Path(p['preparation'])/'PREPARATION.json','preparation_sha256'),
        (CONTROL_CHECK,'training_control_check_sha256'),
        (Path(p['validation_identity_template']),'validation_identity_sha256')):
        if watch.digest(path)!=p[key]:raise ValueError('Changed dependency: '+str(path))


def train_epoch(root,p,arm,epoch,train,validation,identities):
    folder=root/arm;folder.mkdir(exist_ok=True)
    digest=watch.digest(root/'PROTOCOL.json')
    receipt=folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        old=watch.read(receipt)
        if old['protocol_sha256']!=digest or old['updates']!=epoch*75 or old['arm']!=arm:raise ValueError('Stale epoch receipt')
        watch.validate(folder/f'VALIDATION_{epoch:03d}.json',identities)
        return
    net=make_model(arm,p['parent_checkpoint']).cuda()
    assert sum(q.numel() for q in net.parameters())==p['parameters'][arm]
    opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'],foreach=False)
    last=folder/'LAST.pt'
    if last.exists():
        old=torch.load(last,map_location='cpu',weights_only=False)
        if old['epoch']!=epoch-1 or old['protocol_sha256']!=digest or old['arm']!=arm:raise ValueError('Nonconsecutive resume')
        net.load_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
        best,updates=old['best'],old['updates']
        stored=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False)
        if stored['best']!=best or stored['protocol_sha256']!=digest:raise ValueError('Incomplete BEST transaction')
        torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng'])
        del old,stored
    else:
        if epoch!=1:raise ValueError('Missing prior epoch')
        initial=fit.base.validate(net,validation,folder/'VALIDATION_000.json',0)
        actual,_=watch.validate(folder/'VALIDATION_000.json',identities)
        expected,_=watch.validate(Path(p['validation_identity_template']),identities)
        for a,b in zip(actual['by_count'],expected['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):
                if not abs(a[key]-b[key])<2e-6:raise ValueError('Retained parent validation not reproduced')
        best=dict(epoch=0,metric=initial['selection_nmse']);updates=0
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=digest))
        torch.manual_seed(0)
    if len(train)!=2400 or updates!=(epoch-1)*75:raise ValueError('Wrong training budget')
    net.train();opt.zero_grad(set_to_none=True);began=time.time();norms=[];loss_sum=0.
    torch.cuda.reset_peak_memory_stats()
    for index in range(len(train)):
        item=fit.base.batch([train[index]])
        estimates,logits=predict(net,item)
        loss=pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
        loss=loss+.1*F.cross_entropy(logits,item['construction_count']-1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        (loss/32).backward();loss_sum+=float(loss.detach())
        if (index+1)%32==0:
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True);updates+=1
            if updates%5==0:
                watch.write(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=epoch,
                    updates=updates,examples=index+1,total=2400,seconds=time.time()-began,pid=os.getpid(),time=time.time()))
    elapsed=time.time()-began
    watch.write(root/'STATE.json',dict(status='VALIDATING',arm=arm,epoch=epoch,updates=updates,pid=os.getpid(),time=time.time()))
    measured=fit.base.validate(net,validation,folder/f'VALIDATION_{epoch:03d}.json',epoch)
    watch.validate(folder/f'VALIDATION_{epoch:03d}.json',identities)
    if measured['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=measured['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=digest))
    atomic_torch(folder/f'ACTUAL_{epoch:03d}.pt',dict(model=net.state_dict(),arm=arm,epoch=epoch,updates=updates,protocol_sha256=digest))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,arm=arm,epoch=epoch,updates=updates,
        protocol_sha256=digest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    event=dict(arm=arm,epoch=epoch,updates=updates,best=best,parameters=p['parameters'][arm],
        protocol_sha256=digest,train_seconds=elapsed,mean_training_loss=loss_sum/2400,
        preclip_gradient_norms=norms,peak_bytes=torch.cuda.max_memory_allocated(),
        validation={k:v for k,v in measured.items() if k!='rows'})
    watch.write(receipt,event);print(event,flush=True)
    del net,opt,item,estimates,logits,loss,norm
    gc.collect();torch.cuda.empty_cache()


def progress(root,public):
    p=watch.read(root/'PROTOCOL.json');events=[]
    for epoch in (1,2,3):
        for arm in ARMS:
            path=root/arm/f'EPOCH_{epoch:03d}.json'
            if path.exists():events.append(watch.read(path))
    result=dict(status='COMPLETE' if len(events)==6 else 'PARTIAL',events=events,
        protocol_sha256=watch.digest(root/'PROTOCOL.json'),heldout_read=False,time=time.time())
    watch.write(public.with_suffix('.json'),result)
    lines=['# 성분 상호작용 U-Net: 완료 epoch','',
        '동일한 보존 U-Net e2·RFUAV 분할·native RF 배치·원 손실·AdamW1e-5·각 추가3epoch/225업데이트. '
        '원 규모 본체를 양쪽 모두 학습한다. 단일 추론이며 추가층은37,888파라미터다.','',
        '| 군 | 추가 epoch | NMSE 2/3 ↓ | SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ | 선택 epoch |',
        '|---|---:|---|---|---|---:|']
    for e in events:
        a,b=e['validation']['by_count'][1:]
        lines.append(f"|{e['arm']}|{e['epoch']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
            f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{e['best']['epoch']}|")
    lines+=['','한 seed의 반복 개발검증이다. 원래 학습 이력·한 세 성분 조합·다섯 검증 기록 묶음의 한계를 유지한다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))


def run(root,p,public):
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        watch.write(root/'GPU_START_CHECK.json',guard.wait_for_predecessor(
            watch.read(Path(p['dependency'])/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        _,identities=watch.validate(Path(p['validation_identity_template']),None)
        validation=NativeMixtures(p['preparation'],'validation_pack',1)
        for epoch in (1,2,3):
            train=NativeMixtures(p['preparation'],'train_pack',epoch)
            for arm in (ARMS if epoch%2 else tuple(reversed(ARMS))):
                verify(root,p);train_epoch(root,p,arm,epoch,train,validation,identities);progress(root,public)
            del train
        for arm in ARMS:
            saved=torch.load(root/arm/'LAST.pt',map_location='cpu',weights_only=False)
            if saved['updates']!=225 or {int(s['step']) for s in saved['optimizer']['state'].values()}!={225}:
                raise ValueError('Incomplete final optimizer budget')
            if not all(torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Nonfinite final model')
            del saved
        verify(root,p)
        watch.write(root/'COMPLETE.json',dict(status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),
            epochs_per_arm=3,updates_per_arm=225,heldout_read=False,time=time.time()))
        watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
        progress(root,public)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','dependency','public'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=watch.read(root/'PROTOCOL.json') if a.resume else register(root,a.dependency.resolve())
            if p['dependency']!=str(a.dependency.resolve()):raise ValueError('Resume dependency differs')
            run(root,p,a.public.resolve())
        except Exception:
            watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
            raise
