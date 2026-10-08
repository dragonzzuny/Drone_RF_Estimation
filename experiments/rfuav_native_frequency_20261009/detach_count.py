"""Matched native-RF ablation: stop count gradients at the shared encoder.

Count-head weights still learn; forward values, waveform loss and full model
capacity are unchanged. Reuse the completed control's exact five-epoch budget.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from native_data import ROOT, NativeMixtures, sha256, write_json
import run_native as base
from report import summarize
from drone_rf.losses import pit_waveform_loss

original_model = base.model


def detach_input(module, inputs):
    return (inputs[0].detach(),)


def detached_model():
    net=original_model()
    net.count_head.register_forward_pre_hook(detach_input)
    return net


def read(path):
    return json.loads(path.read_text())


def freeze(root, control):
    if not read(control/'gradient/COMPLETE.json')['next_test_triggered']:
        raise ValueError('Registered gradient trigger not met')
    parent=read(control/'GPU_PROTOCOL.json')
    sources=dict(parent['source_sha256'])
    for path in (Path(__file__).resolve(),Path(__file__).with_name('report.py').resolve()):
        sources[str(path.relative_to(ROOT))]=sha256(path)
    plan=dict(parent)
    plan.update(status='REGISTERED_MATCHED_COUNT_GRADIENT_DETACH',source_sha256=sources,
        matched_control_run=str(control),control_protocol_sha256=sha256(control/'GPU_PROTOCOL.json'),
        control_completion_sha256=sha256(control/'GPU_COMPLETE.json'),
        gradient_trigger_sha256=sha256(control/'gradient/COMPLETE.json'),
        change='detach shared encoded input to count_head; train count_head normally; waveform encoder receives waveform gradient only',
        forward_change=False,parameters=32142859,model_capacity_changed=False,
        comparison='same original parent, TRAIN/validation mixtures, fresh optimizer, seed0 and 5epochs/375updates; reuse completed native control',
        adaptive_development_test=True,
        acceptance='both counts2/3 lower NMSE and higher complex SI-SDR than matched selected control; report weak and count separately')
    path=root/'GPU_PROTOCOL.json'
    if path.exists() and read(path)!=plan:raise ValueError('Detached protocol changed')
    for rel,digest in sources.items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Control source changed')
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=digest:raise ValueError('Source snapshot changed')
        if not target.exists():shutil.copyfile(ROOT/rel,target)
    if not path.exists():write_json(path,plan)
    if sha256(base.PARENT)!=parent['parent_sha256']:raise ValueError('Parent checkpoint changed')
    for source,digest in read(control/'PREP_PROTOCOL.json')['data_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Original metadata changed')
    return sha256(path)


def preflight(root, protocol, train):
    path=root/'DETACH_PREFLIGHT.json'
    if path.exists():
        if read(path)['protocol_sha256']!=protocol:raise ValueError('Old preflight changed')
        return
    item=base.batch([train[2]])
    net=original_model().eval()
    with torch.no_grad():before,logits_before=base.predict(net,item)
    handle=net.count_head.register_forward_pre_hook(detach_input)
    after,logits=base.predict(net,item)
    if not torch.equal(after,before) or not torch.equal(logits,logits_before):
        raise ValueError('Detach unexpectedly changed forward output')
    context=list(net.context_encoder.parameters());head=list(net.count_head.parameters())
    count=.1*F.cross_entropy(logits,item['construction_count']-1)
    count_grads=torch.autograd.grad(count,context+head,allow_unused=True)
    if any(g is not None for g in count_grads[:len(context)]):raise ValueError('Count still updates encoder')
    if not any(g is not None and bool(g.abs().sum()>0) for g in count_grads[len(context):]):
        raise ValueError('Count head stopped learning')
    waveform=pit_waveform_loss(after,item['references'],item['active'],item['mixture'])['loss']
    wave_grads=torch.autograd.grad(waveform,context)
    if not all(bool(torch.isfinite(g).all()) for g in wave_grads) or not any(bool(g.abs().sum()>0) for g in wave_grads):
        raise ValueError('Waveform gradient lost')
    write_json(path,dict(status='PASS',protocol_sha256=protocol,forward_bitwise_equal=True,
        count_encoder_gradient_none=True,count_head_gradient_nonzero=True,waveform_encoder_gradient_finite_nonzero=True,
        model_updates=0,train_index=2,heldout_read=False))
    handle.remove();del net,item,before,after,logits,logits_before,count,count_grads,waveform,wave_grads
    gc.collect();torch.cuda.empty_cache()


def comparison(root, control, epoch):
    histories={}
    for name,folder in (('joint_count',control),('detached_count',root)):
        history=[]
        for e in range(epoch+1):
            path=folder/'gpu'/f'VALIDATION_{e:03d}.json'
            summary=summarize(path)
            if e and read(folder/'gpu'/f'EPOCH_{e:03d}.json')['updates']!=75*e:
                raise ValueError('Unequal update budget')
            history.append(dict(epoch=e,**summary))
        histories[name]=dict(history=history,selected=min(history,key=lambda x:x['selection_nmse']))
    for e in range(epoch+1):
        a=read(control/'gpu'/f'VALIDATION_{e:03d}.json')['rows']
        b=read(root/'gpu'/f'VALIDATION_{e:03d}.json')['rows']
        for x,y in zip(a,b):
            for key in ('index','count','categories','pack_ids','nominal_levels_db'):
                if x[key]!=y[key]:raise ValueError('Matched cases differ')
            np.testing.assert_allclose(x['reference_power'],y['reference_power'],rtol=1e-10,atol=1e-15)
            if y['sum_relative_error']>1e-9:raise ValueError('Detached mixture sum failed')
            if e==0:
                for key in ('nmse','si_sdr'):
                    np.testing.assert_allclose(x[key],y[key],rtol=1e-5,atol=1e-7)
    a=histories['joint_count']['selected']['by_count'][1:]
    b=histories['detached_count']['selected']['by_count'][1:]
    result=dict(common_epochs=epoch,updates_per_arm=75*epoch,arms=histories,
        joint_improvement=all(y['mean_nmse']<x['mean_nmse'] and y['mean_si_sdr']>x['mean_si_sdr'] for x,y in zip(a,b)),
        weakest_both_improved=all(y['weakest_nmse']<x['weakest_nmse'] for x,y in zip(a,b)),
        same_initial_predictions=True,same_mixtures=True,independent_test=False,physical_aircraft_count=False)
    write_json(root/f'COMPARISON_{epoch:03d}.json',result)
    lines=['# 개수 손실의 인코더 기울기 차단 대조','',f'같은 초기 가중치·입력·seed0·각 {epoch} epoch/{75*epoch}업데이트.',
        '개수 head는 계속 학습하고, 그 입력만 detach해 공통 복원 인코더로의 개수 분류 기울기를 차단한다. '
        '원 규모와 추론 계산은 유지한다. 기존 완료 대조군을 재사용하는 탐색 개발 비교다.','',
        '| 방법 | 선택 epoch | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR | 3성분 SI-SDR | 약신호 NMSE 2/3 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name,info in histories.items():
        s=info['selected'];g=s['by_count']
        lines.append(f"| {name} | {s['epoch']} | {g[1]['mean_nmse']:.6f} | {g[2]['mean_nmse']:.6f} | {g[1]['mean_si_sdr']:.3f} | {g[2]['mean_si_sdr']:.3f} | {g[1]['weakest_nmse']:.6f} / {g[2]['weakest_nmse']:.6f} |")
    lines+=['',f"두 조건·두 지표 방향상 동시 개선: {result['joint_improvement']}. 양쪽 약신호 NMSE 개선: {result['weakest_both_improved']}.",
        '각 군은 해당 공통 예산 이내의 평균 NMSE로 선택한다. 추가 추론 평균은 적용하지 않았다. '
        '반복 개발 검증이며 새 기종·실제 동시 수신·완전 복원의 성과를 뜻하지 않는다.','']
    (root/f'COMPARISON_{epoch:03d}.md').write_text('\n'.join(lines))
    print(json.dumps(dict(stage='MATCHED_EPOCH_COMPLETE',epoch=epoch,
        selected={k:v['selected']['epoch'] for k,v in histories.items()},
        joint_improvement=result['joint_improvement'],weakest_both_improved=result['weakest_both_improved'],
        detached_by_count=histories['detached_count']['selected']['by_count'])),flush=True)


def run(root, control):
    if (root/'GPU_COMPLETE.json').exists():raise ValueError('Already complete')
    root.mkdir(exist_ok=True)
    preparation=root/'preparation'
    if not preparation.exists():preparation.symlink_to(control/'preparation',target_is_directory=True)
    if preparation.resolve()!=(control/'preparation').resolve():raise ValueError('Wrong prepared inputs')
    if not (root/'PREP_PROTOCOL.json').exists():shutil.copyfile(control/'PREP_PROTOCOL.json',root/'PREP_PROTOCOL.json')
    if sha256(root/'PREP_PROTOCOL.json')!=sha256(control/'PREP_PROTOCOL.json'):raise ValueError('Preparation differs')
    protocol=freeze(root,control)
    val=NativeMixtures(preparation,'validation_pack',1);train=NativeMixtures(preparation,'train_pack',1)
    base.model=detached_model
    with base.LOCK.open('r') as lock:
        write_json(root/'GPU_STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not torch.cuda.is_available():raise RuntimeError('Native CUDA unavailable')
        import subprocess
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
        if any(int(p)!=os.getpid() for p in pids):raise RuntimeError('Another GPU worker active')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write_json(root/'GPU_STATE.json',dict(status='DETACH_PREFLIGHT',pid=os.getpid(),time=time.time()))
        preflight(root,protocol,train)
        base.diagnostic(root,protocol,train)
        for epoch in range(1,6):
            if freeze(root,control)!=protocol:raise ValueError('Detached source changed')
            train=NativeMixtures(preparation,'train_pack',epoch)
            write_json(root/'GPU_STATE.json',dict(status='TRAINING',epoch=epoch,pid=os.getpid(),time=time.time()))
            base.train_epoch(root,epoch,protocol,train,val)
            comparison(root,control,epoch)
        if freeze(root,control)!=protocol:raise ValueError('Final source changed')
        if sha256(root/'gpu/BEST.pt')!=sha256(root/'gpu/SELECTED_005.pt'):raise ValueError('Selected copy mismatch')
        write_json(root/'GPU_COMPLETE.json',dict(status='COMPLETE',protocol_sha256=protocol,epochs=5,updates=375,
            time=time.time(),heldout_read=False))
        write_json(root/'GPU_STATE.json',dict(status='COMPLETED',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{14,15});os.nice(10)
    try:run(args.run.resolve(),args.control.resolve())
    except Exception:
        write_json(args.run/'GPU_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        write_json(args.run/'GPU_STATE.json',dict(status='FAILED',time=time.time()));raise
