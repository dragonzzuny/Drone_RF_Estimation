"""Conditional full-capacity long-complex-context comparison, 5 epochs each.

Registered before the queued GPU TRAIN4 preflight finishes; start only if both
arms pass numerical/fit checks. Fresh initial weights, same native data and
updates. No modification of the already running architecture comparison.
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

import torch
from torch.nn import functional as F

import phase_packing as pp
import phase_data as data
import phase_evaluation as evaluation
import phase_fit_v2 as preflight
import gpu_start_guard as gpu_guard
import watch_epochs as watch
from native_data import NativeMixtures, sha256, write_json
from drone_rf.losses import pit_waveform_loss
from study import atomic_torch

ROOT = preflight.ROOT
ARMS = ('local', 'long')


def register(root, prerequisite, cpu_check):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate registration; use explicit --resume for the same frozen run')
    root.mkdir(parents=True, exist_ok=True)
    prior = watch.read(prerequisite/'PROTOCOL.json')
    check = watch.read(cpu_check)
    if check['status'] != 'PASS' or check['trained'] or check['heldout_read']:
        raise ValueError('Invalid CPU pipeline check')
    if (check['preparation_sha256'] != prior['preparation_sha256']
            or len(check['checked_train_examples']) != 45
            or not check['full_mixture_fine_crop_exact'] or not check['observed_forward_whitelist']):
        raise ValueError('CPU check used another dataset or incomplete coverage')
    source = dict(prior['source_sha256'])
    modules = [Path(__file__), Path(data.__file__), Path(evaluation.__file__), Path(watch.__file__),
               Path(__file__).with_name('check_phase_pipeline.py')]
    for path in modules:
        source[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in check['source_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('CPU-checked pipeline changed')
    template = Path(prior['predecessor'])/'unet_mean/VALIDATION_000.json'
    watch.validate(template, None)
    plan = dict(status='REGISTERED_CONDITIONAL_PHASE_CONTEXT_COMPARISON',
        source_sha256=source, prerequisite=str(prerequisite),
        prerequisite_protocol_sha256=sha256(prerequisite/'PROTOCOL.json'),
        preparation=prior['preparation'], preparation_sha256=prior['preparation_sha256'],
        cpu_check=str(cpu_check), cpu_check_sha256=sha256(cpu_check),
        validation_identity_template=str(template), validation_identity_sha256=sha256(template),
        gpu_display_policy=prior['gpu_display_policy'],
        operational_revision='v2: pin and admit existing RustDesk display context; all research rules unchanged',
        arms={arm: pp.PARAMETERS for arm in ARMS}, seed=0, epochs_per_arm=5, updates_per_arm=375,
        examples_per_epoch=2400, validation_cases=630, effective_batch=32, microbatch=1,
        optimizer='AdamW lr5e-4 wd1e-4 clip1; float32, TF32 disabled',
        loss='PIT NMSE+coherence+inactive/background plus0.1 construction-count CE',
        initialization='fresh seed0, identical parameter tensors; TRAIN4 preflight weights discarded',
        long_samples=pp.SAMPLES, fine_samples=pp.FINE, sample_rate_hz=100_000_000,
        change='out-of-crop complex samples visible only to long arm; same original full RMS and mean power context',
        ordering='round-robin epochs; local/long order reversed in even epochs',
        selection='per-arm min mean NMSE counts2/3 including e0',
        acceptance='NMSE lower and complex SI-SDR higher in both counts2/3; inspect weak/count separately',
        start_condition='both full-input TRAIN4 arms have finite gradients and lower final than initial NMSE',
        comparison='same capacity/data/update budget; early development screen, not convergence or independent test',
        heldout_read=False, physical_aircraft_count=False, registered_at=time.time())
    for rel, digest in source.items():
        if sha256(ROOT/rel) != digest:
            raise ValueError('Frozen source changed')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    write_json(root/'PROTOCOL.json', plan)
    return plan


def verify(root, plan):
    for rel, digest in plan['source_sha256'].items():
        if sha256(ROOT/rel) != digest or sha256(root/'source_snapshot'/rel) != digest:
            raise ValueError('Frozen comparison source changed: '+rel)
    for path, expected in ((Path(plan['prerequisite'])/'PROTOCOL.json',plan['prerequisite_protocol_sha256']),
                          (Path(plan['preparation'])/'PREPARATION.json',plan['preparation_sha256']),
                          (Path(plan['cpu_check']),plan['cpu_check_sha256']),
                          (Path(plan['validation_identity_template']),plan['validation_identity_sha256'])):
        if sha256(path) != expected:
            raise ValueError('Comparison dependency changed: '+str(path))


def train_epoch(root, arm, epoch, plan, train, validation, identities):
    folder = root/arm
    folder.mkdir(exist_ok=True)
    digest = sha256(root/'PROTOCOL.json')
    receipt = folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        old = watch.read(receipt)
        if old['protocol_sha256'] != digest or old['updates'] != epoch*75 or old['arm'] != arm:
            raise ValueError('Invalid completed epoch')
        watch.validate(folder/f'VALIDATION_{epoch:03d}.json', identities)
        if not (folder/f'ACTUAL_{epoch:03d}.pt').exists() or not (folder/f'SELECTED_{epoch:03d}.pt').exists():
            raise ValueError('Missing completed checkpoint')
        return
    torch.manual_seed(0)
    net = pp.PhasePackedWaveNet(arm).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4, foreach=False)
    last = folder/'LAST.pt'
    updates = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest or saved['epoch'] != epoch-1 or saved['arm'] != arm:
            raise ValueError('Nonconsecutive or mismatched resume state')
        net.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        updates, best = saved['updates'], saved['best']
        stored_best = torch.load(folder/'BEST.pt', map_location='cpu', weights_only=False)
        if stored_best['best'] != best or stored_best['protocol_sha256'] != digest:
            raise ValueError('Incomplete checkpoint transaction; explicit recovery required')
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved, stored_best
    else:
        if epoch != 1:
            raise ValueError('Missing previous epoch')
        value = evaluation.evaluate(net, validation, folder/'VALIDATION_000.json', 0, identities)
        best = dict(epoch=0, metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=digest,arm=arm))
    if len(train) != 2400 or updates != (epoch-1)*75:
        raise ValueError('Training budget changed')
    net.train(); opt.zero_grad(set_to_none=True)
    began, total = time.time(), 0.
    torch.cuda.reset_peak_memory_stats()
    for index in range(len(train)):
        item = data.batch(data.example(train,index),'cuda')
        predicted = data.predict(net,item)
        loss = pit_waveform_loss(predicted['estimates'], item['references'], item['active'], item['mixture'])['loss']
        loss = loss+.1*F.cross_entropy(predicted['count_logits'],item['construction_count']-1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss/32).backward(); total += float(loss.detach())
        if (index+1)%32 == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True); updates += 1
            if updates%5 == 0:
                write_json(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=epoch,updates=updates,
                    examples=index+1,total=2400,seconds=time.time()-began,pid=os.getpid(),time=time.time()))
    seconds, peak = time.time()-began, torch.cuda.max_memory_allocated()
    value = evaluation.evaluate(net,validation,folder/f'VALIDATION_{epoch:03d}.json',epoch,identities)
    if value['selection_nmse'] < best['metric']:
        best = dict(epoch=epoch,metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=digest,arm=arm))
    atomic_torch(folder/f'ACTUAL_{epoch:03d}.pt',dict(model=net.state_dict(),epoch=epoch,
        updates=updates,protocol_sha256=digest,arm=arm))
    atomic_torch(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=updates,
        best=best,protocol_sha256=digest,arm=arm,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    shutil.copyfile(folder/'BEST.pt',folder/f'SELECTED_{epoch:03d}.pt')
    result = dict(arm=arm,epoch=epoch,updates=updates,best=best,train_seconds=seconds,peak_bytes=peak,
        mean_training_loss=total/2400,validation={k:v for k,v in value.items() if k!='rows'},protocol_sha256=digest)
    write_json(receipt,result)
    print(json.dumps(result),flush=True)
    del net,opt,item,predicted,loss
    gc.collect(); torch.cuda.empty_cache()


def progress_report(root, plan):
    records, histories = [], {}
    for arm in ARMS:
        history = []
        for epoch in range(6):
            if epoch and not (root/arm/f'EPOCH_{epoch:03d}.json').exists():
                break
            path = root/arm/f'VALIDATION_{epoch:03d}.json'
            if not path.exists():
                break
            value,_ = watch.validate(path,None)
            history.append(value)
            if epoch:
                records.append(watch.read(root/arm/f'EPOCH_{epoch:03d}.json'))
        histories[arm] = history
    common = min(len(h)-1 for h in histories.values())
    selected = {arm:min(history[:common+1],key=lambda x:x['selection_nmse'])
                for arm,history in histories.items()} if common>0 else {}
    write_json(root/'REPORT.json',dict(common_epoch=max(common,0),events=records,
        common_selected=selected,time=time.time(),heldout_read=False))
    lines = ['# 긴 복소문맥 비교: 완료 epoch', '',
        'RFUAV 같은 대역·원 수신 중심 간격 보존. 두 군 모두4,641,795파라미터, '
        '같은 초기값·자료·각5epoch/375업데이트. local도 전체RMS와 평균 전력 특징을 받는다.', '',
        '| 군 | epoch | NMSE 2/3 ↓ | SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ | 선택 epoch |',
        '|---|---:|---|---|---|---:|']
    for record in sorted(records,key=lambda x:(x['epoch'],x['arm'])):
        a,b = record['validation']['by_count'][1:]
        f=lambda x:'미정의' if x is None else f'{x:.4f}'
        lines.append(f"| {record['arm']} | {record['epoch']} | {f(a['mean_nmse'])}/{f(b['mean_nmse'])} | "
            f"{f(a['mean_si_sdr'])}/{f(b['mean_si_sdr'])} | {f(a['weakest_nmse'])}/{f(b['weakest_nmse'])} | {record['best']['epoch']} |")
    lines += ['',f'공통 완료: {max(common,0)}epoch. 개발 검증630개, 보류 기종 미개봉. '
              '초기 제한 예산 비교이며 완전 수렴·실측 동시 수신·물리 드론 대수를 입증하지 않는다.', '']
    watch.write(root/'REPORT_KO.md','\n'.join(lines))
    return histories


def run(root,plan):
    prerequisite=Path(plan['prerequisite'])
    while not (prerequisite/'COMPLETE.json').exists():
        if (prerequisite/'FAILURE.json').exists():
            raise RuntimeError('Preflight failed; full training not started')
        verify(root,plan)
        write_json(root/'STATE.json',dict(status='WAITING_GPU_PREFLIGHT',pid=os.getpid(),time=time.time()))
        time.sleep(30)
    result=watch.read(prerequisite/'COMPLETE.json')
    if result['protocol_sha256']!=plan['prerequisite_protocol_sha256']:
        raise ValueError('Preflight receipt mismatch')
    if not all(result['results'][arm]['improved'] and result['results'][arm]['finite_gradients']
               and result['results'][arm]['parameters']==pp.PARAMETERS for arm in ARMS):
        write_json(root/'STATE.json',dict(status='NOT_STARTED_PREFLIGHT_REQUIRES_REVIEW',pid=os.getpid(),time=time.time()))
        return
    for arm in ARMS:
        fitted=result['results'][arm]
        if (fitted['history'][-1]['step']!=32 or not fitted['weights_discarded']
                or not fitted['final_nmse']<fitted['initial']['mean_nmse']):
            raise ValueError('Preflight numerical evidence does not support the start gate')
    write_json(root/'START_GATE.json',dict(prerequisite_complete_sha256=sha256(prerequisite/'COMPLETE.json'),
        checked_at=time.time(),passed=True,preflight_weights_reused=False))
    with preflight.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify(root,plan)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        previous_pid=watch.read(prerequisite/'STATE.json')['pid']
        write_json(root/'GPU_START_CHECK.json',gpu_guard.wait_for_predecessor(previous_pid,plan['gpu_display_policy']))
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        _,identities=watch.validate(Path(plan['validation_identity_template']),None)
        validation=NativeMixtures(plan['preparation'],'validation_pack',1)
        for epoch in range(1,6):
            train=NativeMixtures(plan['preparation'],'train_pack',epoch)
            for arm in (ARMS if epoch%2 else tuple(reversed(ARMS))):
                verify(root,plan)
                train_epoch(root,arm,epoch,plan,train,validation,identities)
                progress_report(root,plan)
            del train
        verify(root,plan)
        histories=progress_report(root,plan)
        if any(len(h)!=6 for h in histories.values()):
            raise ValueError('Incomplete final evaluation history')
        write_json(root/'COMPLETE.json',dict(status='COMPLETE',epochs_per_arm=5,updates_per_arm=375,
            protocol_sha256=sha256(root/'PROTOCOL.json'),heldout_read=False,time=time.time()))
        write_json(root/'STATE.json',dict(status='COMPLETED',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','prerequisite','cpu-check'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); root=args.run.resolve(); root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.resume:
                plan=watch.read(root/'PROTOCOL.json')
                if plan['prerequisite']!=str(args.prerequisite.resolve()) or plan['cpu_check']!=str(args.cpu_check.resolve()):
                    raise ValueError('Resume dependency mismatch')
                verify(root,plan)
            else:
                plan=register(root,args.prerequisite.resolve(),args.cpu_check.resolve())
            run(root,plan)
        except Exception:
            write_json(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
            write_json(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
            raise
