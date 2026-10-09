"""Queued matched full U-Net loss fine-tuning, three epochs per arm.

Both arms start from the same selected U-Net e4, with identical fresh AdamW
states. The original raw waveform metrics select checkpoints in both arms.
The frozen ongoing experiments and reserved confirmation data are untouched.
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

import robust_nmse as loss_module
import fit_diagnostic as fit
import gpu_start_guard as guard
import watch_epochs as watch
from models import build, predict
from native_data import NativeMixtures
from study import atomic_torch

ARMS = loss_module.MODES
PARAMETERS = 32_142_859


def register(root, dependency, parent, cpu):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate registration; explicit --resume required')
    prior = watch.read(parent/'PROTOCOL.json')
    predecessor = watch.read(dependency/'PROTOCOL.json')
    check = watch.read(cpu)
    if (check['status'] != 'PASS' or check['trained'] or check['recorded_iq_reads']
            or len(check.get('full_input_checks', [])) != 2
            or any(r['parameters'] != PARAMETERS or not r['finite_full_network_gradients']
                   for r in check['full_input_checks'])):
        raise ValueError('Missing synthetic CPU check')
    if prior['preparation_sha256'] != predecessor['preparation_sha256']:
        raise ValueError('Mismatched data')
    checkpoint = parent/'unet_mean/SELECTED_004.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved['arm'] != 'unet_mean' or saved['best']['epoch'] != 4
            or saved['protocol_sha256'] != watch.digest(parent/'PROTOCOL.json')):
        raise ValueError('Wrong selected parent')
    if not all(torch.isfinite(x).all() for x in saved['model'].values()):
        raise ValueError('Nonfinite parent state')
    del saved
    sources = dict(predecessor['source_sha256'])
    for rel, digest in prior['source_sha256'].items():
        if rel in sources and sources[rel] != digest:
            raise ValueError('Conflicting parent sources')
        sources[rel] = digest
    for rel, digest in check['source_sha256'].items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('CPU-checked loss changed')
        sources[rel] = digest
    sources[str(Path(__file__).relative_to(watch.ROOT))] = watch.digest(Path(__file__))
    template = parent/'unet_mean/VALIDATION_004.json'
    watch.validate(template, None)
    protocol = dict(status='REGISTERED_QUEUED_MATCHED_UNET_LOSS_FINETUNE', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=watch.digest(dependency/'PROTOCOL.json'),
        parent_study=str(parent), parent_protocol_sha256=watch.digest(parent/'PROTOCOL.json'),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=watch.digest(checkpoint),
        parent_epoch=4, parent_updates=300, cpu_check=str(cpu), cpu_check_sha256=watch.digest(cpu),
        preparation=prior['preparation'], preparation_sha256=prior['preparation_sha256'],
        validation_identity_template=str(template), validation_identity_sha256=watch.digest(template),
        gpu_display_policy=predecessor['gpu_display_policy'], arms=list(ARMS), parameters=PARAMETERS,
        seed=0, epochs_per_arm=3, new_updates_per_arm=225, effective_batch=32, microbatch=1,
        examples_per_epoch=2400, validation_cases=630, train_schedule_epochs=[1, 2, 3],
        train_reuse='original TRAIN schedule1..3 already seen by parent; no new recordings claimed',
        initialization='same full selected e4 weights; fresh identical optimizer and seed0 for BOTH arms',
        optimizer='AdamW5e-4 wd1e-4 clip1 FP32 TF32off; same as parent fresh screen',
        change='replace selected-slot normalized error u by natural log1p(u), including inactive slots',
        unchanged='original waveform PIT matching, coherence, background loss, 0.1 countCE, data and architecture',
        evaluation='untransformed I/Q NMSE and complex SI-SDR; no oracle output rescaling',
        selection='per-arm minimum mean raw NMSE counts2/3 including e0',
        acceptance='both counts2/3 NMSE lower and SI-SDR higher vs same-budget control; weak metrics separately',
        ordering='original/log1p in odd epochs, reverse in even epochs',
        evidence='single extreme TRAIN batch head-gradient diagnostic; not established whole-model cause',
        claim='early matched development fine-tune, not new architecture or confirmed generalization',
        heldout_read=False, physical_aircraft_count=False, maximum_wait_seconds=24*3600,
        registered_at=time.time())
    for rel, digest in sources.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)
        target = root/'source_snapshot'/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, target)
    watch.write(root/'PROTOCOL.json', protocol)
    return protocol


def verify(root, protocol):
    for rel, digest in protocol['source_sha256'].items():
        if watch.digest(watch.ROOT/rel) != digest or watch.digest(root/'source_snapshot'/rel) != digest:
            raise ValueError('Frozen loss study source changed: '+rel)
    paths = [(Path(protocol['dependency'])/'PROTOCOL.json', 'dependency_protocol_sha256'),
             (Path(protocol['parent_study'])/'PROTOCOL.json', 'parent_protocol_sha256'),
             (Path(protocol['parent_checkpoint']), 'parent_checkpoint_sha256'),
             (Path(protocol['preparation'])/'PREPARATION.json', 'preparation_sha256'),
             (Path(protocol['cpu_check']), 'cpu_check_sha256'),
             (Path(protocol['validation_identity_template']), 'validation_identity_sha256')]
    for path, key in paths:
        if watch.digest(path) != protocol[key]:
            raise ValueError('Dependency changed: '+str(path))


def train_epoch(root, arm, epoch, protocol, train, validation, identities):
    folder = root/arm; folder.mkdir(exist_ok=True)
    digest = watch.digest(root/'PROTOCOL.json'); receipt = folder/f'EPOCH_{epoch:03d}.json'
    if receipt.exists():
        old = watch.read(receipt)
        if old['protocol_sha256'] != digest or old['updates'] != epoch*75 or old['arm'] != arm:
            raise ValueError('Completed receipt mismatch')
        watch.validate(folder/f'VALIDATION_{epoch:03d}.json', identities)
        return
    torch.manual_seed(0)
    net = build('unet_mean').cuda()
    if sum(p.numel() for p in net.parameters()) != PARAMETERS:
        raise ValueError('Full model capacity changed')
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4, foreach=False)
    last = folder/'LAST.pt'; updates = 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest or saved['epoch'] != epoch-1 or saved['arm'] != arm:
            raise ValueError('Nonconsecutive restart')
        net.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        updates, best = saved['updates'], saved['best']
        stored_best = torch.load(folder/'BEST.pt', map_location='cpu', weights_only=False)
        if stored_best['best'] != best or stored_best['protocol_sha256'] != digest:
            raise ValueError('Incomplete checkpoint transaction; explicit recovery needed')
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved, stored_best
    else:
        if epoch != 1:
            raise ValueError('Missing earlier epoch')
        saved = torch.load(protocol['parent_checkpoint'], map_location='cpu', weights_only=False)
        net.load_state_dict(saved['model']); del saved
        # Both arms really recompute e0; they do not inherit a possibly mismatched score.
        initial = fit.base.validate(net, validation, folder/'VALIDATION_000.json', 0)
        watch.validate(folder/'VALIDATION_000.json', identities)
        best = dict(epoch=0, metric=initial['selection_nmse'])
        atomic_torch(folder/'BEST.pt', dict(model=net.state_dict(), best=best, arm=arm, protocol_sha256=digest))
    if len(train) != 2400 or updates != (epoch-1)*75:
        raise ValueError('Training budget changed')
    net.train(); opt.zero_grad(set_to_none=True)
    started = time.time(); total = 0.; norms = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(len(train)):
        item = fit.base.batch([train[index]])
        estimates, logits = predict(net, item)
        result = loss_module.objective(estimates, item['references'], item['active'], item['mixture'], arm)
        loss = result['loss'] + .1*F.cross_entropy(logits, item['construction_count']-1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss/32).backward(); total += float(loss.detach())
        if (index+1)%32 == 0:
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            norms.append(float(norm)); opt.step(); opt.zero_grad(set_to_none=True); updates += 1
            if updates%5 == 0:
                watch.write(root/'STATE.json', dict(status='TRAINING', arm=arm, epoch=epoch, updates=updates,
                    examples=index+1, total=2400, seconds=time.time()-started, pid=os.getpid(), time=time.time()))
    seconds = time.time()-started
    watch.write(root/'STATE.json', dict(status='VALIDATING', arm=arm, epoch=epoch, updates=updates,
                pid=os.getpid(), time=time.time()))
    value = fit.base.validate(net, validation, folder/f'VALIDATION_{epoch:03d}.json', epoch)
    watch.validate(folder/f'VALIDATION_{epoch:03d}.json', identities)
    if value['selection_nmse'] < best['metric']:
        best = dict(epoch=epoch, metric=value['selection_nmse'])
        atomic_torch(folder/'BEST.pt', dict(model=net.state_dict(), best=best, arm=arm, protocol_sha256=digest))
    atomic_torch(folder/f'ACTUAL_{epoch:03d}.pt', dict(model=net.state_dict(), epoch=epoch, updates=updates,
                 arm=arm, protocol_sha256=digest))
    atomic_torch(last, dict(model=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch, updates=updates,
        best=best, arm=arm, protocol_sha256=digest, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    shutil.copyfile(folder/'BEST.pt', folder/f'SELECTED_{epoch:03d}.pt')
    result = dict(arm=arm, epoch=epoch, updates=updates, parent_updates=300, best=best, train_seconds=seconds,
        peak_bytes=torch.cuda.max_memory_allocated(), mean_training_loss=total/2400,
        preclip_gradient_norms=norms, clipped_fraction=float(np.mean(np.asarray(norms)>1)),
        validation={k:v for k,v in value.items() if k!='rows'}, protocol_sha256=digest)
    watch.write(receipt, result); print(json.dumps(result), flush=True)
    del net, opt, item, estimates, logits, loss, result
    gc.collect(); torch.cuda.empty_cache()


def publish_progress(root, public):
    events = []
    for arm in ARMS:
        for path in sorted((root/arm).glob('EPOCH_*.json')):
            events.append(watch.read(path))
    events.sort(key=lambda r:(r['epoch'], ARMS.index(r['arm']) if r['epoch']%2 else -ARMS.index(r['arm'])))
    common = min(sum(r['arm']==arm for r in events) for arm in ARMS)
    report = dict(status='COMPLETE' if common==3 else 'PARTIAL', common_epoch=common, events=events,
        protocol_sha256=watch.digest(root/'PROTOCOL.json'), parent_updates=300,
        snapshot_state=watch.read(root/'STATE.json'), heldout_read=False, time=time.time())
    watch.write(public.with_suffix('.json'), report)
    lines = ['# U-Net 손실 비교: 완료 epoch', '',
        '두 군 모두 같은 원 규모 U-Net e4/300업데이트에서 시작한다. 같은 TRAIN 일정 재사용·'
        '새 optimizer·각 추가3epoch/225업데이트다. 아래 epoch는 추가 미세조정 횟수다. '
        '선택은 두 군 모두 기존 raw NMSE이며 실패 조건도 유지한다.', '',
        '| 손실 | 추가 epoch | NMSE 2/3 ↓ | SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ | 선택 추가 epoch |',
        '|---|---:|---|---|---|---:|']
    for row in events:
        a,b = row['validation']['by_count'][1:]
        lines.append(f"|{row['arm']}|{row['epoch']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
            f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{row['best']['epoch']}|")
    lines += ['', f'공통 완료: {common} epoch. 학습 손실의 숫자 자체는 서로 다른 척도여서 직접 우열로 비교하지 않는다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))


def run(root, protocol, public):
    dependency = Path(protocol['dependency']); started = time.time()
    while not (dependency/'COMPLETE.json').exists():
        verify(root, protocol)
        if (dependency/'FAILURE.json').exists():
            raise RuntimeError('Previous fit diagnostic failed; review before GPU start')
        if time.time()-started > protocol['maximum_wait_seconds']:
            raise TimeoutError('Registered wait exceeded')
        watch.write(root/'STATE.json', dict(status='WAITING_OUTPUT_FIT', model_updates=0,
                    pid=os.getpid(), time=time.time()))
        time.sleep(30)
    completed = watch.read(dependency/'COMPLETE.json')
    if (completed['protocol_sha256'] != protocol['dependency_protocol_sha256']
            or len(completed['results']) != 2
            or any(r['updates'] != 256 or not r['finite_gradients'] for r in completed['results'])):
        raise ValueError('Incomplete predecessor')
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(root, protocol)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        watch.write(root/'GPU_START_CHECK.json', guard.wait_for_predecessor(
            watch.read(dependency/'STATE.json')['pid'], protocol['gpu_display_policy']))
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        _,identities = watch.validate(Path(protocol['validation_identity_template']), None)
        validation = NativeMixtures(protocol['preparation'], 'validation_pack', 1)
        for epoch in range(1, 4):
            train = NativeMixtures(protocol['preparation'], 'train_pack', epoch)
            for arm in (ARMS if epoch%2 else tuple(reversed(ARMS))):
                verify(root, protocol); train_epoch(root, arm, epoch, protocol, train, validation, identities)
                publish_progress(root, public)
            del train
        verify(root, protocol)
        for arm in ARMS:
            final = torch.load(root/arm/'LAST.pt', map_location='cpu', weights_only=False)
            if final['updates'] != 225 or final['epoch'] != 3:
                raise ValueError('Incomplete final optimizer budget')
            if not all(torch.isfinite(x).all() for x in final['model'].values()):
                raise ValueError('Nonfinite final parameters')
            steps = {int(s['step']) for s in final['optimizer']['state'].values() if 'step' in s}
            if steps != {225}:
                raise ValueError('Optimizer update counter mismatch')
            del final
        watch.write(root/'COMPLETE.json', dict(status='COMPLETE', epochs_per_arm=3, new_updates_per_arm=225,
            protocol_sha256=watch.digest(root/'PROTOCOL.json'), heldout_read=False, time=time.time()))
        watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
        publish_progress(root, public)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('run', 'dependency', 'parent', 'cpu-check', 'public'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    a=p.parse_args(); root=a.run.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.resume:
                protocol=watch.read(root/'PROTOCOL.json')
                if (protocol['dependency'] != str(a.dependency.resolve())
                        or protocol['parent_study'] != str(a.parent.resolve())
                        or protocol['cpu_check'] != str(a.cpu_check.resolve())):
                    raise ValueError('Resume dependencies changed')
                verify(root, protocol)
            else:
                protocol=register(root,a.dependency.resolve(),a.parent.resolve(),a.cpu_check.resolve())
            run(root,protocol,a.public.resolve())
        except Exception:
            watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
            watch.write(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
            raise
