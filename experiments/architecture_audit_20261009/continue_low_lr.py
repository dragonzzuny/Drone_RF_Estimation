"""Continue both audited low-LR arms through total fine-tune epochs 2 and 3.

The completed first epoch is imported with explicit provenance, not retrained.
Only checkpoint protocol metadata is changed in derived copies. Model tensors,
AdamW moments/steps and RNG states are retained exactly. The original study and
its frozen files remain untouched. Training uses its unmodified train_epoch.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

import low_lr_unet_comparison as prior
import watch_epochs as watch
from study import atomic_torch

CHECK = watch.ROOT/'reports/2026-10-09/LOW_LR_CONTINUATION_CPU_CHECK.json'


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def derive_checkpoint(source, destination, old_protocol, new_protocol):
    value = torch.load(source, map_location='cpu', weights_only=False)
    if value['protocol_sha256'] != old_protocol:
        raise ValueError('Imported checkpoint protocol differs')
    value['protocol_sha256'] = new_protocol
    value['imported_from'] = dict(sha256=watch.digest(source), protocol_sha256=old_protocol,
                                  epoch_already_trained=1, new_training_updates=0)
    atomic_torch(destination, value)
    restored = torch.load(destination, map_location='cpu', weights_only=False)
    restored.pop('imported_from')
    restored['protocol_sha256'] = old_protocol
    original = torch.load(source, map_location='cpu', weights_only=False)
    if not identical(restored, original):
        raise ValueError('Checkpoint state changed while importing')


def register(root, dependency):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate continuation; use --resume')
    old = watch.read(dependency/'PROTOCOL.json')
    check = watch.read(CHECK)
    if (check['status'] != 'PASS' or check['worker_sha256'] != watch.digest(Path(__file__))
            or check['rf_training_updates'] != 0 or check['recorded_iq_reads'] != 0):
        raise ValueError('Missing continuation state check')
    if old['epochs_per_arm'] != 1 or old['learning_rate'] != 1e-4:
        raise ValueError('Expected the matched one-epoch low-LR study')
    plan = dict(old)
    sources = dict(old['source_sha256'])
    for path in (Path(__file__), Path(__file__).with_name('check_low_lr_continuation.py'),
                 Path(__file__).with_name('audit_low_lr_continuation.py')):
        sources[str(path.relative_to(watch.ROOT))] = watch.digest(path)
    plan.update(status='REGISTERED_LOW_LR_CONTINUATION', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=watch.digest(dependency/'PROTOCOL.json'),
        epochs_per_arm=3, new_updates_per_arm=225, train_schedule_epochs=[1, 2, 3],
        imported_epochs_per_arm=1, imported_updates_per_arm=75,
        executed_epochs=[2, 3], executed_updates_per_arm=150, learning_rate=1e-4,
        initialization='exact e1 model, optimizer and RNG continuation in both arms; no restart',
        budget_definition='225 total fine-tuning updates since U-Net e4: 75 imported plus 150 executed here',
        decision_basis='one-epoch original arm reduced count3 NMSE but worsened SI; extend BOTH arms to reference total budget',
        selection='minimum mean raw NMSE counts2/3 across original parent and total fine-tune epochs1/2/3',
        limitation='adaptive development extension, decided after original e1; not independent confirmation or convergence proof',
        dependency_audit=str(watch.ROOT/'reports/2026-10-09/LOW_LR_UNET_FINAL_AUDIT.json'),
        continuation_check_sha256=watch.digest(CHECK), registered_at=time.time())
    for rel, digest in sources.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Changed inherited source')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, target)
    watch.write(root/'PROTOCOL.json', plan)
    return plan


def verify(root, plan):
    for rel, expected in plan['source_sha256'].items():
        if watch.digest(watch.ROOT/rel) != expected or watch.digest(root/'source_snapshot'/rel) != expected:
            raise ValueError('Frozen source changed: '+rel)
    for path, expected in ((Path(plan['dependency'])/'PROTOCOL.json', plan['dependency_protocol_sha256']),
                           (Path(plan['preparation'])/'PREPARATION.json', plan['preparation_sha256']),
                           (Path(plan['parent_checkpoint']), plan['parent_checkpoint_sha256']),
                           (CHECK, plan['continuation_check_sha256'])):
        if watch.digest(path) != expected:
            raise ValueError('Continuation dependency changed')


def bootstrap(root, plan):
    """Create audited derived copies; preserve every original artifact."""
    if (root/'IMPORT.json').exists():
        saved = watch.read(root/'IMPORT.json')
        for rel, expected in saved['source_sha256'].items():
            if watch.digest(Path(plan['dependency'])/rel) != expected:
                raise ValueError('Imported source changed')
        return
    old = Path(plan['dependency']); digest = watch.digest(root/'PROTOCOL.json')
    hashes, imported = {}, {}
    for arm in plan['arms']:
        folder = root/arm; folder.mkdir(exist_ok=True)
        for name in ('VALIDATION_000.json', 'VALIDATION_001.json'):
            source = old/arm/name
            shutil.copyfile(source, folder/name)
            hashes[f'{arm}/{name}'] = watch.digest(source)
        for name in ('LAST.pt', 'BEST.pt', 'ACTUAL_001.pt', 'SELECTED_001.pt'):
            source = old/arm/name
            derive_checkpoint(source, folder/name, plan['dependency_protocol_sha256'], digest)
            hashes[f'{arm}/{name}'] = watch.digest(source)
            imported[f'{arm}/{name}'] = watch.digest(folder/name)
        shutil.copyfile(folder/'LAST.pt', folder/'RESUME_001.pt')
        imported[f'{arm}/RESUME_001.pt'] = watch.digest(folder/'RESUME_001.pt')
        receipt = watch.read(old/arm/'EPOCH_001.json')
        if receipt['protocol_sha256'] != plan['dependency_protocol_sha256'] or receipt['updates'] != 75:
            raise ValueError('Imported receipt differs')
        hashes[f'{arm}/EPOCH_001.json'] = watch.digest(old/arm/'EPOCH_001.json')
        receipt.update(protocol_sha256=digest, imported_epoch=True, new_training_updates=0,
            imported_receipt_sha256=hashes[f'{arm}/EPOCH_001.json'],
            imported_protocol_sha256=plan['dependency_protocol_sha256'])
        watch.write(folder/'EPOCH_001.json', receipt)
    watch.write(root/'IMPORT.json', dict(status='PASS', source_sha256=hashes,
        derived_checkpoint_sha256=imported, exact_state_roundtrip_checked=True,
        imported_updates_per_arm=75, new_training_updates=0,
        dependency_audit_sha256=watch.digest(Path(plan['dependency_audit'])), time=time.time()))


def progress(root, public):
    plan = watch.read(root/'PROTOCOL.json'); events = []
    for epoch in (1, 2, 3):
        for arm in (plan['arms'] if epoch%2 else list(reversed(plan['arms']))):
            path = root/arm/f'EPOCH_{epoch:03d}.json'
            if path.exists(): events.append(watch.read(path))
    common = min(sum(r['arm']==arm for r in events) for arm in plan['arms'])
    watch.write(public.with_suffix('.json'), dict(status='COMPLETE' if common==3 else 'PARTIAL',
        common_epoch=common, events=events, epoch1_imported=True,
        protocol_sha256=watch.digest(root/'PROTOCOL.json'), heldout_read=False, time=time.time()))
    lines = ['# 낮은 학습률 U-Net: 두 군의 3 epoch 연속 비교', '',
        '같은 RFUAV 분할·원 RF 대역·native100MS/s·seed0·32,142,859파라미터. '
        '부모e4 이후 첫75업데이트는 앞선 실험에서 가져왔고, optimizer/RNG를 그대로 이어 각각150업데이트를 추가한다. '
        'e1은 재학습하지 않았다. 각epoch2400학습·630개발검증이며 보류 기록은 미개봉이다.', '',
        '| 손실 | 누적 추가 epoch | 상태 | NMSE 2/3 ↓ | SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ | 선택 epoch |',
        '|---|---:|---|---|---|---|---:|']
    for row in events:
        a,b = row['validation']['by_count'][1:]
        lines.append(f"|{row['arm']}|{row['epoch']}|{'이전 완료분' if row['epoch']==1 else '이번 추가학습'}|"
            f"{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|"
            f"{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{row['best']['epoch']}|")
    lines += ['', f'두 군의 공통 완료: 누적 추가{common}epoch. 선택은e0 포함raw NMSE이며, 네 파형 지표의 동시 개선과 별개다.',
        '기존 결과를 보고 결정한 개발 연장 실험이다. 충분한 수렴·미학습 기종 일반화·실제 드론 대수의 검증이 아니다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))


def run(root, plan, public):
    dependency = Path(plan['dependency']); audit_path = Path(plan['dependency_audit'])
    began = time.time()
    while not (dependency/'COMPLETE.json').exists() or not audit_path.exists():
        verify(root, plan)
        if (dependency/'FAILURE.json').exists(): raise RuntimeError('First epoch study failed')
        if time.time()-began > 24*3600: raise TimeoutError('Dependency wait exceeded')
        watch.write(root/'STATE.json',dict(status='WAITING_AUDITED_FIRST_EPOCH',pid=os.getpid(),time=time.time()))
        time.sleep(30)
    audit = watch.read(audit_path)
    if (audit['status'] != 'PASS' or audit['study_protocol_sha256'] != plan['dependency_protocol_sha256']
            or audit['complete_sha256'] != watch.digest(dependency/'COMPLETE.json')):
        raise ValueError('First epoch completion audit differs')
    with prior.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(root, plan)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        watch.write(root/'GPU_START_CHECK.json',prior.guard.wait_for_predecessor(
            watch.read(dependency/'STATE.json')['pid'],plan['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        bootstrap(root, plan); progress(root, public)
        _, identities = watch.validate(Path(plan['validation_identity_template']),None)
        validation = prior.NativeMixtures(plan['preparation'],'validation_pack',1)
        for epoch in (2,3):
            train = prior.NativeMixtures(plan['preparation'],'train_pack',epoch)
            for arm in (plan['arms'] if epoch%2 else list(reversed(plan['arms']))):
                verify(root, plan)
                prior.train_epoch(root,arm,epoch,plan,train,validation,identities)
                progress(root, public)
            del train
        verify(root,plan)
        for arm in plan['arms']:
            saved=torch.load(root/arm/'LAST.pt',map_location='cpu',weights_only=False)
            if saved['epoch']!=3 or saved['updates']!=225:raise ValueError('Final continuation budget differs')
            if {int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s}!={225}:
                raise ValueError('Optimizer was reset')
            del saved
        watch.write(root/'COMPLETE.json',dict(status='COMPLETE',epochs_per_arm=3,new_updates_per_arm=225,
            imported_updates_per_arm=75,executed_updates_per_arm=150,protocol_sha256=watch.digest(root/'PROTOCOL.json'),
            heldout_read=False,time=time.time()))
        watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
        progress(root,public)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','dependency','public'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=watch.read(root/'PROTOCOL.json') if a.resume else register(root,a.dependency.resolve())
            if plan['dependency']!=str(a.dependency.resolve()):raise ValueError('Changed resume dependency')
            run(root,plan,a.public.resolve())
        except Exception:
            watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
            watch.write(root/'STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()));raise
