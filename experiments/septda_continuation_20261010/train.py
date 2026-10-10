"""Matched continued learning; one block is 2,400 mixtures / 75 updates."""
import argparse
import fcntl
import gc
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/septda_rf_20261010'))
spec = importlib.util.spec_from_file_location('septda_parent_for_continuation', ROOT / 'experiments/septda_rf_20261010/train.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
w = s.w
PUBLIC = ROOT / 'reports/2026-10-10'
PLAN = PUBLIC / 'SEPTDA_RF_CONTINUATION_PLAN_KO.md'
MILESTONES = (5, 10, 20, 30, 40, 50)
ORIGINS = {'control': ROOT / 'local/fresh_schedule_20261010_v1',
           'septda': ROOT / 'local/septda_rf_20261010_v1'}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def restore(arm, path, device, expected_epoch):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['epoch'] == expected_epoch and saved['updates'] == 75 * expected_epoch
    net = s.make_net() if arm == 'septda' else s.worker.make_model('retained_unet')
    net.load_state_dict(saved['model']); net.to(device)
    assert sum(p.numel() for p in net.parameters()) == (54596107 if arm == 'septda' else 32142859)
    if arm == 'septda':
        groups = [dict(params=[p for k, p in net.named_parameters() if not k.startswith('septda.')], lr=1e-5),
                  dict(params=list(net.septda.parameters()), lr=1e-4)]
    else:
        groups = [dict(params=list(net.parameters()), lr=1e-5)]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4, foreach=False)
    opt.load_state_dict(saved['optimizer'])
    assert equal(net.state_dict(), saved['model']) and equal(opt.state_dict(), saved['optimizer'])
    assert len(opt.state) == len(list(net.parameters()))
    assert {int(v['step']) for v in opt.state.values()} == {75 * expected_epoch}
    torch.set_rng_state(saved['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        assert equal(torch.cuda.get_rng_state_all(), saved['cuda_rng'])
    assert torch.equal(torch.get_rng_state(), saved['torch_rng'])
    best, monitor = saved['best'], saved.get('lr_monitor', dict(best=None, bad=0, reductions=0))
    provenance = dict(checkpoint_sha256=w.digest(path), source_protocol_sha256=saved['protocol_sha256'],
                      exact_model_optimizer_rng_restore=True, previous_updates=saved['updates'])
    del saved
    return net, opt, best, monitor, provenance


def register(root):
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Already registered; use --resume')
    original = w.read(ORIGINS['septda'] / 'PROTOCOL.json'); s.verify(ORIGINS['septda'], original)
    for name in ('SEPTDA_RF_AUDIT.json', 'FRESH_SCHEDULE_AUDIT.json'):
        assert w.read(PUBLIC / name)['status'] == 'PASS'
    check = w.read(PUBLIC / 'SEPTDA_RF_CONTINUATION_CHECK.json')
    assert check['status'] == 'PASS' and check['worker_sha256'] == w.digest(Path(__file__))
    sources = dict(original['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = w.digest(Path(__file__))
    sources[str(Path(__file__).with_name('check.py').relative_to(ROOT))] = w.digest(Path(__file__).with_name('check.py'))
    pins = {str(f): w.digest(f) for f in [PLAN, PUBLIC / 'SEPTDA_RF_CONTINUATION_CHECK.json',
        PUBLIC / 'SEPTDA_RF_AUDIT.json', PUBLIC / 'FRESH_SCHEDULE_AUDIT.json']}
    for arm, folder in ORIGINS.items():
        for name in ('PROTOCOL.json', 'LAST.pt', 'BEST.pt', 'VALIDATION_000.json', 'VALIDATION_001.json', 'EPOCH_001.json'):
            pins[str(folder / name)] = w.digest(folder / name)
    for rel, sha in sources.items():
        assert w.digest(ROOT / rel) == sha
        dst = root / 'source_snapshot' / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    p = dict(status='REGISTERED_MATCHED_LONG_CONTINUATION', source_sha256=sources, pinned_files=pins,
        original_protocol=original, origins={k: str(v) for k, v in ORIGINS.items()},
        max_epochs=50, examples_per_epoch=2400, updates_per_epoch=75, imported_epochs=1,
        schedule='((epoch+1)%5)+1; repeat fixed five sealed schedules, not fresh recordings',
        milestones=list(MILESTONES), lr_rule='constant through20; then halve after10 nonimprovements of >=1e-5 mean NMSE2/3',
        early_stopping=False, seed=0, effective_batch=32, microbatch=1, heldout_read=False,
        retention='LAST/BEST and milestone weights; all metrics, hashes and epoch receipts retained',
        independent_test=False, registered_at=time.time())
    w.write(root / 'PROTOCOL.json', p)
    for arm, folder in ORIGINS.items():
        target = root / arm; target.mkdir(exist_ok=True)
        for name in ('VALIDATION_000.json', 'VALIDATION_001.json'):
            shutil.copyfile(folder / name, target / name)
        event = dict(w.read(folder / 'EPOCH_001.json'), arm=arm, imported=True,
                     original_event_sha256=w.digest(folder / 'EPOCH_001.json'))
        w.write(target / 'EPOCH_001.json', event)
        w.write(target / 'BEST_ORIGIN.json', dict(path=str(folder / 'BEST.pt'), sha256=w.digest(folder / 'BEST.pt')))
    return p


def verify(root, p):
    for rel, sha in p['source_sha256'].items():
        assert w.digest(ROOT / rel) == w.digest(root / 'source_snapshot' / rel) == sha, rel
    for name, sha in p['pinned_files'].items():
        assert w.digest(Path(name)) == sha, name
    s.verify(Path(p['origins']['septda']), p['original_protocol'])


def progress(root, p):
    events = []
    for epoch in range(1, p['max_epochs'] + 1):
        for arm in ('control', 'septda'):
            f = root / arm / f'EPOCH_{epoch:03d}.json'
            if f.exists():
                v = w.read(f)
                events.append({k: v[k] for k in ('arm', 'epoch', 'updates', 'validation', 'best', 'mean_training_loss', 'train_seconds')})
    counts = {arm: sum(r['arm'] == arm for r in events) for arm in ('control', 'septda')}
    out = dict(status='COMPLETE' if min(counts.values()) == p['max_epochs'] else 'PARTIAL',
               completed_by_arm=counts, common_epoch=min(counts.values()), events=events,
               protocol_sha256=w.digest(root / 'PROTOCOL.json'), heldout_read=False, time=time.time())
    w.write(PUBLIC / 'SEPTDA_RF_CONTINUATION_PROGRESS.json', out)
    lines = ['# SepTDA 참고 RF 모델의 연속 학습', '',
        '같은 RFUAV 원기록 분할·원RF대역·native100MS/s·seed0. 실제 e1 가중치와 optimizer/RNG에서 이어간다. '
        '한 구간은2,400혼합·75업데이트이며, 기존5개 일정의12,000혼합을 반복한다. '
        '총50구간은 자원 상한이고 수렴 보장이 아니다. 매번 새 원기록이나 새 합성을 읽는 것이 아니다.', '',
        '| 모델 | 누적 구간 | 누적 업데이트 | 훈련 손실 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ | 최약NMSE2/3 ↓ | 선택 |',
        '|---|---:|---:|---:|---|---|---|---:|']
    for e in events:
        a, b = e['validation']['by_count'][1:]
        lines.append(f"|{e['arm']}|{e['epoch']}|{e['updates']}|{e['mean_training_loss']:.6f}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{w.fmt(a['mean_si_sdr'],3)}/{w.fmt(b['mean_si_sdr'],3)}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{e['best']['epoch']}|")
    lines += ['', f"두 군의 공통 완료 구간: {out['common_epoch']}.",
        '같은 DEV의 반복 선택이다. e1은 이미 완료한 결과이며 재학습하지 않았다. 구조 계열의 실패·수렴 완료를 초기 구간만으로 판단하지 않는다.', '',
        '[선행의 학습량](TRAINING_BUDGET_LITERATURE_KO.md) · [규약](SEPTDA_RF_CONTINUATION_PLAN_KO.md)']
    w.write(PUBLIC / 'SEPTDA_RF_CONTINUATION_PROGRESS_KO.md', '\n'.join(lines) + '\n')


def epoch_run(root, p, arm, epoch, train, dev, ids):
    folder = root / arm; ph = w.digest(root / 'PROTOCOL.json')
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise RuntimeError('Less than3GiB available for atomic checkpoint saving')
    source = folder / 'LAST.pt' if epoch > 2 else Path(p['origins'][arm]) / 'LAST.pt'
    resume_input = folder / 'RESUME_INPUT.pt'
    if epoch > 2:
        probe = torch.load(source, map_location='cpu', weights_only=False)
        recorded_epoch = probe['epoch']; del probe
        if recorded_epoch == epoch:
            # Interrupted after saving LAST but before the completion receipt.
            source = resume_input
        else:
            assert recorded_epoch == epoch - 1
    if source != resume_input:
        if resume_input.exists(): resume_input.unlink()
        os.link(source, resume_input)
    source = resume_input
    net, opt, best, monitor, provenance = restore(arm, source, 'cuda', epoch - 1)
    expected_protocol = ph if epoch > 2 else w.digest(Path(p['origins'][arm]) / 'PROTOCOL.json')
    assert provenance['source_protocol_sha256'] == expected_protocol
    counts = [0, 0, 0]; loss_sum = 0.; norms = []; started = time.time()
    net.train(); opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    for index in range(len(train)):
        item = s.worker.fit.base.batch([train[index]])
        output, logits = s.checkpoint_predict(net, item, s.worker.predict)
        loss = s.worker.pit_waveform_loss(output, item['references'], item['active'], item['mixture'])['loss']
        loss = loss + .1 * torch.nn.functional.cross_entropy(logits, item['construction_count'] - 1)
        assert torch.isfinite(loss); (loss / 32).backward(); loss_sum += float(loss.detach())
        counts[int(item['construction_count']) - 1] += 1
        if (index + 1) % 32 == 0:
            assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in net.parameters())
            norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)))
            opt.step(); opt.zero_grad(set_to_none=True)
            w.write(root / 'STATE.json', dict(status='TRAINING', arm=arm, epoch=epoch, schedule_epoch=train.epoch,
                updates_in_epoch=len(norms), cumulative_updates=75*(epoch-1)+len(norms), examples=index+1,
                mean_training_loss_so_far=loss_sum/(index+1), seconds=time.time()-started, pid=os.getpid(), time=time.time()))
    elapsed = time.time() - started
    assert len(norms) == 75 and counts == [800, 800, 800]
    assert {int(v['step']) for v in opt.state.values()} == {75 * epoch}
    actual_path = folder / 'LATEST_ACTUAL.pt'
    s.worker.atomic_torch(actual_path, dict(model=net.state_dict(), epoch=epoch, updates=75*epoch, protocol_sha256=ph))
    actual_sha = w.digest(actual_path)
    w.write(root / 'STATE.json', dict(status='VALIDATING', arm=arm, epoch=epoch, updates=75*epoch, pid=os.getpid(), time=time.time()))
    vp = folder / f'VALIDATION_{epoch:03d}.json'
    s.t.validate(net, dev, vp, epoch, s.worker.predict, s.worker)
    actual, _ = w.validate(vp, ids)
    if actual['selection_nmse'] < best['metric']:
        best = dict(epoch=epoch, metric=actual['selection_nmse'])
        s.worker.atomic_torch(folder / 'BEST.pt', dict(model=net.state_dict(), best=best, protocol_sha256=ph))
    used_lr = [g['lr'] for g in opt.param_groups]
    if epoch >= 20:
        value = actual['selection_nmse']
        if monitor['best'] is None or value < monitor['best'] - 1e-5:
            monitor.update(best=value, bad=0)
        else:
            monitor['bad'] += 1
            if monitor['bad'] >= 10:
                for g in opt.param_groups: g['lr'] *= .5
                monitor['bad'] = 0; monitor['reductions'] += 1
    s.worker.atomic_torch(folder / 'LAST.pt', dict(model=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
        updates=75*epoch, best=best, lr_monitor=monitor, protocol_sha256=ph,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    saved = torch.load(folder / 'LAST.pt', map_location='cpu', weights_only=False)
    assert equal(saved['model'], net.state_dict()) and equal(saved['optimizer'], opt.state_dict())
    assert saved['epoch'] == epoch and saved['updates'] == 75*epoch
    del saved
    if epoch in MILESTONES:
        destination = folder / f'ACTUAL_{epoch:03d}.pt'
        if destination.exists():
            destination.rename(folder / f'ACTUAL_{epoch:03d}_interrupted_{time.time_ns()}.pt')
        os.link(actual_path, destination)
    event = dict(arm=arm, epoch=epoch, updates=75*epoch, protocol_sha256=ph, schedule_epoch=train.epoch,
        rows_sha256=train.rows_hash, validation=actual, best=best, mean_training_loss=loss_sum/2400,
        count_examples=counts, gradient_norm_mean=sum(norms)/len(norms), gradient_norm_max=max(norms),
        train_seconds=elapsed, used_learning_rates=used_lr, next_learning_rates=[g['lr'] for g in opt.param_groups],
        lr_monitor=monitor, resume_provenance=provenance, actual_weights_sha256=actual_sha,
        last_sha256=w.digest(folder / 'LAST.pt'), validation_sha256=w.digest(vp),
        exact_last_roundtrip_checked=True, peak_bytes=torch.cuda.max_memory_allocated())
    w.write(folder / f'EPOCH_{epoch:03d}.json', event)
    print(dict(event='EPOCH_COMPLETE', arm=arm, epoch=epoch, validation=actual, best=best), flush=True)


def run(root, p):
    with s.worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); verify(root, p)
        assert torch.cuda.is_available()
        s.worker.guard.wait_for_predecessor(872424, p['original_protocol']['original_protocol']['gpu_display_policy'])
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        old = p['original_protocol']['original_protocol']
        _, ids = w.validate(Path(old['baseline']), None)
        dev = s.worker.NativeMixtures(old['preparation'], 'validation_pack', 1)
        progress(root, p)
        for epoch in range(2, p['max_epochs'] + 1):
            train = s.worker.NativeMixtures(old['preparation'], 'train_pack', (epoch + 1) % 5 + 1)
            assert len(train) == 2400
            for arm in (('septda', 'control') if epoch % 2 == 0 else ('control', 'septda')):
                if (root / arm / f'EPOCH_{epoch:03d}.json').exists(): continue
                verify(root, p); epoch_run(root, p, arm, epoch, train, dev, ids)
                gc.collect(); torch.cuda.empty_cache(); progress(root, p)
            del train
        verify(root, p)
        w.write(root / 'COMPLETE.json', dict(status='BUDGET_COMPLETE', epochs_per_arm=50, updates_per_arm=3750,
                 protocol_sha256=w.digest(root / 'PROTOCOL.json'), convergence_claimed=False, time=time.time()))
        w.write(root / 'STATE.json', dict(status='BUDGET_COMPLETE', pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    root = args.run.resolve(); root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            p = w.read(root / 'PROTOCOL.json') if args.resume else register(root)
            run(root, p)
    except Exception:
        w.write(root / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
