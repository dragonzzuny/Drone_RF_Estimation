"""Verified epoch50 handoff; parent source and completed results stay immutable."""
import fcntl
import os
import shutil
import time

import torch

from data import *

PARENT = ROOT/'local/bandwidth_timeblock_20261011_v1'
PUBLIC = ROOT/'reports/2026-10-12'
PLAN = PUBLIC/'TIMEBLOCK100_PLAN_KO.md'


def verify_parent():
    p = read(PARENT/'PROTOCOL.json')
    for rel, h in p['source_sha256'].items():
        assert sha256(ROOT/rel) == sha256(PARENT/'source_snapshot'/rel) == h, rel
    for name, h in p['pinned_files'].items():
        assert sha256(name) == h, name
    return p


def register_extension():
    assert not (OUT/'PROTOCOL.json').exists(), 'Extension already registered'
    parent = verify_parent()
    ready = read(OUT/'READY.json')
    assert ready['status'] == 'PASS'
    for rel, h in ready['source_sha256'].items():
        assert sha256(ROOT/rel) == h
    implementation = read(PUBLIC/'TIMEBLOCK100_IMPLEMENTATION_CHECK.json')
    assert implementation['status'] == 'PASS'
    for rel, h in implementation['source_sha256'].items():
        assert sha256(ROOT/rel) == h
    p = dict(parent)
    p.update(status='REGISTERED_CONTINUATION_TO_TOTAL100', epochs=100,
        first_extension_epoch=51, additional_epochs=50, total_updates=7500,
        initialization='Continue audited parent epoch50 LAST, including optimizer/scheduler/RNG; no reset',
        same_B_checkpoint_reuse=True, unrelated_checkpoint_reuse=False,
        parent_protocol_sha256=sha256(PARENT/'PROTOCOL.json'),
        parent_run=str(PARENT.relative_to(ROOT)), registered_at=time.time(),
        amendment='User explicitly extended total budget from50 to100 on2026-10-12',
        performance_early_stopping=False)
    p.pop('old_checkpoint_reuse', None)
    hashes = dict(parent['source_sha256'])
    hashes.update({str(f.relative_to(ROOT)):sha256(f) for f in HERE.glob('*.py')})
    for rel, h in hashes.items():
        target = OUT/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        assert sha256(target) == h
    p['source_sha256'] = hashes
    pins = dict(parent['pinned_files'])
    for f in [PARENT/'PROTOCOL.json', PLAN, OUT/'READY.json', PREP/'PREPARATION.json',
              PUBLIC/'TIMEBLOCK100_IMPLEMENTATION_CHECK.json']:
        pins[str(f)] = sha256(f)
    p['pinned_files'] = pins
    dev = Mixtures(ROLES[1])
    assert dev.rows_hash == parent['fixed_dev_schedule_sha256']
    write_json(OUT/'PROTOCOL.json', p)
    write_json(PUBLIC/'TIMEBLOCK100_REGISTRATION.json', dict(
        status=p['status'], epochs=100, first_extension_epoch=51, total_updates=7500,
        protocol_sha256=sha256(OUT/'PROTOCOL.json'),
        parent_protocol_sha256=p['parent_protocol_sha256'],
        fixed_dev_schedule_sha256=dev.rows_hash,
        preserved_model_loss_split=True, performance_early_stopping=False,
        independent_recording_validation=False))
    print('REGISTERED_TOTAL100_EXTENSION', flush=True)
    return p


def audited_event(run, epoch):
    event_path = run/f'EPOCH_{epoch:03d}.json'
    audit = read(run/f'AUDIT_{epoch:03d}.json')
    event = read(event_path)
    assert audit['status'] == 'PASS'
    assert audit['event_sha256'] == sha256(event_path)
    assert audit['protocol_sha256'] == event['protocol_sha256'] == sha256(run/'PROTOCOL.json')
    return event


def wait_parent(state, stop):
    """No CUDA allocation or GPU lock until parent finishes its existing budget."""
    verify_parent()
    while True:
        if stop():
            state('STOPPED_BEFORE_HANDOFF')
            raise SystemExit(0)
        worker = read(PARENT/'STATE.json')
        if worker['status'] == 'COMPLETE_AUDITED':
            assert worker['epoch'] == 50 and worker['updates'] == 3750
            audited_event(PARENT, 50)
            # Completion is written before the parent's contexts release locks.
            with (PARENT/'.run.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(2)
                    continue
            proc = Path(f"/proc/{worker['pid']}/cmdline")
            try:
                exiting = b'bandwidth_timeblock_20261011/train.py' in proc.read_bytes()
            except FileNotFoundError:
                exiting = False
            if exiting:
                time.sleep(2)
                continue
            return
        proc = Path(f"/proc/{worker['pid']}/cmdline")
        try:
            alive = b'bandwidth_timeblock_20261011/train.py' in proc.read_bytes()
        except FileNotFoundError:
            alive = False
        assert alive and worker['status'] not in ('FAILED','STOPPED_WITH_RECOVERY'), \
            'Parent stopped before audited epoch50; preserve state and diagnose'
        state('WAITING_PARENT_E050', epoch=worker.get('epoch',0),
              updates=worker.get('updates',0), parent_state=worker)
        time.sleep(10)


def exact(a, b):
    """Exact numerical equality across serialization/device moves, including RNG."""
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b.to(a.device))
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            exact(x,y)
    else:
        assert a == b, (a,b)


def restore(saved, net, opt, scheduler, cuda=True):
    net.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['torch_rng'])
    if cuda:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    exact(saved['model'], net.state_dict())
    exact(saved['optimizer'], opt.state_dict())
    exact(saved['scheduler'], scheduler.state_dict())
    exact(saved['torch_rng'], torch.get_rng_state())
    if cuda:
        exact(saved['cuda_rng'], torch.cuda.get_rng_state_all())


def install_parent(net, opt, scheduler, ph, save_checkpoint):
    assert not (OUT/'HANDOFF.json').exists(), 'Use --resume after handoff'
    verify_parent()
    event = audited_event(PARENT, 50)
    saved = torch.load(PARENT/'LAST.pt', map_location='cpu', weights_only=False, mmap=True)
    assert saved['epoch'] == 50 and saved['updates'] == 3750 and saved['epoch_evaluated']
    assert saved['protocol_sha256'] == sha256(PARENT/'PROTOCOL.json')
    assert {int(x['step']) for x in saved['optimizer']['state'].values()} == {3750}
    assert saved['best'] == event['best']
    actual = torch.load(PARENT/'ACTUAL_050.pt', map_location='cpu', weights_only=False, mmap=True)
    assert sha256(PARENT/'ACTUAL_050.pt') == event['checkpoint_sha256']
    exact(saved['model'], actual['model'])
    for slot in saved['optimizer']['state'].values():
        for value in slot.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
    restore(saved, net, opt, scheduler)
    from study import atomic_torch
    # Carry the best over all epochs; continuation itself always uses LAST50.
    best = torch.load(PARENT/'BEST.pt', map_location='cpu', weights_only=False, mmap=True)
    be = saved['best']['epoch']
    assert best['epoch'] == be
    selected_path = PARENT/(f'ACTUAL_{be:03d}.pt' if be else 'INITIAL.pt')
    selected = torch.load(selected_path, map_location='cpu', weights_only=False, mmap=True)
    exact(best['model'], selected['model'])
    atomic_torch(OUT/'BEST.pt', dict(model=best['model'], epoch=be,
        updates=best['updates'], best=saved['best'], protocol_sha256=ph,
        inherited_parent_protocol_sha256=saved['protocol_sha256']))
    # Hard links retain the exact parent handoff states without duplicate disk use.
    for src,name in [('LAST.pt','PARENT_LAST_E050.pt'),('BEST.pt','PARENT_BEST_AT_E050.pt')]:
        target = OUT/name
        if not target.exists():
            os.link(PARENT/src, target)
        assert sha256(target) == sha256(PARENT/src)
    save_checkpoint(OUT/'LAST.pt',net,opt,scheduler,50,3750,saved['best'],ph,epoch_evaluated=True)
    receipt = dict(status='PASS', parent_epoch=50, updates=3750,
        next_epoch=51, total_epochs=100, parent_last_sha256=sha256(PARENT/'LAST.pt'),
        parent_best_sha256=sha256(PARENT/'BEST.pt'),
        parent_epoch_event_sha256=sha256(PARENT/'EPOCH_050.json'),
        optimizer_model_scheduler_rng_exact=True, optimizer_step=3750,
        learning_rate=opt.param_groups[0]['lr'], best=saved['best'],
        parent_protocol_sha256=saved['protocol_sha256'], protocol_sha256=ph)
    write_json(OUT/'HANDOFF.json', receipt)
    write_json(PUBLIC/'TIMEBLOCK100_HANDOFF.json', receipt)
    print('PARENT_E050_HANDOFF_PASS '+str(saved['best']), flush=True)
    return 51, 3750, saved['best']


def render_history():
    lines = ['# 동일10MHz B 진단: 총100epoch 연속 학습', '',
        '1–50epoch와51–100epoch는 같은 모델·분할·손실·고정DEV를 사용한다. '
        '동일 녹음 시간 분리 진단이며 독립 녹음 일반화 결과가 아니다.', '',
        '| Epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 최약 NMSE 2/3 ↓ | 선택 epoch |',
        '|---:|---|---|---|---:|']
    for k in range(1,101):
        run = PARENT if k <= 50 else OUT
        if not (run/f'AUDIT_{k:03d}.json').exists():
            break
        e = audited_event(run,k)
        a,b = e['validation']['by_count'][1:]
        si = lambda v:'미정의' if v is None else f'{v:.3f}'
        lines.append(f"|{k}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
            f"{si(a['mean_si_sdr'])}/{si(b['mean_si_sdr'])}|"
            f"{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{e['best']['epoch']}|")
    (PUBLIC/'TIMEBLOCK100_EPOCH_REPORTS_KO.md').write_text('\n'.join(lines)+'\n')
