"""Bounded full-model TRAIN4 comparison of NEW-head learning rate only.

Queued behind the full matched study and its audit. Original 32.14M parameters
keep lr=1e-5 in both arms; only added 37,888 parameters vary. No validation
waveforms, no checkpoints reused, and no claim of architecture superiority.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
import train_comparison as worker
import output_fit
import gpu_start_guard as guard
from drone_rf.losses import pit_waveform_loss

w = worker.watch
ARMS = {'head_lr_1e5': 1e-5, 'head_lr_1e3': 1e-3}
CHECK = w.ROOT / 'reports/2026-10-10/HEAD_LR_PROBE_CPU_CHECK.json'
AUDIT = w.ROOT / 'reports/2026-10-10/SOURCE_INTERACTION_FINAL_AUDIT.json'


def optimizer(net, head_lr):
    old, new = [], []
    for name, parameter in net.named_parameters():
        (new if name.startswith('output.') and not name.startswith('output.base.') else old).append(parameter)
    if sum(p.numel() for p in old) != 32_142_859 or sum(p.numel() for p in new) != 37_888:
        raise ValueError('Parameter group size mismatch')
    if len({id(p) for p in old + new}) != len(list(net.parameters())):
        raise ValueError('Duplicate or missing optimizer parameter')
    if not all(p.requires_grad for p in old + new):
        raise ValueError('The full model must remain trainable')
    return torch.optim.AdamW([{'params': old, 'lr': 1e-5}, {'params': new, 'lr': head_lr}],
                            weight_decay=1e-4, foreach=False)


def check():
    torch.set_num_threads(2)
    net = worker.make_model('source_interaction')
    groups = {}
    for arm, lr in ARMS.items():
        opt = optimizer(net, lr)
        groups[arm] = [dict(parameters=sum(p.numel() for p in g['params']),
                            lr=g['lr'], weight_decay=g['weight_decay']) for g in opt.param_groups]
    result = dict(status='PASS', worker_sha256=w.digest(Path(__file__)), groups=groups,
        unique_exhaustive_parameter_partition=True, all_32180747_parameters_trainable=True,
        recorded_iq_reads=0, gpu_use=False, optimizer_steps=0)
    w.write(CHECK, result)
    return result


def verify(root, p):
    for rel, sha in p['source_sha256'].items():
        if w.digest(w.ROOT / rel) != sha or w.digest(root / 'source_snapshot' / rel) != sha:
            raise ValueError('Frozen probe source changed: ' + rel)
    for path, sha in ((Path(p['dependency']) / 'PROTOCOL.json', p['dependency_protocol_sha256']),
                      (Path(p['parent_checkpoint']), p['parent_checkpoint_sha256']),
                      (Path(p['preparation']) / 'PREPARATION.json', p['preparation_sha256']),
                      (CHECK, p['cpu_check_sha256'])):
        if w.digest(path) != sha:
            raise ValueError('Frozen probe dependency changed')


def register(root, dependency):
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate learning-rate probe')
    base = w.read(dependency / 'PROTOCOL.json')
    c = w.read(CHECK)
    if c['status'] != 'PASS' or c['worker_sha256'] != w.digest(Path(__file__)):
        raise ValueError('Optimizer partition check required')
    sources = dict(base['source_sha256'])
    sources[str(Path(__file__).relative_to(w.ROOT))] = w.digest(Path(__file__))
    pre = w.read(Path(base['dependency']) / 'PROTOCOL.json')
    p = dict(status='REGISTERED_QUEUED_NEW_HEAD_LR_PROBE', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=w.digest(dependency / 'PROTOCOL.json'),
        cpu_check_sha256=w.digest(CHECK), preparation=base['preparation'],
        preparation_sha256=base['preparation_sha256'], parent_checkpoint=base['parent_checkpoint'],
        parent_checkpoint_sha256=base['parent_checkpoint_sha256'], gpu_display_policy=base['gpu_display_policy'],
        train_indices=pre['train_indices'], steps_per_arm=32, effective_batch=4, microbatch=1,
        parameters_per_arm=32_180_747, backbone_parameters=32_142_859, added_parameters=37_888,
        backbone_lr=1e-5, added_head_lr=ARMS, arms=list(ARMS), seed=0,
        optimizer='fresh AdamW wd1e-4 clip1; FP32 TF32off; two explicit parameter groups',
        loss='same original waveform PIT NMSE+coherence+inactive/background plus0.1 count CE',
        initialization='same retained parent and freshly zero-initialized readout in both arms',
        observations=[0, 1, 8, 16, 32], validation_read=False, heldout_read=False, weights_discarded=True,
        success='Report finite updates, counts2/3 NMSE and complex SI-SDR, and direct on/off output effect. Higher LR is not preselected as better.',
        purpose='Optimization diagnosis of a small newly initialized branch, not a model or generalization comparison',
        limitation='Same four repeatedly examined TRAIN mixtures; 100x head-LR perturbation is a probe, not an optimum.',
        registered_at=time.time(), maximum_wait_seconds=24*3600)
    for rel, sha in sources.items():
        if w.digest(w.ROOT / rel) != sha:
            raise ValueError('Source changed before registration')
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(w.ROOT / rel, target)
    w.write(root / 'PROTOCOL.json', p)
    return p


@torch.no_grad()
def effect(net, items):
    net.eval()
    head = net.output
    rows = []
    for index, item in enumerate(items):
        net.output = head
        on, on_logits = worker.predict(net, item)
        net.output = head.base
        off, off_logits = worker.predict(net, item)
        net.output = head
        torch.testing.assert_close(on_logits, off_logits, rtol=0, atol=0)
        energy = (on.to(torch.complex128) - off.to(torch.complex128)).abs().square().mean(-1)
        mix_power = item['mixture'].to(torch.complex128).abs().square().mean()
        rows.append(dict(case=index, raw_slot_delta_energy_over_mixture=(energy[0] / mix_power).tolist()))
    return rows


def run(root, p, public):
    dependency = Path(p['dependency'])
    while not (dependency / 'COMPLETE.json').exists() or not AUDIT.exists():
        if (dependency / 'FAILURE.json').exists() or (w.ROOT / 'local/source_interaction_audit_20261010_v1/FAILURE.json').exists():
            raise RuntimeError('Predecessor or its audit failed')
        if time.time() - p['registered_at'] > p['maximum_wait_seconds']:
            raise TimeoutError('Predecessor wait expired')
        w.write(root / 'STATE.json', dict(status='WAITING_FULL_STUDY_AND_AUDIT', updates=0,
            pid=os.getpid(), time=time.time()))
        time.sleep(30)
    audit = w.read(AUDIT)
    if audit['status'] != 'PASS' or audit['study_protocol_sha256'] != p['dependency_protocol_sha256'] or audit['complete_sha256'] != w.digest(dependency / 'COMPLETE.json'):
        raise ValueError('Completed study audit mismatch')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(root, p)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        w.write(root / 'GPU_START_CHECK.json', guard.wait_for_predecessor(
            w.read(dependency / 'STATE.json')['pid'], p['gpu_display_policy']))
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        train = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
        items = [worker.fit.base.batch([train[i]]) for i in p['train_indices']]
        initial = None
        results = []
        for arm, lr in ARMS.items():
            verify(root, p)
            net = worker.make_model('source_interaction', Path(p['parent_checkpoint'])).cuda().eval()
            with torch.no_grad():
                values = [tuple(x.cpu() for x in worker.predict(net, item)) for item in items]
            if initial is None:
                initial = values
            else:
                for a, b in zip(initial, values):
                    for x, y in zip(a, b):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
            del values
            torch.manual_seed(0)
            opt = optimizer(net, lr)
            history = [dict(step=0, **output_fit.score(net, items))]
            began = time.time()
            for step in range(1, 33):
                net.train()
                opt.zero_grad(set_to_none=True)
                for item in items:
                    estimates, logits = worker.predict(net, item)
                    loss = pit_waveform_loss(estimates, item['references'], item['active'], item['mixture'])['loss']
                    loss = loss + .1 * F.cross_entropy(logits, item['construction_count'] - 1)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    (loss / 4).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                if step in p['observations']:
                    history.append(dict(step=step, **output_fit.score(net, items)))
                    w.write(root / f'{arm}_HISTORY.json', dict(history=history, partial=step < 32))
                    w.write(root / 'STATE.json', dict(status='GPU_TRAIN4_HEAD_LR_PROBE', arm=arm,
                        step=step, total=32, pid=os.getpid(), time=time.time()))
            if {int(s['step']) for s in opt.state.values()} != {32} or not all(torch.isfinite(v).all() for v in net.parameters()):
                raise ValueError('Optimizer or finite parameter check failed')
            results.append(dict(arm=arm, updates=32, history=history, direct_head_effect=effect(net, items),
                final_readout_norm=float(net.output.readout.weight.norm()),
                optimizer_groups=[dict(lr=g['lr'], parameters=sum(t.numel() for t in g['params'])) for g in opt.param_groups],
                seconds=time.time() - began))
            w.write(root / f'{arm}_COMPLETE.json', results[-1])
            del net, opt, item, estimates, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
        verify(root, p)
        result = dict(status='COMPLETE', protocol_sha256=w.digest(root / 'PROTOCOL.json'),
            results=results, initial_predictions_exactly_equal=True, validation_read=False,
            heldout_read=False, weights_discarded=True, time=time.time())
        w.write(root / 'COMPLETE.json', result)
        w.write(public, result)
        w.write(root / 'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
        print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    for key in ('study', 'run', 'public'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    if args.check_only:
        print(check())
    else:
        if any(getattr(args, key) is None for key in ('study', 'run', 'public')):
            parser.error('--study --run --public required unless --check-only')
        root = args.run.resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            with (root / '.run.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                plan = register(root, args.study.resolve())
                run(root, plan, args.public.resolve())
        except Exception:
            w.write(root / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
            raise
