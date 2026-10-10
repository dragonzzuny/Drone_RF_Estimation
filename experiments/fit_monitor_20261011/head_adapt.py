"""Conditional full-model/output-layer adaptation; no architecture reduction."""
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import diversity as base

ROOT = Path(__file__).resolve().parents[2]
fit, w, study, worker = base.fit, base.w, base.study, base.worker
OUT = ROOT / 'local/tfgridnet_head_adapt_20261011_v1'
COMPARATOR = ROOT / 'local/tfgridnet_diversity_20261011_v1'


def state(status, **fields):
    w.write(OUT / 'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **fields))


def evaluate(net, data, indices):
    # The shared evaluator only saves status and returns metrics. Redirect its
    # process-local output path; no file or concurrently running process changes.
    base.OUT = OUT
    point = base.evaluate(net, data, indices)
    for group in point['by_count']:
        group['median_nmse'] = float(np.median([v for r in point['rows']
                                               if r['count'] == group['count'] for v in r['nmse']]))
    return point


def compare(a, b):
    return [dict(count=x['count'], nmse=y['mean_nmse'] < x['mean_nmse'],
                 si=y['mean_si_sdr'] > x['mean_si_sdr'], weak=y['weakest_nmse'] <= x['weakest_nmse'],
                 median=y['median_nmse'] <= x['median_nmse'])
            for x, y in zip(a['by_count'], b['by_count'])]


def main():
    comparator = w.read(COMPARATOR / 'COMPLETE.json')
    assert comparator['status'] == 'COMPLETE' and not comparator['probe_improvement_candidate']
    assert w.read(ROOT / 'reports/2026-10-11/TFGRIDNET_DIVERSITY_AUDIT.json')['status'] == 'PASS'
    preflight = ROOT / 'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_CPU_CHECK.json'
    check = w.read(preflight)
    assert check['status'] == 'PASS'
    for rel, digest in check['source_sha256'].items():
        assert w.digest(ROOT / rel) == digest
    OUT.mkdir(exist_ok=False)
    p = copy.deepcopy(w.read(COMPARATOR / 'PROTOCOL.json'))
    fit.verify(COMPARATOR, p)
    checkpoint = Path(p['parent_checkpoint'])
    assert w.digest(checkpoint) == p['parent_checkpoint_sha256']
    plan = ROOT / 'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_PLAN_KO.md'
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))] = w.digest(Path(__file__))
    for rel in p['source_sha256']:
        target = OUT / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    p['pinned_files'][str(plan)] = w.digest(plan)
    p['pinned_files'][str(preflight)] = w.digest(preflight)
    p.update(training_parameters=9224, trainable_names=['deconv.weight','deconv.bias'],
             maximum_training_seconds=1200, observations=[0,16,32],
             stop_rule='Stop at first saved point without joint improvement including median; max32 or20minutes',
             selection='Retain last jointly improving saved point, starting at64; median criterion prospectively added',
             success='TRAIN probe improvement only; not DEV or new recordings',
             control='Reuse completed full-model adaptation at matched updates; no new control training',
             registered_at=time.time())
    w.write(OUT / 'PROTOCOL.json', p)
    ph = w.digest(OUT / 'PROTOCOL.json')
    fit.verify(OUT, p)
    data = worker.NativeMixtures(p['preparation'], 'train_pack', 1)
    assert data.rows_hash == p['source_rows_sha256']
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK')
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert torch.cuda.is_available()
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net = base.build().cuda()
        net.load_state_dict(saved['model'])
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name in p['trainable_names'])
        assert sum(t.numel() for t in net.parameters()) == p['parameters']
        assert sum(t.numel() for t in net.parameters() if t.requires_grad) == p['training_parameters']
        for module in net.modules():
            if isinstance(module, base.ChunkedLSTM):
                module.chunk = p['lstm_chunk']
        # Keep the original optimizer state/order. Frozen parameters receive no
        # gradient, so AdamW neither steps nor decays them.
        opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'],
                               weight_decay=p['weight_decay'], foreach=False)
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        frozen = {k:v for k,v in saved['model'].items() if k not in p['trainable_names']}
        del saved
        began = time.time()
        history = []
        selected = None
        reason = 'MAX32_REVIEW'
        for added in range(33):
            if added:
                net.train()
                opt.zero_grad(set_to_none=True)
                total = 0.
                for case, index in enumerate(p['batches'][added-1]):
                    state('TRAINING', completed_added_updates=added-1, working_added_update=added,
                          total_added_updates=32, case=case+1, index=index, seconds=time.time()-began)
                    item = worker.fit.base.batch([data[index]])
                    out, logits = worker.predict(net, item)
                    loss = study.original.prior.pit_waveform_loss(out, item['references'],
                                item['active'], item['mixture'])['loss']
                    loss = loss + .1*torch.nn.functional.cross_entropy(logits, item['construction_count']-1)
                    assert torch.isfinite(loss)
                    (loss/4).backward()
                    total += float(loss.detach())/4
                assert {n for n,t in net.named_parameters() if t.grad is not None} == set(p['trainable_names'])
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                opt.step()
                for name, parameter in net.named_parameters():
                    assert int(opt.state[parameter]['step']) == (64+added if parameter.requires_grad else 64)
            timed = time.time()-began >= 1200
            if added not in (0,16,32) and not timed:
                continue
            point = dict(added_updates=added, total_updates=64+added,
                         **evaluate(net, data, p['probe_indices']), seconds=time.time()-began)
            for key, value in frozen.items():
                assert torch.equal(net.state_dict()[key].detach().cpu(), value), key
            if not added:
                original = w.read(COMPARATOR / 'ADDED_000.json')
                for a,b in zip(original['rows'], point['rows']):
                    assert a['index'] == b['index']
                    np.testing.assert_allclose(a['nmse'], b['nmse'], rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(a['si_sdr'], b['si_sdr'], rtol=1e-5, atol=1e-4)
                shutil.copyfile(checkpoint, OUT / 'SELECTED.pt')
                selected = point
            else:
                worker.atomic_torch(OUT / 'LAST.pt', dict(model=net.state_dict(), optimizer=opt.state_dict(),
                    updates=64+added, protocol_sha256=ph, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()))
                shutil.copyfile(OUT / 'LAST.pt', OUT / f'ADDED_{added:03d}.pt')
                point.update(checkpoint_sha256=w.digest(OUT / 'LAST.pt'), training_loss=total,
                             gradient_norm_before_clip=norm)
                checks = compare(selected, point)
                passed = all(c['nmse'] and c['si'] and c['weak'] and c['median'] for c in checks)
                point.update(checks=checks, continuation_pass=passed)
                if passed:
                    selected = point
                    shutil.copyfile(OUT / 'LAST.pt', OUT / 'SELECTED.pt')
                else:
                    reason = 'NO_JOINT_IMPROVEMENT_REVIEW'
            history.append(point)
            w.write(OUT / f'ADDED_{added:03d}.json', point)
            progress = dict(status='TRAIN_PROBE_DIAGNOSIS', history=history, added_updates=added,
                protocol_sha256=ph, selected_added_updates=selected['added_updates'],
                validation_read=False, heldout_read=False, incumbent_replaced=False)
            w.write(OUT / 'HISTORY.json', progress)
            w.write(ROOT / 'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_PROGRESS.json', progress)
            print(dict(added_updates=added, by_count=point['by_count']), flush=True)
            if added and not passed:
                break
            if timed:
                reason = 'TIME_BUDGET_REVIEW'
                break
        fit.verify(OUT, p)
        result = dict(status='COMPLETE', reason=reason, history=history, final=point,
            added_updates=added, total_updates=64+added, selected=selected,
            selected_checkpoint_sha256=w.digest(OUT / 'SELECTED.pt'), parameters=p['parameters'],
            training_parameters=p['training_parameters'], frozen_parameters_bitwise_unchanged=True,
            protocol_sha256=ph, validation_read=False, heldout_read=False, incumbent_replaced=False,
            seconds=time.time()-began)
        w.write(OUT / 'COMPLETE.json', result)
        w.write(ROOT / 'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_RESULT.json', result)
        state('COMPLETED', added_updates=added, selected_added_updates=selected['added_updates'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            w.write(OUT / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
            state('FAILED')
        raise
