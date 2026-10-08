"""Native-size CUDA checks on training mixtures only; updates are discarded."""
import gc
import json
import math
import time
import numpy as np
import torch
from late_models import build, predict, stack_batch, PARENT, WaveformUNet
from drone_rf.context_training_data import write_json
from drone_rf.waveform import waveform_metrics


def fixed_train_diagnostic(arm, data, folder, frozen, optimizer_for, objective):
    """Fit four TRAIN cases only; never keep these updates for the comparison."""
    start = time.time()
    indices = [int(i) for k in (2, 3) for i in np.flatnonzero(data.rows['count'] == k)[:2]]
    item = stack_batch([data[i] for i in indices], 'cuda')
    net = build(arm).cuda()
    optimizer = optimizer_for(net)
    curve = []

    @torch.no_grad()
    def score(step):
        net.eval()
        estimate, _ = predict(net, item)
        metrics = waveform_metrics(estimate, item['references'], item['active'], item['mixture'])
        per_case = [float(metrics['nmse'][i][item['active'][i]].mean()) for i in range(4)]
        return dict(step=step, objective=float(objective(net, item)),
            case_nmse=per_case, mean_nmse=float(np.mean(per_case)))

    curve.append(score(0))
    for step in range(1, 33):
        net.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective(net, item)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite fixed-train objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (8, 16, 32):
            curve.append(score(step))
            progress = dict(stage='TRAIN_ONLY_FIXED_DIAGNOSTIC', arm=arm, step=step,
                total_steps=32, curve=curve, time=time.time())
            write_json(folder.parent/'PROGRESS.json', progress)
            print(json.dumps(progress), flush=True)
    accepted = (curve[-1]['objective'] < curve[0]['objective']
                and curve[-1]['mean_nmse'] < curve[0]['mean_nmse']
                and all(math.isfinite(r['mean_nmse']) for r in curve))
    receipt = dict(status='PASS' if accepted else 'FAIL', protocol_sha256=frozen,
        indices=indices, construction_counts=[2, 2, 3, 3], discarded_updates=32,
        dataset_role='train_pack epoch1 only', initialization='fresh common parent and seed0 branch',
        learning_rates=[1e-5, 1e-4], curve=curve, seconds=time.time()-start,
        interpretation='implementation fitting diagnostic, not validation or evidence of long-context superiority')
    write_json(folder/'TRAIN_ONLY_DIAGNOSTIC.json', receipt)
    del net, optimizer, item, loss
    gc.collect(); torch.cuda.empty_cache()
    if not accepted:
        raise RuntimeError('Fixed-train diagnostic did not reduce objective and unscaled NMSE')
    return receipt


def preflight(arm, data, folder, frozen, optimizer_for, objective, microbatch):
    path = folder / 'GPU_PREFLIGHT.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['protocol_sha256'] != frozen or saved['status'] != 'PASS':
            raise RuntimeError('Preflight receipt changed')
        diagnostic = json.loads((folder/'TRAIN_ONLY_DIAGNOSTIC.json').read_text())
        if diagnostic['protocol_sha256'] != frozen or diagnostic['status'] != 'PASS':
            raise RuntimeError('Missing or incompatible fixed-train gate')
        return
    start = time.time()
    net = build(arm).cuda().eval()
    indices = [int(np.flatnonzero(data.rows['count'] == k)[0]) for k in (1, 2, 3, 2)]
    items = [data[i] for i in indices]
    one = stack_batch(items[1:2], 'cuda')
    parent = WaveformUNet(False).cuda().eval()
    parent.load_state_dict(torch.load(PARENT, map_location='cpu', weights_only=False)['model'])
    with torch.no_grad():
        net.fusion_enabled = False
        bypass, _ = predict(net, one)
        original, _ = predict(parent, one)
        if not torch.equal(bypass, original):
            raise RuntimeError('Local path differs from the frozen parent with fusion bypassed')
        net.fusion_enabled = True
        before, _ = predict(net, one)
        offsets = one['crop_start'] - one['long_start']
        positions = torch.arange(one['long_mixture'].shape[-1], device='cuda')[None]
        outside = (positions < offsets[:, None]) | (positions >= offsets[:, None] + one['mixture'].shape[-1])
        changed = dict(one, long_mixture=torch.where(outside, one['long_mixture'] * 1j, one['long_mixture']))
        after, _ = predict(net, changed)
        sensitivity = float((after - before).abs().square().sum() / before.abs().square().sum().clamp_min(1e-20))
        if (arm == 'local_only' and not torch.equal(before, after)) or (arm == 'local_global' and not sensitivity > 0):
            raise RuntimeError('Context information control failed')
        changed = dict(one, references=torch.zeros_like(one['references']), construction_count=torch.ones_like(one['construction_count']))
        blind, _ = predict(net, changed)
        if not torch.equal(before, blind):
            raise RuntimeError('Labels reached inference')
        sum_error = float((before.sum(1) - one['mixture']).abs().square().sum() / one['mixture'].abs().square().sum())
        if sum_error > 1e-9:
            raise RuntimeError('Mixture consistency failed')
        joined = stack_batch(items, 'cuda')
        together, _ = predict(net, joined)
        batch_error = float((together[1] - before[0]).abs().square().sum() / before[0].abs().square().sum())
        if batch_error > 1e-9:
            raise RuntimeError('Microbatch changed predictions beyond numerical tolerance')
    del parent, bypass, original, before, after, blind, changed, together, joined
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net.train()
    optimizer = optimizer_for(net)
    joined = stack_batch(items[:microbatch], 'cuda')
    loss = objective(net, joined)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite preflight loss')
    loss.backward()
    gradient = {}
    for name in ('iq_context', 'decoder_fusion'):
        parameters = list(getattr(net, name).parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('Missing/nonfinite context gradient')
        gradient[name] = math.sqrt(sum(float(p.grad.square().sum()) for p in parameters))
        if not gradient[name] > 0:
            raise RuntimeError('Inactive context gradient')
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); torch.cuda.synchronize()
    receipt = dict(status='PASS', protocol_sha256=frozen, parameters=sum(p.numel() for p in net.parameters()),
        microbatch=microbatch, counts=[1, 2, 3, 2][:microbatch], discarded_updates=1,
        parent_bypass_exact=True, reference_count_input_independence=True,
        outside_phase_perturbation_relative_output_energy=sensitivity, microbatch_relative_error=batch_error,
        sum_relative_error=sum_error, gradient_norms=gradient, total_gradient_norm=float(norm),
        loss=float(loss.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-start)
    del net, optimizer, joined, loss, one
    gc.collect(); torch.cuda.empty_cache()
    receipt['train_only_diagnostic'] = fixed_train_diagnostic(arm, data, folder, frozen, optimizer_for, objective)
    write_json(path, receipt); print(json.dumps(dict(arm=arm, **receipt)), flush=True)
