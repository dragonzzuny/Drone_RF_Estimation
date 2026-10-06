"""GPU-only matched full U-Net temporal comparison on recorded-component mixtures.

Synthetic-count accuracy is not physical drone-count accuracy. Validation uses
pack-disjoint short waveform crops, not whole-record oracle window matching.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from drone_rf.context_model import ContextualSeparator
from drone_rf.context_training_data import ContextMixtures, write_json
from drone_rf.data import sha256
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import predict, waveform_metrics


def atomic_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model(config, arm):
    return ContextualSeparator(config['context_kind'], context_mode=arm)


def batch(item, device='cuda'):
    keys = ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')
    return {key: torch.as_tensor(item[key], device=device)[None] for key in keys}


def objective(net, item, config):
    estimate, count_logits = predict(net, item)
    wave = pit_waveform_loss(estimate, item['references'], item['active'], item['mixture'])['loss']
    # Explicitly a separate synthetic construction task, not the gated physical
    # source_count_loss API. Nothing here certifies physical source provenance.
    count = F.cross_entropy(count_logits, item['construction_count'] - 1)
    return wave + config['construction_count_loss_weight'] * count, wave, count


def state_hash(net):
    h = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def serialize_extended(tensor):
    values, statuses = [], []
    for value in tensor.detach().cpu().tolist():
        if np.isfinite(value):
            values.append(value); statuses.append('finite')
        else:
            values.append(None)
            statuses.append('undefined' if np.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity')
    return values, statuses


def validate(net, dataset, output, epoch, save_examples=False):
    net.eval()
    rows = []
    saved_examples = []
    example_indices = {int(np.flatnonzero(dataset.rows['count'] == n)[0]) for n in (1, 2, 3)}
    started = time.time()
    with torch.no_grad():
        for index in range(len(dataset)):
            item = batch(dataset[index])
            estimates, logits = predict(net, item)
            value = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
            active = item['active'][0]
            si, si_status = serialize_extended(value['si_sdr'][0][active])
            input_si, input_si_status = serialize_extended(value['input_si_sdr'][0][active])
            gain = value['si_sdr'][0][active] - value['input_si_sdr'][0][active]
            gains, gain_status = serialize_extended(gain)
            row = dict(index=index, construction_count=int(item['construction_count'][0]),
                predicted_construction_count=int(logits.argmax(-1)[0]) + 1,
                count_probabilities=logits.softmax(-1)[0].cpu().tolist(),
                assignment=value['assignment'][0].cpu().tolist(),
                active=active.cpu().tolist(), nmse=value['nmse'][0][active].cpu().tolist(),
                si_sdr=si, si_status=si_status, input_si_sdr=input_si, input_si_status=input_si_status,
                si_sdr_improvement=gains, gain_status=gain_status,
                all_components_positive_improvement=bool(torch.all(gain > 0)),
                reference_power=value['reference_power'][0][active].cpu().tolist(),
                inactive_leak=float(value['inactive_leak'][0].sum()),
                background_nmse=float(value['background_nmse'][0]),
                sum_relative_error=float(value['sum_relative_error'][0]))
            rows.append(row)
            if save_examples and index in example_indices:
                directory = output.parent / (output.stem + '_examples')
                directory.mkdir(exist_ok=True)
                path = directory / f'case_{index:03d}.npz'
                np.savez_compressed(path, mixture=item['mixture'][0].cpu().numpy(),
                    references=item['references'][0].cpu().numpy(), predictions=estimates[0].cpu().numpy(),
                    assignment=value['assignment'][0].cpu().numpy(),
                    context_features=item['context_features'][0].cpu().numpy(),
                    crop_start=int(item['crop_start'][0]))
                saved_examples.append(dict(index=index, construction_count=row['construction_count'],
                    path=str(path), sha256=sha256(path), selected_by='first scheduled example of each count; not performance'))
            if index % 50 == 0:
                write_json(output.parent.parent / 'PROGRESS.json', dict(stage='VALIDATION', time=time.time(),
                    arm=output.parent.name, epoch=epoch, example=index + 1, examples=len(dataset), pid=os.getpid()))
    groups = []
    confusion = np.zeros((3, 3), dtype=int)
    for row in rows:
        confusion[row['construction_count'] - 1, row['predicted_construction_count'] - 1] += 1
    for count in (1, 2, 3):
        selected = [r for r in rows if r['construction_count'] == count]
        all_si_finite = all(s == 'finite' for r in selected for s in r['si_status'])
        all_gain_finite = all(s == 'finite' for r in selected for s in r['gain_status'])
        groups.append(dict(count=count, cases=len(selected),
            mean_component_nmse=float(np.mean([np.mean(r['nmse']) for r in selected])),
            mean_component_si_sdr=float(np.mean([np.mean(r['si_sdr']) for r in selected])) if all_si_finite else None,
            nonfinite_component_si_count=sum(s != 'finite' for r in selected for s in r['si_status']),
            mean_si_sdr_improvement=float(np.mean([np.mean(r['si_sdr_improvement']) for r in selected])) if count > 1 and all_gain_finite else None,
            all_components_positive_improvement=float(np.mean([r['all_components_positive_improvement'] for r in selected])) if count > 1 else None,
            weakest_component_nmse=float(np.mean([r['nmse'][int(np.argmin(r['reference_power']))] for r in selected])),
            count_accuracy=float(np.mean([r['construction_count'] == r['predicted_construction_count'] for r in selected]))))
    summary = dict(epoch=epoch, cases=len(rows), by_count=groups,
        macro_component_nmse=float(np.mean([r['mean_component_nmse'] for r in groups])),
        construction_count_accuracy=float(np.trace(confusion) / len(rows)),
        construction_count_mae=float(np.mean([abs(r['construction_count'] - r['predicted_construction_count']) for r in rows])),
        construction_count_confusion=confusion.tolist(), count_majority_baseline=1 / 3,
        max_sum_relative_error=max(r['sum_relative_error'] for r in rows),
        elapsed_seconds=time.time() - started, independent_test=False,
        physical_drone_count_evaluated=False, whole_record_tracking_evaluated=False,
        si_sdr_convention='mean removed; complex scalar invariant',
        single_component_note='input already equals reference; SI-SDR improvement is not a success criterion',
        saved_examples=saved_examples, rows=rows)
    write_json(output, summary)
    return summary


def wait_features(preparation, role, epoch, root):
    path = preparation / 'features' / f'{role}_{epoch:03d}.json'
    while not path.exists():
        failure = preparation / 'FEATURE_FAILURE.json'
        if failure.exists():
            raise RuntimeError('CPU feature worker failed')
        write_json(root / 'PROGRESS.json', dict(stage='WAITING_CPU_FEATURES', role=role,
            epoch=epoch, time=time.time(), pid=os.getpid(), gpu_training_active=False))
        time.sleep(5)


def verify_freeze(root):
    freeze = json.loads((root / 'FREEZE.json').read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen experiment changed: {Path(path).name}')
    return freeze


def preflight(args, config):
    receipt = args.run / 'GPU_PREFLIGHT.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['status'] != 'PASS_FULL_CONTEXT_GPU_TRAINING' or old['freeze_sha256'] != sha256(args.run / 'FREEZE.json'):
            raise ValueError('GPU preflight is not valid for this experiment')
        return
    wait_features(args.preparation, 'train_pack', 1, args.run)
    dataset = ContextMixtures(args.preparation, 'train_pack', 1)
    selected = [int(np.flatnonzero(dataset.rows['count'] == n)[0]) for n in (1, 2, 3)]
    arms = []
    for arm in config['arms']:
        seed()
        net = model(config, arm).cuda().train()
        initial_hash = state_hash(net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
            weight_decay=config['weight_decay'], foreach=False)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in selected:
            item = batch(dataset[index])
            loss, wave, count = objective(net, item, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite full-input preflight loss')
            (loss / len(selected)).backward()
            losses.append(float(loss.detach()))
        nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        if net.context_projection.weight.grad.abs().sum() == 0:
            raise ValueError('No gradient to temporal context')
        optimizer.step()
        torch.cuda.synchronize()
        arms.append(dict(arm=arm, input_samples=config['window_samples'], parameters=sum(p.numel() for p in net.parameters()),
            component_counts=[1, 2, 3], discarded_optimizer_updates=1,
            initial_parameters_sha256=initial_hash, losses=losses,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.time() - started))
        del loss, wave, count, item, optimizer, net
        gc.collect(); torch.cuda.empty_cache()
    if len({arm['initial_parameters_sha256'] for arm in arms}) != 1:
        raise ValueError('Matched arms have different initial parameters')
    write_json(receipt, dict(status='PASS_FULL_CONTEXT_GPU_TRAINING', time=time.time(), arms=arms,
        freeze_sha256=sha256(args.run / 'FREEZE.json'), updates_discarded=True,
        torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0),
        precision='float32', microbatch=1, training_effective_batch=32,
        exact_bitwise_gpu_determinism_claimed=False))


def save_checkpoint(path, net, optimizer, epoch, updates, best, args):
    atomic_torch(path, dict(model=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
        updates=updates, best=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        freeze_sha256=sha256(args.run / 'FREEZE.json')))


def fixed_mixture_diagnostic(args, config):
    """Full model fit to three TRAINING examples; not generalization evidence."""
    path = args.run / 'FIXED_MIXTURE_DIAGNOSTIC.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['freeze_sha256'] != sha256(args.run / 'FREEZE.json') or not value['wave_loss_decreased']:
            raise ValueError('Previous fixed-mixture diagnosis requires review')
        return
    seed()
    net = model(config, 'ordered').cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], foreach=False)
    dataset = ContextMixtures(args.preparation, 'train_pack', 1)
    indices = [int(np.flatnonzero(dataset.rows['count'] == n)[0]) for n in (1, 2, 3)]
    items = [batch(dataset[i]) for i in indices]
    def measure():
        net.eval()
        values = []
        with torch.no_grad():
            for item in items:
                _, wave, count = objective(net, item, config)
                values.append(dict(wave_loss=float(wave), construction_count_loss=float(count)))
        return values
    before = measure()
    started = time.time()
    net.train()
    for update in range(48):
        optimizer.zero_grad(set_to_none=True)
        for item in items:
            loss, _, _ = objective(net, item, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fixed-mixture loss')
            (loss / len(items)).backward()
        nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        if update % 4 == 0:
            write_json(args.run / 'PROGRESS.json', dict(stage='FIXED_TRAINING_MIXTURE_DIAGNOSTIC',
                updates=update + 1, target_updates=48, time=time.time(), pid=os.getpid()))
    after = measure()
    decreased = np.mean([r['wave_loss'] for r in after]) < np.mean([r['wave_loss'] for r in before])
    write_json(path, dict(status='FULL_MODEL_FIXED_MIXTURE_DIAGNOSTIC', time=time.time(),
        freeze_sha256=sha256(args.run / 'FREEZE.json'), training_example_indices=indices,
        component_counts=[1, 2, 3], updates=48, before=before, after=after,
        wave_loss_decreased=bool(decreased), elapsed_seconds=time.time() - started,
        generalization_evaluated=False, physical_aircraft_count_evaluated=False,
        weights_discarded=True, main_comparison_starts_from_original_seed=True))
    del net, optimizer, items, item, loss, dataset
    gc.collect(); torch.cuda.empty_cache()
    if not decreased:
        raise RuntimeError('Fixed-mixture fit did not reduce waveform loss; review before long training')


def train_arm_epoch(args, config, arm, epoch, validation):
    folder = args.run / arm
    folder.mkdir(exist_ok=True)
    seed()
    net = model(config, arm).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], foreach=False)
    last = folder / 'LAST.pt'
    if last.exists():
        saved = torch.load(last, map_location='cuda', weights_only=False)
        if saved['freeze_sha256'] != sha256(args.run / 'FREEZE.json'):
            raise ValueError('Checkpoint belongs to a different frozen experiment')
        if saved['epoch'] >= epoch:
            required = [folder / f'EPOCH_{epoch:03d}.json']
            if epoch in config['checkpoints']:
                required += [folder / f'STAGE_{epoch:03d}.json', folder / f'SELECTED_{epoch:03d}.json',
                             folder / f'SELECTED_{epoch:03d}.pt']
            if not all(path.exists() for path in required):
                raise RuntimeError('Checkpoint exists but epoch artifacts are incomplete; explicit recovery required')
            del net, optimizer, saved
            return
        if saved['epoch'] != epoch - 1:
            raise ValueError('Missing epoch checkpoint')
        net.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        best, updates = saved['best'], saved['updates']
        del saved
    else:
        if epoch != 1:
            raise ValueError('Missing initial checkpoint')
        # Both arms are generated from the same seed before any optimization.
        initial = state_hash(net)
        expected = json.loads((args.run / 'GPU_PREFLIGHT.json').read_text())['arms'][0]['initial_parameters_sha256']
        if initial != expected:
            raise ValueError('Training initialization differs from matched preflight')
        result = validate(net, validation, folder / 'VALIDATION_000.json', 0)
        best = dict(epoch=0, metric=result['macro_component_nmse'])
        updates = 0
        atomic_torch(folder / 'BEST.pt', dict(model=net.state_dict(), best=best,
            freeze_sha256=sha256(args.run / 'FREEZE.json')))
        save_checkpoint(last, net, optimizer, 0, updates, best, args)
    wait_features(args.preparation, 'train_pack', epoch, args.run)
    dataset = ContextMixtures(args.preparation, 'train_pack', epoch)
    if len(dataset) % config['effective_batch']:
        raise ValueError('Partial optimizer batch not in this protocol')
    net.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    totals = np.zeros(3)
    for index in range(len(dataset)):
        item = batch(dataset[index])
        loss, wave, count_loss = objective(net, item, config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss / config['effective_batch']).backward()
        totals += [float(loss.detach()), float(wave.detach()), float(count_loss.detach())]
        if (index + 1) % config['effective_batch'] == 0:
            nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
            if updates % 5 == 0:
                write_json(args.run / 'PROGRESS.json', dict(stage='TRAIN', arm=arm, epoch=epoch,
                    examples=index + 1, epoch_examples=len(dataset), updates=updates,
                    average_losses=(totals / (index + 1)).tolist(), elapsed_seconds=time.time() - started,
                    time=time.time(), pid=os.getpid(), gpu_training_active=True))
    train_seconds = time.time() - started
    result = validate(net, validation, folder / f'VALIDATION_{epoch:03d}.json', epoch)
    if result['macro_component_nmse'] < best['metric']:
        best = dict(epoch=epoch, metric=result['macro_component_nmse'])
        atomic_torch(folder / 'BEST.pt', dict(model=net.state_dict(), best=best,
            freeze_sha256=sha256(args.run / 'FREEZE.json')))
    save_checkpoint(last, net, optimizer, epoch, updates, best, args)
    entry = dict(arm=arm, epoch=epoch, updates=updates, train_seconds=train_seconds,
        training_mean_losses=(totals / len(dataset)).tolist(), best=best,
        validation={k: v for k, v in result.items() if k != 'rows'}, time=time.time())
    write_json(folder / f'EPOCH_{epoch:03d}.json', entry)
    write_json(args.run / 'PROGRESS.json', dict(stage='EPOCH_COMPLETE', **entry))
    if epoch in config['checkpoints']:
        saved = torch.load(folder / 'BEST.pt', map_location='cuda', weights_only=False)
        net.load_state_dict(saved['model'])
        selected = validate(net, validation, folder / f'SELECTED_{epoch:03d}.json', best['epoch'], save_examples=True)
        selected_checkpoint = folder / f'SELECTED_{epoch:03d}.pt'
        atomic_torch(selected_checkpoint, saved)
        write_json(folder / f'STAGE_{epoch:03d}.json', dict(completed_epoch=epoch, updates=updates,
            selected_epoch=best['epoch'], metrics={k: v for k, v in selected.items() if k != 'rows'},
            checkpoint_sha256=sha256(selected_checkpoint),
            independent_test=False, physical_drone_count_evaluated=False))
        del saved
    del dataset, net, optimizer, item, loss, wave, count_loss
    gc.collect(); torch.cuda.empty_cache()


def run(args):
    verify_freeze(args.run)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; there is no CPU-training fallback')
    torch.set_num_threads(2)
    config = json.loads((args.preparation / 'PREPARATION.json').read_text())
    preflight(args, config)
    if args.preflight_only:
        return
    fixed_mixture_diagnostic(args, config)
    wait_features(args.preparation, 'validation_pack', 1, args.run)
    validation = ContextMixtures(args.preparation, 'validation_pack', 1)
    completed = {'ordered': 0, 'mean': 0}
    progress = args.run / 'COMPLETED_EPOCHS.json'
    if progress.exists():
        completed = json.loads(progress.read_text())
    for epoch in range(1, config['epochs'] + 1):
        for arm in config['arms'] if epoch % 2 else list(reversed(config['arms'])):
            if completed[arm] >= epoch:
                continue
            verify_freeze(args.run)
            train_arm_epoch(args, config, arm, epoch, validation)
            completed[arm] = epoch
            write_json(progress, completed)
    write_json(args.run / 'COMPLETE.json', dict(status='MATCHED_CONTEXT_DEVELOPMENT_COMPLETE',
        time=time.time(), epochs_per_arm=50, updates_per_arm=3750, arms=config['arms'],
        independent_test=False, physical_aircraft_count=False, whole_record_tracking_evaluated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    os.sched_setaffinity(0, {10, 11, 12, 13})
    try:
        run(args)
    except Exception:
        write_json(args.run / 'FAILURE.json', dict(status='FAILED', time=time.time(),
            pid=os.getpid(), traceback=traceback.format_exc(), automatic_restart=False))
        raise
