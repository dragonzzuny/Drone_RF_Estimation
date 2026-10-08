"""Matched full-size GPU study using the frozen SAME-BAND RFUAV development set.

No automatic queue attachment, no CPU-training fallback, no held-out access.
Resume only an identical frozen protocol; never consume the existing 50-epoch
weights as a head start for one arm. --epochs is a matched stopping milestone.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from models import ARMS, build, predict
from drone_rf.context_training_data import ContextMixtures, write_json
from drone_rf.data import sha256
from drone_rf.losses import pit_waveform_loss
from drone_rf.mixture_constraints import validate_sources
from drone_rf.waveform import waveform_metrics
from dense_data import DenseContextMixtures


HERE = Path(__file__).resolve().parent
DEFAULT_PREPARATION = Path('/home/pyj/문서/GitHub/Drone_RF_Estimation/local/within_dataset_context_20261007_v1/preparation')


def batch(item, device):
    keys = ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')
    return {key: torch.as_tensor(item[key], device=device)[None] for key in keys}


def admitted_dataset(preparation, role, epoch=1):
    config = json.loads((preparation / 'PREPARATION.json').read_text())
    expected = dict(dataset='RFUAV', cross_dataset_synthesis=False, cross_band_synthesis=False,
                    native_center_offsets_preserved=False, controller_classes_excluded=True,
                    fs_hz=100000000, window_samples=63872, context_samples=2097152)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Not the approved same-band, center-aligned protocol')
    loader = DenseContextMixtures if config['status'] == 'DENSE_GATED_COMPARISON_ADMITTED' else ContextMixtures
    data = loader(preparation, role, epoch)
    for row in data.rows:
        validate_sources([dict(data.library.clips[int(i)], dataset='RFUAV')
                          for i in row['indices'][:int(row['count'])]])
    return data


def make_model(arm, config):
    net = build(arm)
    if 'initialization_checkpoint' in config:
        if arm not in ('unet_mean', 'unet_gated'):
            raise ValueError('Warm-start comparison is restricted to the two U-Nets')
        path = Path(config['initialization_checkpoint'])
        if sha256(path) != config['initialization_checkpoint_sha256']:
            raise ValueError('Shared initialization checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['best']['epoch'] != config['initialization_selected_epoch']:
            raise ValueError('Unexpected initialization epoch')
        result = net.load_state_dict(saved['model'], strict=False)
        expected_missing = {k for k in net.state_dict() if k.startswith('gates.')}
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError('Incomplete shared backbone initialization')
    return net


def objective(net, item, config):
    estimates, logits = predict(net, item)
    wave = pit_waveform_loss(estimates, item['references'], item['active'], item['mixture'])['loss']
    count = F.cross_entropy(logits, item['construction_count'] - 1)
    return wave + config['construction_count_loss_weight'] * count


def atomic_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def finite_values(tensor):
    # Undefined/infinite SI-SDR never becomes a plausible finite score.
    values = tensor.detach().cpu().tolist()
    return [float(x) if np.isfinite(x) else None for x in values]


def value_status(tensor):
    return ['finite' if np.isfinite(x) else 'undefined' if np.isnan(x) else
            'positive_infinity' if x > 0 else 'negative_infinity'
            for x in tensor.detach().cpu().tolist()]


@torch.no_grad()
def evaluate(net, dataset, path, epoch):
    net.eval()
    rows = []
    for index in range(len(dataset)):
        item = batch(dataset[index], 'cuda')
        estimates, logits = predict(net, item)
        metrics = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
        active = item['active'][0]
        count = int(item['construction_count'][0])
        power = metrics['reference_power'][0][active]
        si = metrics['si_sdr'][0][active]
        gain = si - metrics['input_si_sdr'][0][active]
        source_row = dataset.rows[index]
        clips = [dataset.library.clips[int(i)] for i in source_row['indices'][:count]]
        weakest = int(power.argmin())
        rows.append(dict(index=index, count=count, categories=[c['category'] for c in clips],
            pack_ids=[c['pack_id'] for c in clips], nominal_levels_db=source_row['levels'][:count].tolist(),
            reference_power=power.cpu().tolist(), assignment=metrics['assignment'][0].cpu().tolist(),
            nmse=finite_values(metrics['nmse'][0][active]), si_sdr=finite_values(si),
            si_status=value_status(si), input_si_sdr=finite_values(metrics['input_si_sdr'][0][active]),
            si_sdr_gain=finite_values(gain), gain_status=value_status(gain), weakest_index=weakest,
            all_sources_gain_positive=bool(torch.all(torch.isfinite(gain) & (gain > 0))) if count > 1 else None,
            predicted_count=int(logits.argmax(-1)[0]) + 1,
            inactive_leak=float(metrics['inactive_leak'][0].sum()),
            background_nmse=float(metrics['background_nmse'][0]),
            sum_relative_error=float(metrics['sum_relative_error'][0])))
        if index % 50 == 0:
            write_json(path.parent.parent / 'PROGRESS.json', dict(stage='VALIDATION', arm=path.parent.name,
                epoch=epoch, examples=index + 1, total=len(dataset), time=time.time()))
    groups = []
    for count in (1, 2, 3):
        group = [r for r in rows if r['count'] == count]
        if len(group) != 210:
            raise ValueError('Unexpected validation stratum size')
        if any(x is None for r in group for x in r['nmse']):
            raise ValueError('Nonfinite validation NMSE')
        def aggregate(key):
            values = [x for r in group for x in r[key]]
            return float(np.mean(values)) if all(x is not None for x in values) else None
        groups.append(dict(count=count, cases=len(group), mean_nmse=aggregate('nmse'),
            mean_si_sdr=aggregate('si_sdr'), mean_si_sdr_gain=aggregate('si_sdr_gain') if count > 1 else None,
            nonfinite_si_sdr=sum(x is None for r in group for x in r['si_sdr']),
            nonfinite_si_sdr_gain=sum(x is None for r in group for x in r['si_sdr_gain']),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            all_sources_gain_positive=float(np.mean([r['all_sources_gain_positive'] for r in group])) if count > 1 else None,
            construction_count_accuracy=float(np.mean([r['count'] == r['predicted_count'] for r in group]))))
    result = dict(epoch=epoch, by_count=groups,
        selection_nmse=float(np.mean([r['mean_nmse'] for r in groups])),
        two_three_nmse=float(np.mean([r['mean_nmse'] for r in groups if r['count'] > 1])),
        independent_test=False, physical_aircraft_count=False, whole_record_tracking=False, rows=rows)
    write_json(path, result)
    return result


def freeze(preparation, root):
    config = json.loads((preparation / 'PREPARATION.json').read_text())
    if config['effective_batch'] != 32 or config['examples_per_epoch'] != 2400:
        raise ValueError('Matched update budget changed')
    code = sorted(HERE.glob('*.py')) + sorted((HERE / 'vendor/drone_rf').glob('*.py'))
    files = {str(p.relative_to(HERE)): sha256(p) for p in code}
    data_files = [preparation / name for name in ('PREPARATION.json', 'TRAIN.npy', 'VALIDATION.npy')]
    data_files += [Path(config['manifest'])]
    arms = tuple(config.get('comparison_arms', ARMS))
    if 'initialization_checkpoint' in config:
        data_files.append(Path(config['initialization_checkpoint']))
        if sha256(data_files[-1]) != config['initialization_checkpoint_sha256']:
            raise ValueError('Warm-start checkpoint changed')
    value = dict(arms=list(arms), seed=0, source_files=files,
        dataset_files={str(p): sha256(p) for p in data_files}, optimizer=config['optimizer'],
        parameters_matched=False, updates_matched=True,
        initialization=config.get('initialization', 'seed0; shared UNet tensors identical'),
        selection='minimum validation macro NMSE over counts 1,2,3, including epoch0; report 2/3 separately',
        milestones=config.get('checkpoints', [5, 25, 50]), effective_batch=32, updates_per_epoch=75,
        sample_rate_hz=100000000, native_center_offsets_preserved=False,
        dense_corpus_used=config.get('dense_corpus_used', False))
    path = root / 'FREEZE.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Code/data/protocol changed; use a NEW run directory')
    if not path.exists():
        write_json(path, value)
    return config, sha256(path)


def preflight(arm, train, config, folder, frozen_hash):
    path = folder / 'GPU_PREFLIGHT.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['freeze_sha256'] != frozen_hash or previous['status'] != 'PASS':
            raise ValueError('Invalid GPU preflight')
        return
    net = make_model(arm, config).cuda().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], foreach=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses = []
    for count in (1, 2, 3):
        index = int(np.flatnonzero(train.rows['count'] == count)[0])
        loss = objective(net, batch(train[index], 'cuda'), config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite GPU preflight loss')
        (loss / 3).backward()
        losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    if not float(norm) > 0:
        raise ValueError('No training gradient')
    optimizer.step()
    torch.cuda.synchronize()
    write_json(path, dict(status='PASS', freeze_sha256=frozen_hash, parameters=sum(p.numel() for p in net.parameters()),
        length=63872, source_counts=[1, 2, 3], losses=losses, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.time() - started, discarded_updates=1, precision='float32'))
    del net, optimizer, loss
    gc.collect()
    torch.cuda.empty_cache()


def train_epoch(arm, epoch, train, validation, config, root, frozen_hash):
    folder = root / arm
    net = make_model(arm, config).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], foreach=False)
    last = folder / 'LAST.pt'
    updates, best = 0, None
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['freeze_sha256'] != frozen_hash:
            raise ValueError('Checkpoint freeze mismatch')
        if saved['epoch'] >= epoch:
            if not (folder / f'EPOCH_{epoch:03d}.json').exists():
                raise ValueError('Incomplete epoch receipt; explicit recovery required')
            if epoch in config['checkpoints'] and not (folder / f'SELECTED_{epoch:03d}.pt').exists():
                raise ValueError('Incomplete milestone checkpoint; explicit recovery required')
            return
        if saved['epoch'] != epoch - 1:
            raise ValueError('Nonconsecutive resume')
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        updates, best = saved['updates'], saved['best']
        best_saved = torch.load(folder / 'BEST.pt', map_location='cpu', weights_only=False)
        if best_saved['best'] != best or best_saved['freeze_sha256'] != frozen_hash:
            raise ValueError('Partial checkpoint transaction; explicit recovery required')
        del best_saved
        del saved
    elif epoch != 1:
        raise ValueError('Missing resume checkpoint')
    else:
        initial = evaluate(net, validation, folder / 'VALIDATION_000.json', 0)
        best = dict(epoch=0, metric=initial['selection_nmse'])
        atomic_torch(folder / 'BEST.pt', dict(model=net.state_dict(), best=best, freeze_sha256=frozen_hash))
    net.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    total = 0.
    if len(train) != 2400:
        raise ValueError('Training schedule changed')
    for index in range(len(train)):
        loss = objective(net, batch(train[index], 'cuda'), config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        (loss / 32).backward()
        total += float(loss.detach())
        if (index + 1) % 32 == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 5 == 0:
                write_json(root / 'PROGRESS.json', dict(stage='TRAIN', arm=arm, epoch=epoch,
                    updates=updates, examples=index + 1, time=time.time(), pid=os.getpid()))
    torch.cuda.synchronize()
    train_seconds = time.time() - started
    peak = torch.cuda.max_memory_allocated()
    metrics = evaluate(net, validation, folder / f'VALIDATION_{epoch:03d}.json', epoch)
    if metrics['selection_nmse'] < best['metric']:
        best = dict(epoch=epoch, metric=metrics['selection_nmse'])
        atomic_torch(folder / 'BEST.pt', dict(model=net.state_dict(), best=best, freeze_sha256=frozen_hash))
    atomic_torch(last, dict(model=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
        updates=updates, best=best, freeze_sha256=frozen_hash,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    write_json(folder / f'EPOCH_{epoch:03d}.json', dict(epoch=epoch, updates=updates,
        parameters=sum(p.numel() for p in net.parameters()), train_seconds=train_seconds,
        peak_allocated_bytes=peak, average_loss=total / len(train), best=best,
        metrics={k: v for k, v in metrics.items() if k != 'rows'}, freeze_sha256=frozen_hash))
    if epoch in config['checkpoints']:
        saved = torch.load(folder / 'BEST.pt', map_location='cpu', weights_only=False)
        atomic_torch(folder / f'SELECTED_{epoch:03d}.pt', saved)
    del net, optimizer, loss
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    args.run.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        write_json(args.run / 'BLOCKED_GPU.json', dict(status='NOT_STARTED_CUDA_UNAVAILABLE',
            time=time.time(), torch_version=torch.__version__, cuda_devices=torch.cuda.device_count(),
            cpu_training_fallback=False, source_data_read=False))
        return 2
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, frozen_hash = freeze(args.preparation, args.run)
    arms = tuple(config.get('comparison_arms', ARMS))
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('Invalid comparison arms')
    if args.epochs > config['epochs']:
        raise ValueError('Requested epochs exceed the prepared matched budget')
    train = admitted_dataset(args.preparation, 'train_pack', 1)
    for arm in arms:
        folder = args.run / arm
        folder.mkdir(exist_ok=True)
        preflight(arm, train, config, folder, frozen_hash)
    del train
    if args.preflight_only:
        return 0
    validation = admitted_dataset(args.preparation, 'validation_pack', 1)
    for epoch in range(1, args.epochs + 1):
        # Recheck code and frozen source lists before each matched epoch.
        freeze(args.preparation, args.run)
        train = admitted_dataset(args.preparation, 'train_pack', epoch)
        for arm in arms if epoch % 2 else tuple(reversed(arms)):
            train_epoch(arm, epoch, train, validation, config, args.run, frozen_hash)
        del train
        write_json(args.run / 'MATCHED_PROGRESS.json', dict(completed_epochs=epoch,
            updates_per_arm=75 * epoch, arms=list(arms), time=time.time()))
    write_json(args.run / f'MILESTONE_{args.epochs:03d}.json', dict(status='MATCHED_DEVELOPMENT_COMPLETE',
        epochs=args.epochs, updates_per_arm=75 * args.epochs, arms=list(arms),
        freeze_sha256=frozen_hash, independent_test=False, physical_aircraft_count=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--epochs', type=int, choices=(1, 5, 25, 50), default=5)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[-2:]))
    os.nice(10)
    args.run.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers to the same new experiment only. This does not
    # coordinate with, stop, or change any existing host GPU queue.
    with (args.run / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(args))
