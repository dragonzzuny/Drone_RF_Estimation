"""Bounded memorization diagnostic on four TRAINING mixtures, not validation.

The full mean-context U-Net starts from the preserved e22 checkpoint. Fitting
these same four examples tests optimization; it is never a generalization score.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from models import predict
from study import admitted_dataset, batch, make_model, objective, atomic_torch
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256
from drone_rf.waveform import analyze, synthesize, waveform_metrics

PREPARATION = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')
LOCK = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/drff_v91_candidates/neural_queue.lock')


def select(data):
    items, records = [], []
    for count in (2, 3):
        accepted = 0
        for index in np.flatnonzero(data.rows['count'] == count):
            item = data[int(index)]
            power = np.mean(np.abs(item['references'][:count]).astype(np.float64) ** 2, axis=-1)
            if np.any(power <= 0):
                continue
            gap = float(10 * np.log10(power.max() / power.min()))
            if gap > 10:
                continue
            row = data.rows[index]
            clips = [data.library.clips[int(i)] for i in row['indices'][:count]]
            assert all(c['role'] == 'train_pack' for c in clips)
            records.append(dict(schedule_index=int(index), count=count, power_gap_db=gap,
                clip_ids=[c['clip_id'] for c in clips], categories=[c['category'] for c in clips],
                cache_sha256=[c['cache_sha256'] for c in clips],
                crop_start=int(row['crop_start']), reference_power=power.tolist()))
            items.append(batch(item, 'cuda'))
            accepted += 1
            if accepted == 2:
                break
        if accepted != 2:
            raise RuntimeError('Insufficient eligible training mixtures')
    return items, records


@torch.no_grad()
def evaluate(net, items, step):
    net.eval()
    rows = []
    for i, item in enumerate(items):
        estimates, _ = predict(net, item)
        active = item['active'][0]
        m = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
        nmse = m['nmse'][0][active].cpu().tolist()
        si = m['si_sdr'][0][active].cpu().tolist()
        if not all(math.isfinite(v) for v in nmse + si):
            raise RuntimeError('Nonfinite fit metric')
        rows.append(dict(example=i, nmse=nmse, si_sdr=si,
                         mixture_sum_relative_error=float(m['sum_relative_error'][0])))
    errors = [v for row in rows for v in row['nmse']]
    scores = [v for row in rows for v in row['si_sdr']]
    return dict(step=step, rows=rows, mean_nmse=float(np.mean(errors)),
                worst_component_nmse=max(errors), mean_si_sdr=float(np.mean(scores)),
                weakest_si_sdr=min(scores),
                fit_target_met=all(v < .05 for v in errors) and all(v > 10 for v in scores),
                diagnostic_training_fit_only=True, independent_test=False)


def run(root):
    if any(root.iterdir()):
        raise RuntimeError('Use a fresh diagnostic directory; no silent restart')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'], text=True).splitlines()
    for line in processes:
        pid, memory = line.split(','); pid = int(pid)
        if pid == os.getpid():
            continue
        process = Path('/proc', str(pid))
        if (process.joinpath('exe').resolve() != Path('/usr/share/rustdesk/rustdesk')
                or process.stat().st_uid != os.getuid() or not 0 <= float(memory) <= 256):
            raise RuntimeError('Other GPU work active')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = admitted_dataset(PREPARATION, 'train_pack', 1)
    items, records = select(data)
    config = json.loads((PREPARATION / 'PREPARATION.json').read_text())
    files = [Path(__file__).resolve(), *HERE.glob('*.py'), *(HERE / 'vendor/drone_rf').glob('*.py')]
    protocol = dict(status='FIXED_TRAINING_FIT_DIAGNOSTIC', seed=0, max_updates=300,
        learning_rate=1e-4, weight_decay=0., effective_batch=4, microbatch=1,
        full_model=True, architecture='unet_mean', initialization='preserved same-band e22',
        initialization_sha256=config['initialization_checkpoint_sha256'],
        source_sha256={str(p): sha256(p) for p in sorted(files)},
        preparation_sha256=sha256(PREPARATION / 'PREPARATION.json'),
        examples=records, selection='first two per source count with local power gap <=10dB',
        fit_target='all ten active components NMSE<0.05 and complex SI-SDR>10dB',
        failure_interpretation='not fit under this budget; does not prove model incapacity',
        no_validation_or_heldout_iq=True, generalization_claim=False,
        original_checkpoints_untouched=True)
    write_json(root / 'PROTOCOL.json', protocol)
    for item in items:
        recovered = synthesize(analyze(item['mixture']), item['mixture'].shape[-1])
        error = float((recovered-item['mixture']).abs().square().sum()/item['mixture'].abs().square().sum())
        if error >= 1e-10:
            raise RuntimeError('STFT round-trip check failed')
    net = make_model('unet_mean', config).cuda()
    assert sum(p.numel() for p in net.parameters()) == 32142859
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0., foreach=False)
    start = time.time()
    value = evaluate(net, items, 0)
    write_json(root / 'STEP_000.json', value)
    best = value['mean_nmse']
    for step in range(1, 301):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for item in items:
            loss = objective(net, item, config)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            (loss/4).backward()
            loss_sum += float(loss.detach())/4
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0:
            progress = dict(stage='FIT_TRAINING', step=step, max_updates=300,
                            loss=loss_sum, seconds=time.time()-start, time=time.time(), pid=os.getpid())
            write_json(root / 'PROGRESS.json', progress)
            print(json.dumps(progress), flush=True)
        if step in (25, 75, 150, 300):
            value = evaluate(net, items, step)
            write_json(root / f'STEP_{step:03d}.json', value)
            if value['mean_nmse'] < best:
                best = value['mean_nmse']
                atomic_torch(root / 'BEST.pt', dict(model=net.state_dict(), step=step,
                                                   training_fit_mean_nmse=best))
    torch.cuda.synchronize()
    write_json(root / 'COMPLETE.json', dict(status='FIT_DIAGNOSTIC_COMPLETE', updates=300,
        wall_seconds=time.time()-start, final_metrics=value, generalization_claim=False))
    print('FIT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    os.nice(10)
    args.run.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.run)
    except Exception:
        write_json(args.run / 'FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
