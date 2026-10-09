"""Queued 256-update full U-Net TRAIN4 output-parameterization diagnosis.

Waits for the current registered long-context comparison to finish. Uses the
shared GPU lock and pinned display-process exception. No development or heldout
waveforms are opened, no fitted weights are reused for performance comparison.
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

import output_parameterization as candidate
import fit_diagnostic as fit
import gpu_start_guard as guard
import watch_epochs as watch
from native_data import NativeMixtures
from models import predict
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics


def register(root, dependency, cpu):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate diagnostic registration')
    parent = watch.read(dependency/'PROTOCOL.json')
    check = watch.read(cpu)
    if (check['status'] != 'PASS' or check['recorded_iq_reads'] != 0
            or not check['mapping_nonzero_head_matches_parent_exactly']):
        raise ValueError('Full CPU check missing')
    for rel, digest in check['source_sha256'].items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('CPU checked source changed')
    train = NativeMixtures(parent['preparation'], 'train_pack', 1)
    indices = [int(i) for count in (2, 3) for i in np.flatnonzero(train.rows['count'] == count)[:2]]
    source = dict(parent['source_sha256'])
    for path in (Path(__file__), Path(candidate.__file__), Path(__file__).with_name('check_output_parameterization.py')):
        source[str(path.relative_to(watch.ROOT))] = watch.digest(path)
    protocol = dict(status='REGISTERED_QUEUED_OUTPUT_FIT', source_sha256=source,
        dependency=str(dependency), dependency_protocol_sha256=watch.digest(dependency/'PROTOCOL.json'),
        cpu_check=str(cpu), cpu_check_sha256=watch.digest(cpu),
        preparation=parent['preparation'], preparation_sha256=parent['preparation_sha256'],
        gpu_display_policy=parent['gpu_display_policy'], train_indices=indices, role='train_pack',
        arms=list(candidate.MODES), parameters_per_arm=candidate.PARAMETERS,
        steps_per_arm=256, effective_batch=4, microbatch=1, seed=0,
        observed_complex_samples=63872, long_context='same20.8896ms time-mean power features in both arms',
        optimizer='fresh AdamW5e-4 wd1e-4 clip1, FP32, TF32 off',
        loss='same PIT NMSE+coherence+inactive/background plus0.1 construction-count CE',
        initialization='fresh identical parameter tensors; BOTH output heads zeroed; same mixture/4 initial predictions',
        change='mapping head times global STFT RMS vs masking head times observed complex STFT bin',
        success='finite gradients and per-count fixed TRAIN NMSE decrease; separately report <=0.1 fit target',
        purpose='optimization and output inductive-bias diagnosis, not validation or new model superiority',
        validation_read=False, heldout_read=False, weights_discarded=True,
        maximum_wait_seconds=24*3600, registered_at=time.time())
    for rel, digest in source.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Frozen source changed')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, target)
    watch.write(root/'PROTOCOL.json', protocol)
    return protocol


def verify(root, protocol):
    for rel, digest in protocol['source_sha256'].items():
        if watch.digest(watch.ROOT/rel) != digest or watch.digest(root/'source_snapshot'/rel) != digest:
            raise ValueError('Frozen diagnostic source changed: '+rel)
    for path, key in [(Path(protocol['dependency'])/'PROTOCOL.json', 'dependency_protocol_sha256'),
                      (Path(protocol['cpu_check']), 'cpu_check_sha256'),
                      (Path(protocol['preparation'])/'PREPARATION.json', 'preparation_sha256')]:
        if watch.digest(path) != protocol[key]:
            raise ValueError('Dependency changed')


@torch.no_grad()
def score(net, items):
    net.eval()
    rows = []
    for index, item in enumerate(items):
        estimates, _ = predict(net, item)
        value = waveform_metrics(estimates, item['references'], item['active'], item['mixture'])
        count = int(item['construction_count'])
        if value['sum_relative_error'].max() > 1e-9:
            raise ValueError('Mixture inconsistency')
        row = dict(case=index, count=count, nmse=value['nmse'][item['active']].tolist(),
                   si_sdr=value['si_sdr'][item['active']].tolist())
        if not np.isfinite(row['nmse']+row['si_sdr']).all():
            raise ValueError('Nonfinite fit metrics')
        rows.append(row)
    return dict(rows=rows, by_count=[dict(count=c,
        mean_nmse=float(np.mean([v for r in rows if r['count']==c for v in r['nmse']])),
        mean_si_sdr=float(np.mean([v for r in rows if r['count']==c for v in r['si_sdr']]))) for c in (2,3)])


def run(root, protocol, public):
    began = time.time()
    dependency = Path(protocol['dependency'])
    while not (dependency/'COMPLETE.json').exists():
        verify(root, protocol)
        if (dependency/'FAILURE.json').exists():
            raise RuntimeError('Prior study failed; no automatic bypass')
        if time.time()-began > protocol['maximum_wait_seconds']:
            raise TimeoutError('Registered wait exceeded')
        watch.write(root/'STATE.json', dict(status='WAITING_PHASE_COMPARISON',
                    pid=os.getpid(), model_updates=0, time=time.time()))
        time.sleep(30)
    completed = watch.read(dependency/'COMPLETE.json')
    if (completed['protocol_sha256'] != protocol['dependency_protocol_sha256']
            or completed['epochs_per_arm'] != 5 or completed['updates_per_arm'] != 375):
        raise ValueError('Prior study incomplete or changed')
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(root, protocol)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        watch.write(root/'GPU_START_CHECK.json', guard.wait_for_predecessor(
            watch.read(dependency/'STATE.json')['pid'], protocol['gpu_display_policy']))
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        train = NativeMixtures(protocol['preparation'], 'train_pack', 1)
        items = [fit.base.batch([train[i]]) for i in protocol['train_indices']]
        results = []
        for arm in candidate.MODES:
            verify(root, protocol)
            net = candidate.build(arm).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4, foreach=False)
            torch.cuda.reset_peak_memory_stats()
            history = [dict(step=0, **score(net, items))]
            started = time.time()
            for step in range(1, 257):
                net.train(); opt.zero_grad(set_to_none=True)
                for item in items:
                    estimates, logits = predict(net, item)
                    loss = pit_waveform_loss(estimates, item['references'], item['active'], item['mixture'])['loss']
                    loss = loss + .1*F.cross_entropy(logits, item['construction_count']-1)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    (loss/4).backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                if not float(norm) > 0:
                    raise ValueError('No gradient')
                opt.step()
                if step in (1, 8, 16, 32, 64, 128, 192, 256):
                    history.append(dict(step=step, **score(net, items)))
                    watch.write(root/f'{arm}_HISTORY.json', dict(history=history, partial=step<256))
                    print(json.dumps(dict(arm=arm, **history[-1])), flush=True)
                if step % 8 == 0:
                    watch.write(root/'STATE.json', dict(status='GPU_TRAIN4_FIT', arm=arm,
                        step=step, total=256, seconds=time.time()-started, pid=os.getpid(), time=time.time()))
            passed = all(b['mean_nmse'] < a['mean_nmse'] for a, b in
                         zip(history[0]['by_count'], history[-1]['by_count']))
            result = dict(arm=arm, parameters=candidate.PARAMETERS, updates=256,
                history=history, finite_gradients=True, both_counts_improved=passed,
                both_counts_nmse_le_point1=all(s['mean_nmse'] <= .1 for s in history[-1]['by_count']),
                seconds=time.time()-started, peak_bytes=torch.cuda.max_memory_allocated(), weights_discarded=True)
            results.append(result); watch.write(root/f'{arm}_COMPLETE.json', result)
            del net, opt, estimates, logits, loss, norm
            gc.collect(); torch.cuda.empty_cache()
        verify(root, protocol)
        complete = dict(status='COMPLETE', protocol_sha256=watch.digest(root/'PROTOCOL.json'),
            source_sha256=protocol['source_sha256'], results=results,
            validation_read=False, heldout_read=False, weights_discarded=True, time=time.time())
        watch.write(root/'COMPLETE.json', complete)
        watch.write(public.with_suffix('.json'), complete)
        lines = ['# 출력 표현별 원 규모 U-Net TRAIN4 적합 진단', '',
            '같은 TRAIN4·초기값·전체 입력·각 256업데이트다. 두 군 모두 출력층을 0으로 초기화했다. '
            '가중치는 본학습에 재사용하지 않는다. 아래 수치는 고정 학습 사례의 적합도이며 일반화 성과가 아니다.', '',
            '| 군 | 단계 | 2성분 NMSE ↓ | 3성분 NMSE ↓ | 2성분 복소 SI-SDR ↑ dB | 3성분 복소 SI-SDR ↑ dB |',
            '|---|---:|---:|---:|---:|---:|']
        for r in results:
            for h in r['history']:
                a, b = h['by_count']
                lines.append(f"|{r['arm']}|{h['step']}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|")
        watch.write(public.with_suffix('.md'), '\n'.join(lines)+'\n')
        watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'dependency', 'cpu-check', 'public'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args(); root = a.run.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protocol = register(root, a.dependency.resolve(), a.cpu_check.resolve())
            run(root, protocol, a.public.resolve())
        except Exception:
            watch.write(root/'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
            watch.write(root/'STATE.json', dict(status='FAILED', pid=os.getpid(), time=time.time()))
            raise
