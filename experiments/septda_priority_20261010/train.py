"""Run only the existing SepTDA trajectory after an audited epoch boundary.

Training math, data, optimizer/RNG resume and the original protocol stay fixed.
The user changed execution order; the common control is deferred and reusable.
"""
import argparse
import fcntl
import gc
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('priority_continuation', ROOT / 'experiments/septda_continuation_20261010/train.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
w = c.w
PUBLIC = ROOT / 'reports/2026-10-10'


def pending_epochs(root, maximum):
    complete = sorted(int(p.stem.split('_')[1]) for p in (root / 'septda').glob('EPOCH_*.json'))
    assert complete and complete == list(range(1, max(complete) + 1)), 'Noncontiguous receipts'
    assert max(complete) <= maximum
    return list(range(max(complete) + 1, maximum + 1))


def verify(root, protocol, amendment):
    c.verify(root, protocol)
    assert amendment['original_protocol_sha256'] == w.digest(root / 'PROTOCOL.json')
    assert amendment['active_arms'] == ['septda'] and amendment['max_epochs'] == protocol['max_epochs'] == 50
    for path, sha in amendment['source_sha256'].items():
        assert w.digest(ROOT / path) == w.digest(root / 'priority_snapshot' / path) == sha
    assert w.digest(PUBLIC / 'SEPTDA_FIRST_PLAN_KO.md') == amendment['plan_sha256']


def progress(root, protocol):
    old = protocol['original_protocol']['original_protocol']
    _, identities = w.validate(Path(old['baseline']), None)
    events = []
    for path in sorted((root / 'septda').glob('EPOCH_*.json')):
        event = w.read(path)
        epoch = event['epoch']
        actual, _ = w.validate(root / 'septda' / f'VALIDATION_{epoch:03d}.json', identities)
        assert actual == event['validation']
        events.append(dict(epoch=epoch, updates=event['updates'], best=event['best'],
                           validation=actual, mean_training_loss=event['mean_training_loss']))
    assert [r['epoch'] for r in events] == list(range(1, len(events)+1))
    control_epochs = sorted(int(p.stem.split('_')[1]) for p in (root/'control').glob('EPOCH_*.json'))
    result = dict(status='COMPLETE' if len(events)==50 else 'PARTIAL', active_arm='septda',
        completed_epochs=len(events), max_epochs=50, events=events,
        full_fixed_set_passes=len(events)/5, control_completed_epochs=control_epochs,
        control_additional_training='DEFERRED_BY_USER', paired_results_available_through=max(control_epochs),
        protocol_sha256=w.digest(root/'PROTOCOL.json'), amendment_sha256=w.digest(root/'EXECUTION_PRIORITY.json'),
        heldout_read=False, time=time.time())
    w.write(PUBLIC/'SEPTDA_PRIORITY_PROGRESS.json', result)
    lines = ['# SepTDA 참고 후보 우선 학습', '',
        '사용자 지시에 따라 SepTDA 후보만 총50구간까지 먼저 학습한다. 기존 학습 상태를 이어가며 '
        '구조·자료·손실·학습률 규칙은 같다. 대조의 추가 학습은 보류하고 저장된 공통 대조를 재사용한다. '
        '1구간=2,400예제·75업데이트,5구간=고정12,000합성의1순회다.', '',
        '| 완료 구간 | 업데이트 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ | 선택 구간 |',
        '|---|---:|---|---|---|---:|']
    for event in events:
        a,b=event['validation']['by_count'][1:]
        lines.append(f"|{event['epoch']}|{event['updates']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
            f"{w.fmt(a['mean_si_sdr'],3)}/{w.fmt(b['mean_si_sdr'],3)}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{event['best']['epoch']}|")
    lines += ['', f'대조 결과가 존재하는 마지막 구간: {max(control_epochs)}. 이후 후보 결과를 같은 학습량 대조보다 우수하다고 표시하지 않는다.',
        '', '[실행 변경 규약](SEPTDA_FIRST_PLAN_KO.md) · [후보5개](CANDIDATE_SHORTLIST_KO.md)']
    w.write(PUBLIC/'SEPTDA_PRIORITY_PROGRESS_KO.md', '\n'.join(lines)+'\n')


def run(root):
    protocol = w.read(root/'PROTOCOL.json')
    amendment = w.read(root/'EXECUTION_PRIORITY.json')
    with (root/'.run.lock').open('a') as local_lock:
        fcntl.flock(local_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with c.s.worker.fit.base.LOCK.open('r') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            verify(root, protocol, amendment)
            transition = w.read(root/'PRIORITY_HANDOFF.json')
            assert transition['status']=='OLD_WORKER_STOPPED_AFTER_SAVED_EPOCH'
            assert torch.cuda.is_available()
            old = protocol['original_protocol']['original_protocol']
            c.s.worker.guard.wait_for_predecessor(amendment['predecessor_pid'], old['gpu_display_policy'])
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            _, ids = w.validate(Path(old['baseline']), None)
            dev = c.s.worker.NativeMixtures(old['preparation'], 'validation_pack', 1)
            progress(root, protocol)
            for epoch in pending_epochs(root, 50):
                verify(root, protocol, amendment)
                train = c.s.worker.NativeMixtures(old['preparation'], 'train_pack', (epoch+1)%5+1)
                assert len(train)==2400
                c.epoch_run(root, protocol, 'septda', epoch, train, dev, ids)
                del train
                gc.collect(); torch.cuda.empty_cache()
                progress(root, protocol)
            verify(root, protocol, amendment)
            w.write(root/'SEPTDA_PRIORITY_COMPLETE.json',dict(status='SEPTDA_BUDGET_COMPLETE',
                epochs=50,updates=3750,control_complete=False,convergence_claimed=False,
                amendment_sha256=w.digest(root/'EXECUTION_PRIORITY.json'),time=time.time()))
            w.write(root/'STATE.json',dict(status='SEPTDA_BUDGET_COMPLETE',arm='septda',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    try:
        run(root)
    except Exception:
        w.write(root/'PRIORITY_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
