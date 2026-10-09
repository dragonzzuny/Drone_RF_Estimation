"""Read-only epoch observer for the registered long-complex-context comparison.

Uses completed validation receipts only. Does not open I/Q, change workers,
load checkpoint tensors, or send chat messages.
"""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import watch_epochs as watch


def snapshot(root):
    plan = watch.read(root / 'PROTOCOL.json')
    protocol_sha = watch.digest(root / 'PROTOCOL.json')
    for rel, expected in plan['source_sha256'].items():
        if (watch.digest(watch.ROOT / rel) != expected or
                watch.digest(root / 'source_snapshot' / rel) != expected):
            raise ValueError('Frozen training source changed: ' + rel)
    template = Path(plan['validation_identity_template'])
    if watch.digest(template) != plan['validation_identity_sha256']:
        raise ValueError('Validation identity template changed')
    _, identities = watch.validate(template, None)
    histories, events = {}, []
    for arm in plan['arms']:
        history = []
        for epoch in range(plan['epochs_per_arm'] + 1):
            folder = root / arm
            path = folder / f'VALIDATION_{epoch:03d}.json'
            receipt = folder / f'EPOCH_{epoch:03d}.json'
            if not path.exists() or (epoch and not receipt.exists()):
                break
            metrics, _ = watch.validate(path, identities)
            if metrics['epoch'] != epoch:
                raise ValueError('Validation epoch mismatch')
            history.append(metrics)
            if epoch:
                value = watch.read(receipt)
                if (value['protocol_sha256'] != protocol_sha or value['arm'] != arm
                        or value['epoch'] != epoch or value['updates'] != 75 * epoch):
                    raise ValueError('Epoch budget/identity mismatch')
                best = min(history, key=lambda h: h['selection_nmse'])
                if (value['best']['epoch'] != best['epoch'] or
                        not watch.close(value['best']['metric'], best['selection_nmse'])):
                    raise ValueError('Checkpoint selection mismatch')
                for group in metrics['by_count']:
                    stored = next(g for g in value['validation']['by_count']
                                  if g['count'] == group['count'])
                    if any(not watch.close(v, stored[k]) for k, v in group.items()):
                        raise ValueError('Receipt and validation aggregate differ')
                for name in (f'ACTUAL_{epoch:03d}.pt', f'SELECTED_{epoch:03d}.pt'):
                    if not (folder / name).is_file():
                        raise ValueError('Completed checkpoint missing: ' + name)
                events.append(dict(arm=arm, epoch=epoch, updates=value['updates'],
                    train_seconds=value['train_seconds'], peak_bytes=value['peak_bytes'],
                    selected_epoch=best['epoch'], metrics=metrics,
                    receipt_sha256=watch.digest(receipt), validation_sha256=watch.digest(path),
                    completed_time=receipt.stat().st_mtime))
        histories[arm] = history
    events.sort(key=lambda e: e['completed_time'])
    state = watch.read(root / 'STATE.json')
    try:
        pid = int(state['pid'])
        proc = Path(f'/proc/{pid}')
        alive = (b'phase_comparison_v2.py' in (proc / 'cmdline').read_bytes()
                 and (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z')
    except (FileNotFoundError, KeyError):
        alive = False
    status = ('FAILED' if (root / 'FAILURE.json').exists() else
              'COMPLETED' if (root / 'COMPLETE.json').exists() else
              state['status'] if alive else 'WORKER_NOT_RUNNING')
    if status == 'COMPLETED':
        done = watch.read(root / 'COMPLETE.json')
        if (len(events) != 10 or done['protocol_sha256'] != protocol_sha or
                done['epochs_per_arm'] != 5 or done['updates_per_arm'] != 375):
            raise ValueError('Completion receipt/budget mismatch')
    common = max(0, min(len(h)-1 for h in histories.values()))
    return dict(status=status, worker_alive=alive, state=state, events=events,
        histories=histories, common_epoch=common, protocol_sha256=protocol_sha,
        common_selected={arm: min(h[:common+1], key=lambda x: x['selection_nmse'])
                         for arm, h in histories.items()} if common else {},
        parameters=plan['arms'], heldout_read=False, independent_test=False,
        checkpoint_tensors_audited=False, chat_notification_service=False,
        observer_source_sha256=watch.digest(Path(__file__)),
        reported_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'))


def markdown(value):
    state = value['state']
    lines = ['# 긴 복소 I/Q 문맥: epoch별 진행', '',
        f"확인: {value['reported_at']} / 상태: {value['status']} / 프로세스 살아 있음: {value['worker_alive']}", '',
        f"현재 저장 상태: {state.get('arm', '—')}, epoch {state.get('epoch', '—')}, "
        f"{state.get('examples', '—')}/{state.get('total', '—')}개.", '',
        'local/long 모두4,641,795파라미터·새 seed0 초기화·같은RFUAV 원대역 자료·'
        '각5epoch/375업데이트다. 차이는0.63872ms 복원 구간 밖 복소 표본의 가림 여부이며 '
        '두 군 모두20.8896ms 전체RMS와 시간평균 전력 특징을 받는다.', '',
        '|군|완료 epoch|NMSE 2/3↓|복소 SI-SDR 2/3↑ dB|약신호 NMSE 2/3↓|선택 epoch|',
        '|---|---:|---|---|---|---:|']
    for event in value['events']:
        a, b = event['metrics']['by_count'][1:]
        pair = lambda key: f"{watch.fmt(a[key])}/{watch.fmt(b[key])}"
        lines.append(f"|{event['arm']}|{event['epoch']}|{pair('mean_nmse')}|"
                     f"{pair('mean_si_sdr')}|{pair('weakest_nmse')}|{event['selected_epoch']}|")
    lines += ['', f"두 군의 공통 완료 예산: {value['common_epoch']}epoch.",
        '완료 검증630행·집계·자료 정체성·선택 규칙·학습 소스 해시를 확인한다. '
        '가중치 파일 존재를 확인하지만 이 감시기에서 가중치 텐서 검산은 수행하지 않는다. '
        '미개봉 확인자료·새 I/Q를 읽지 않으며 물리 드론 대수·독립 시험·완전 수렴의 근거가 아니다. '
        '이 파일은 로컬 보고이며 자동 채팅 발송 서비스가 아니다.', '']
    return '\n'.join(lines)


def run(root, output, once):
    folder = root / 'epoch_observer'
    folder.mkdir(exist_ok=True)
    with (folder / 'LOCK').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = {p.name: watch.digest(p) for p in (Path(__file__), Path(watch.__file__))}
        for p in (Path(__file__), Path(watch.__file__)):
            dest = folder / p.name
            if dest.exists() and watch.digest(dest) != expected[p.name]:
                raise ValueError('Observer snapshot changed')
            if not dest.exists():
                shutil.copyfile(p, dest)
        seen_path = folder / 'SEEN.json'
        seen = watch.read(seen_path) if seen_path.exists() else {}
        while True:
            for p in (Path(__file__), Path(watch.__file__)):
                if watch.digest(p) != expected[p.name]:
                    raise ValueError('Live observer source changed')
            value = snapshot(root)
            for event in value['events']:
                key = f"{event['arm']}_e{event['epoch']:03d}"
                if key in seen and seen[key] != event['receipt_sha256']:
                    raise ValueError('Already reported epoch changed')
                if key not in seen:
                    watch.write(folder / 'events' / (key + '.json'), event)
                    print(f"EPOCH_COMPLETE {key}", flush=True)
                    seen[key] = event['receipt_sha256']
            watch.write(seen_path, seen)
            watch.write(output.with_suffix('.json'), value)
            watch.write(output.with_suffix('.md'), markdown(value))
            if once or value['status'] in ('FAILED', 'COMPLETED', 'WORKER_NOT_RUNNING'):
                print(value['status'], flush=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    try:
        run(args.run.resolve(), args.output.resolve(), args.once)
    except Exception as exc:
        watch.write(args.output.with_name(args.output.name + '_OBSERVER_FAILURE.json'),
                    dict(error=repr(exc), time=time.time(), pid=os.getpid()))
        raise
