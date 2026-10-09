"""Read-only training observer; audit completed epochs and retain Korean reports.

Never starts/stops training, opens held-out data, or sends chat notifications.
The terminal emits one event per completed arm/epoch. Its local status report
also records process liveness, including gaps while training validates/saves.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'wavenet_cycle10': '기본 WaveNet',
         'wavenet_cycle15': '긴 수용범위 WaveNet', 'unet_mean': 'STFT U-Net'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(value if isinstance(value, str) else
                    json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def close(a, b):
    return a is None and b is None or (
        a is not None and b is not None and
        math.isfinite(a) and math.isfinite(b) and
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10))


def validate(path, identities):
    value = read(path)
    rows = sorted(value['rows'], key=lambda r: r['index'])
    if len(rows) != 630 or [r['index'] for r in rows] != list(range(630)):
        raise ValueError(f'Missing/duplicate validation cases: {path}')
    found = []
    for row in rows:
        if len(row['nmse']) != row['count'] or len(row['si_sdr']) != row['count']:
            raise ValueError('Wrong source count in metrics')
        if any(not math.isfinite(x) or x < 0 for x in row['nmse']):
            raise ValueError('Invalid NMSE')
        if not math.isfinite(row['sum_relative_error']) or row['sum_relative_error'] > 1e-9:
            raise ValueError('Mixture consistency failed')
        found.append({k: row[k] for k in ('index', 'count', 'categories', 'pack_ids',
                                          'nominal_levels_db', 'reference_power')})
    if identities is not None and found != identities:
        raise ValueError('Validation mixtures changed')
    groups = []
    for count in (1, 2, 3):
        subset = [r for r in rows if r['count'] == count]
        if len(subset) != 210:
            raise ValueError('Validation balance changed')
        si = [x for r in subset for x in r['si_sdr']]
        group = dict(count=count, cases=210,
            mean_nmse=statistics.mean(x for r in subset for x in r['nmse']),
            mean_si_sdr=statistics.mean(si) if all(x is not None for x in si) else None,
            nonfinite_si_sdr=sum(x is None for x in si),
            weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in subset),
            construction_count_accuracy=statistics.mean(r['predicted_count'] == count for r in subset))
        stored = next(x for x in value['by_count'] if x['count'] == count)
        if any(not close(v, stored[k]) for k, v in group.items()):
            raise ValueError('Saved metrics disagree with the 630 rows')
        groups.append(group)
    metric = statistics.mean(g['mean_nmse'] for g in groups[1:])
    if not close(metric, value['selection_nmse']):
        raise ValueError('Selection metric mismatch')
    return dict(epoch=value['epoch'], by_count=groups, selection_nmse=metric), found


def live_process(state):
    pid = state.get('pid')
    if not pid:
        return False
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        return raw.rsplit(')', 1)[1].split()[0] != 'Z' and b'train_comparison.py' in cmd
    except FileNotFoundError:
        return False


def snapshot(run):
    plan = read(run / 'PROTOCOL.json')
    sha = digest(run / 'PROTOCOL.json')
    for rel, expected in plan['source_sha256'].items():
        if digest(ROOT / rel) != expected or digest(run / 'source_snapshot' / rel) != expected:
            raise ValueError(f'Frozen source mismatch: {rel}')
    if digest(Path(plan['preparation']) / 'PREPARATION.json') != plan['preparation_sha256']:
        raise ValueError('Preparation manifest changed')
    histories, identities, events = {}, None, []
    for arm in plan['arms']:
        history = []
        folder = run / arm
        initial = folder / 'VALIDATION_000.json'
        if initial.exists():
            metrics, identities = validate(initial, identities)
            history.append(metrics)
        for epoch in range(1, plan['epochs_per_arm'] + 1):
            receipt = folder / f'EPOCH_{epoch:03d}.json'
            if not receipt.exists():
                break
            value = read(receipt)
            if value['protocol_sha256'] != sha or value['arm'] != arm or value['epoch'] != epoch:
                raise ValueError('Epoch identity mismatch')
            if value['updates'] != 75 * epoch:
                raise ValueError('Update budget mismatch')
            validation = folder / f'VALIDATION_{epoch:03d}.json'
            metrics, identities = validate(validation, identities)
            history.append(metrics)
            best = min(history, key=lambda x: x['selection_nmse'])
            if value['best']['epoch'] != best['epoch'] or not close(value['best']['metric'], best['selection_nmse']):
                raise ValueError('Checkpoint selection mismatch')
            events.append(dict(arm=arm, epoch=epoch, updates=value['updates'],
                receipt_sha256=digest(receipt), validation_sha256=digest(validation),
                completed_time=receipt.stat().st_mtime, train_seconds=value['train_seconds'],
                selected_epoch=best['epoch'], metrics=metrics))
        histories[arm] = history
    events.sort(key=lambda x: x['completed_time'])
    state = read(run / 'STATE.json')
    alive = live_process(state)
    status = ('FAILED' if (run / 'FAILURE.json').exists() else
              'COMPLETED' if (run / 'COMPLETE.json').exists() else
              'WORKER_ALIVE' if alive else 'WORKER_NOT_RUNNING')
    if status == 'COMPLETED' and len(events) != len(plan['arms']) * plan['epochs_per_arm']:
        raise ValueError('Completion missing epoch receipts')
    common = min((len(h) - 1 if h else 0 for h in histories.values()), default=0)
    return dict(status=status, worker_alive=alive, state=state,
        reported_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
        protocol_sha256=sha, events=events, histories=histories, common_epoch=common,
        common_selected={arm: min(h[:common + 1], key=lambda x: x['selection_nmse'])
                         for arm, h in histories.items()} if common else {},
        independent_test=False, heldout_read=False, chat_notification_service=False)


def fmt(value, digits=4):
    return '미정의' if value is None else f'{value:.{digits}f}'


def markdown(report):
    state = report['state']
    lines = ['# 구조 비교: epoch별 진행 보고', '',
        f"확인 시각: {report['reported_at']} / 상태: {report['status']} / "
        f"학습 프로세스 살아 있음: {report['worker_alive']}", '',
        f"마지막 학습 저장 상태: {NAMES.get(state.get('arm'), state.get('arm', '—'))}, "
        f"epoch {state.get('epoch', '—')}, {state.get('examples', '—')}/{state.get('total', '—')}개. "
        '검증·가중치 저장 중에는 학습 진행값이 이전 단계에 머물 수 있다.', '',
        'RFUAV 동일 대역·원 수신 중심 간격 유지. 새 초기화 seed 0, 각 5 epoch/375업데이트, '
        '매 epoch 학습 2,400혼합·검증 630혼합. 모델 규모·계산량은 서로 다르다. '
        '아래는 완료된 epoch의 현재 가중치 점수이며, 선택 가중치 epoch를 별도로 표시한다.', '',
        '| 모델 | 완료 epoch | 업데이트 | NMSE 2개 ↓ | NMSE 3개 ↓ | SI-SDR 2개 ↑ dB | SI-SDR 3개 ↑ dB | 약신호 NMSE 2/3 ↓ | 선택 epoch |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for event in report['events']:
        a, b = event['metrics']['by_count'][1:]
        lines.append(f"| {NAMES[event['arm']]} | {event['epoch']} | {event['updates']} | "
            f"{fmt(a['mean_nmse'])} | {fmt(b['mean_nmse'])} | {fmt(a['mean_si_sdr'],3)} | "
            f"{fmt(b['mean_si_sdr'],3)} | {fmt(a['weakest_nmse'])}/{fmt(b['weakest_nmse'])} | {event['selected_epoch']} |")
    lines += ['', f"세 모델 모두 완료한 공통 예산: {report['common_epoch']} epoch.",
        '630개 결과·조건별 집계·선택 규칙·동결 소스·준비 명세 해시를 확인했다. '
        'NMSE는 정확도 백분율이 아니다. 한 seed의 반복 개발 검증이며 수렴한 모델 계열의 우열이나 '
        '미학습 기종의 성능을 확정하지 않는다. 보류 확인 자료는 읽지 않았다.', '',
        '이 감시기는 로컬 보고서를 갱신하고 완료 이벤트를 기록한다. 채팅 메시지 자체를 자동 발송하는 서비스는 아니다.', '']
    return '\n'.join(lines)


def run_watch(run, output, interval, once):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'WATCH_LOCK'
    import fcntl
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen_path = output / 'SEEN.json'
        seen = read(seen_path) if seen_path.exists() else {}
        while True:
            report = snapshot(run)
            for event in report['events']:
                key = f"{event['arm']}_e{event['epoch']:03d}"
                if key in seen and seen[key] != event['receipt_sha256']:
                    raise ValueError('Previously reported epoch changed')
                if key not in seen:
                    write(output / 'events' / f'{key}.json', event)
                    print(json.dumps(dict(event='EPOCH_COMPLETE', **event), ensure_ascii=False), flush=True)
                    seen[key] = event['receipt_sha256']
            write(seen_path, seen)
            write(output / 'LATEST.json', report)
            write(output / 'LATEST_KO.md', markdown(report))
            if once or report['status'] != 'WORKER_ALIVE':
                print(json.dumps(dict(event='OBSERVER_STATUS', status=report['status'],
                    completed=len(report['events']), common_epoch=report['common_epoch'])), flush=True)
                return
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('interval must be between 1 and 60 seconds')
    try:
        run_watch(args.run.resolve(), args.output.resolve(), args.interval, args.once)
    except Exception as exc:
        write(args.output / 'OBSERVER_FAILURE.json', dict(error=repr(exc), time=time.time()))
        raise
