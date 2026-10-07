"""CPU-only milestone audit watcher; never controls or modifies GPU training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from audit_context_evidence import audit
from drone_rf.data import sha256


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'WATCHER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze = json.loads((args.snapshot / 'FREEZE.json').read_text())
        for path, digest in freeze['files'].items():
            if sha256(path) != digest:
                raise ValueError('Audit worker snapshot changed')
        os.nice(10)
        os.sched_setaffinity(0, {14, 15})
        started, completed = time.time(), []
        for epoch in (1, 5, 10, 25, 50):
            destination = args.output / f'EPOCH_{epoch:03d}.json'
            if destination.exists():
                previous = json.loads(destination.read_text())
                if previous['epoch'] != epoch or previous['freeze_sha256'] != sha256(args.experiment / 'run/FREEZE.json'):
                    raise ValueError('Existing audit belongs to another experiment')
                completed.append(epoch)
                continue
            required = [args.experiment / 'run' / arm / f'{kind}_{epoch:03d}.json'
                        for arm in ('ordered', 'mean') for kind in ('EPOCH', 'VALIDATION')]
            while not all(path.exists() for path in required):
                if (args.experiment / 'run/FAILURE.json').exists():
                    raise RuntimeError('Training reported failure; evidence watcher stops')
                if time.time() - started > 72 * 3600:
                    raise TimeoutError('72-hour audit watch limit reached; GPU run is untouched')
                write(args.output / 'STATE.json', dict(status='WAITING_MATCHED_EPOCH',
                    target_epoch=epoch, completed=completed, pid=os.getpid(), time=time.time(),
                    gpu_used=False, snapshot_sha256=sha256(args.snapshot / 'FREEZE.json')))
                time.sleep(30)
            result = audit(args.experiment, epoch)
            write(destination, result)
            completed.append(epoch)
            print(json.dumps(dict(epoch=epoch, status=result['status'],
                nmse={arm: result['arms'][arm]['separation_only_count_macro_nmse']
                      for arm in ('ordered', 'mean')})), flush=True)
        write(args.output / 'STATE.json', dict(status='ALL_MILESTONE_AUDITS_COMPLETE',
            completed=completed, pid=os.getpid(), time=time.time(), gpu_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / 'FAILURE.json', dict(time=time.time(), pid=os.getpid(),
            traceback=traceback.format_exc(), gpu_training_modified=False))
        raise
