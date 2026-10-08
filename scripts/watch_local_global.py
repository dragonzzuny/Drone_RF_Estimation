"""Emit each saved fusion epoch and maintain local status; no external messages."""
import argparse
import json
import os
from pathlib import Path
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(cpus[-4:-2]))
    out = args.run/'monitor'
    out.mkdir(exist_ok=True)
    seen = set()
    while True:
        files = sorted(args.run.glob('*/EPOCH_*.json'), key=lambda p:p.stat().st_mtime)
        for path in files:
            if str(path) in seen:
                continue
            row = json.loads(path.read_text())
            event = dict(arm=path.parent.name, **row)
            (out/f'{path.parent.name}_{path.name}').write_text(json.dumps(event, indent=2)+'\n')
            print(json.dumps(event), flush=True)
            seen.add(str(path))
        status = dict(updated_at=time.time(), completed_arm_epochs=len(files))
        for name in ('RUN_STATE.json','PROGRESS.json'):
            path = args.run/name
            if path.exists():
                status[name] = json.loads(path.read_text())
        temporary = out/'STATUS.json.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        os.replace(temporary, out/'STATUS.json')
        if (args.run/'COMPLETE.json').exists() or (args.run/'FAILURE.json').exists():
            print('MONITOR_STOPPED', flush=True)
            break
        time.sleep(10)
