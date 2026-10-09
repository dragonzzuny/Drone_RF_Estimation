"""Observe receipts and refresh paired reports without touching GPU training.

Emits durable epoch events to the terminal. The active assistant must relay
these to chat; this worker does not implement a chat notification service.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import paired_diagnostics as paired

w = paired.watch


def run(study, root, output):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen_path = root / 'SEEN.json'
        seen = w.read(seen_path) if seen_path.exists() else {}
        sources = {str(Path(p).relative_to(paired.ROOT)): w.digest(Path(p))
                   for p in (__file__, paired.__file__, w.__file__)}
        protocol_path = root / 'PROTOCOL.json'
        protocol = dict(study_protocol_sha256=w.digest(study / 'PROTOCOL.json'),
            source_sha256=sources, waveform_reads=0, checkpoint_reads=0,
            chat_notification_service=False)
        if protocol_path.exists() and w.read(protocol_path) != protocol:
            raise ValueError('Observer registration changed')
        w.write(protocol_path, protocol)
        changed = True
        while True:
            if sources != {rel: w.digest(paired.ROOT / rel) for rel in sources}:
                raise ValueError('Observer source changed')
            snapshot = w.snapshot(study)
            for event in snapshot['events']:
                key = f"{event['arm']}:{event['epoch']}"
                if key in seen and seen[key] != event['receipt_sha256']:
                    raise ValueError('Completed epoch changed')
                if key not in seen:
                    w.write(root / 'events' / (key.replace(':', '_') + '.json'), event)
                    print(json.dumps(dict(event='EPOCH_COMPLETE', **event)), flush=True)
                    seen[key] = event['receipt_sha256']
                    changed = True
            if changed:
                paired.run(study, output)
                w.write(seen_path, seen)
                changed = False
            w.write(root / 'STATE.json', dict(status=snapshot['status'],
                training=snapshot['state'], epoch_events=len(seen),
                common_epoch=snapshot['common_epoch'], pid=os.getpid(), time=time.time()))
            if snapshot['status'] in ('COMPLETED', 'FAILED', 'WORKER_NOT_RUNNING'):
                paired.run(study, output)
                print(snapshot['status'], flush=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'run', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.study.resolve(), args.run.resolve(), args.output.resolve())
    except Exception:
        w.write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
