"""Local audited epoch notifications; no chat or external messaging."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'local/bandwidth_timeblock_20261011_v1'
OUT=RUN/'notifications'
spec=importlib.util.spec_from_file_location('timeblock_notify', ROOT/'experiments/epoch_notifications_20261010/notify_epochs.py')
notify=importlib.util.module_from_spec(spec);spec.loader.exec_module(notify)


def render(event, previous):
    rows=event['validation']['by_count']
    lines=[]
    fmt=lambda x:'미정의' if x is None else f'{x:.4f}'
    for k in (2,3):
        a,b=rows[k-1],previous['by_count'][k-1]
        ds=None if a['mean_si_sdr'] is None or b['mean_si_sdr'] is None else a['mean_si_sdr']-b['mean_si_sdr']
        lines.append(f"{k}신호 NMSE {a['mean_nmse']:.6f} ({a['mean_nmse']-b['mean_nmse']:+.6f}), "
            f"복소 SI-SDR {fmt(a['mean_si_sdr'])} dB (Δ{fmt(ds)}), "
            f"최약 NMSE {a['weakest_nmse']:.6f} ({a['weakest_nmse']-b['weakest_nmse']:+.6f})")
    lines.append(f"B 진단 선택 e{event['best']['epoch']}; CPU 감사 PASS. 동일 녹음 시간 분리, 독립 녹음 검증 아님.")
    return f"동일10MHz 새 U-Net {event['epoch']}/50 epoch 완료",'\n'.join(lines)


def run():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=OUT/'STATE.json'
        state=notify.read(saved) if saved.exists() else dict(seen=0,events=[])
        ph=notify.digest(RUN/'PROTOCOL.json')
        while True:
            assert notify.digest(RUN/'PROTOCOL.json')==ph
            worker=notify.read(RUN/'STATE.json')
            for epoch in range(state['seen']+1,51):
                audit=RUN/f'AUDIT_{epoch:03d}.json'
                if not audit.exists():break
                event_path=RUN/f'EPOCH_{epoch:03d}.json'
                checked=notify.read(audit);event=notify.read(event_path)
                assert checked['status']=='PASS' and checked['event_sha256']==notify.digest(event_path)
                assert checked['protocol_sha256']==event['protocol_sha256']==ph
                previous=notify.read(RUN/f'VALIDATION_{epoch-1:03d}.json')
                title,body=render(event,previous)
                try:
                    delivery=dict(notification_id=notify.send(title,body),delivery='LOCAL_DESKTOP_ACKNOWLEDGED')
                except Exception as error:
                    delivery=dict(delivery='LOCAL_DESKTOP_FAILED_FILE_REPORT_SAVED',error=repr(error))
                state['events'].append(dict(epoch=epoch,title=title,body=body,time=time.time(),**delivery))
                state['seen']=epoch
                print(title+'\n'+body,flush=True)
            proc=Path(f"/proc/{worker['pid']}/cmdline")
            alive=proc.exists() and b'bandwidth_timeblock_20261011/train.py' in proc.read_bytes()
            state.update(status='WATCHING' if alive else 'WORKER_EXITED',pid=os.getpid(),worker=worker,
                worker_alive=alive,protocol_sha256=ph,time=time.time())
            notify.write(saved,state)
            if not alive or worker['status'] in ('COMPLETE_AUDITED','STOPPED_WITH_RECOVERY','FAILED'):return
            time.sleep(10)


if __name__=='__main__':run()
