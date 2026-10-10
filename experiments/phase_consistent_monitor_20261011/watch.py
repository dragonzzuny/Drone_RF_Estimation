"""Read-only C8 epoch reports after CPU audit, with local desktop delivery.

Never updates/stops training or opens I/Q. Desktop acknowledgement does not
mean a human read the message, and does not send an automatic chat reply.
"""
import fcntl
import importlib.util
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'local/phase_consistent_20261011_v1'
OUT=ROOT/'local/phase_consistent_notifications_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'
spec=importlib.util.spec_from_file_location('c8_local_notify',ROOT/'experiments/epoch_notifications_20261010/notify_epochs.py')
notify=importlib.util.module_from_spec(spec);spec.loader.exec_module(notify)

def render(event):
    current={r['count']:r for r in event['eight']['by_count']}
    before={r['count']:r for r in event['preceding_epoch']['by_count']}
    fmt=lambda v:'미정의' if v is None else f'{v:.4f}'
    lines=[]
    for k in (2,3):
        a,b=current[k],before[k]
        assert a['cases']==210 and b['cases']==210
        delta_si=None if a['mean_si_sdr'] is None or b['mean_si_sdr'] is None else a['mean_si_sdr']-b['mean_si_sdr']
        lines.append(f"{k}신호: NMSE {a['mean_nmse']:.6f} ({a['mean_nmse']-b['mean_nmse']:+.6f}), "
                     f"SI-SDR {fmt(a['mean_si_sdr'])} dB (변화 {fmt(delta_si)}), "
                     f"최약NMSE {a['weakest_nmse']:.6f} ({a['weakest_nmse']-b['weakest_nmse']:+.6f})")
    lines += [f"공동 개선 기준: {'통과' if event['candidate'] else '미통과'}; 연구 내부 선택 e{event['best']['epoch']}",
              'DEV630 · CPU 저장점/집계 검사 PASS · 독립 확인평가 아님']
    return f"8위상 일관성 U-Net 추가 {event['epoch']}/10 epoch 완료",'\n'.join(lines)

def report(state,worker_state):
    text=['# 8위상 일관성 U-Net epoch 보고','',
          f"실행 상태: {worker_state.get('status')}; 추가epoch {worker_state.get('epoch',0)}/10; "
          f"업데이트 {worker_state.get('updates',0)}/750; 현재혼합 {worker_state.get('examples',0)}/2400.",'',
          f"상태 확인 시각 Unix {time.time():.3f}. 완료·검산 후 보고한epoch: {state['seen']}개.",
          '로컬 바탕화면 알림과 보고 파일이다. 자동 채팅 발송이나 사용자의 열람을 의미하지 않는다.','']
    for e in state['events']:
        text += [f"## {e['title']}",'',e['body'],'']
    path=PUBLIC/'PHASE_CONSISTENT_EPOCH_REPORTS_KO.md'
    temp=path.with_suffix('.tmp');temp.write_text('\n'.join(text));os.replace(temp,path)

def run():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=OUT/'STATE.json'
        state=notify.read(saved) if saved.exists() else dict(seen=0,events=[])
        protocol=notify.digest(RUN/'PROTOCOL.json')
        while True:
            try:
                assert notify.digest(RUN/'PROTOCOL.json')==protocol
                worker_state=notify.read(RUN/'STATE.json')
                for epoch in range(state['seen']+1,11):
                    path=RUN/f'EPOCH_{epoch:03d}.json'
                    audit=PUBLIC/f'PHASE_CONSISTENT_AUDIT_E{epoch:03d}.json'
                    if not path.exists() or not audit.exists():break
                    event=notify.read(path);checked=notify.read(audit)
                    assert event['epoch']==epoch and event['updates']==75*epoch
                    assert event['protocol_sha256']==checked['protocol_sha256']==protocol
                    assert checked['status']=='PASS' and checked['checkpoint_sha256']==event['checkpoint_sha256']
                    title,body=render(event)
                    notification_id=notify.send(title,body)
                    receipt=dict(epoch=epoch,title=title,body=body,epoch_sha256=notify.digest(path),
                        audit_sha256=notify.digest(audit),notification_id=notification_id,
                        delivery='LOCAL_DESKTOP_SERVER_ACKNOWLEDGED',human_seen=False,time=time.time())
                    state['events'].append(receipt);state['seen']=epoch
                    notify.write(saved,state);print(receipt,flush=True)
                status='COMPLETE' if state['seen']==10 else 'WATCHING'
                if worker_state['status']=='FAILED':status='TRAINING_FAILED'
                pid=worker_state['pid']
                cmd=Path(f'/proc/{pid}/cmdline')
                alive=cmd.exists() and b'phase_consistent_20261011/train.py' in cmd.read_bytes()
                if not alive and status=='WATCHING':status='TRAINING_PROCESS_EXITED'
                state.update(status=status,pid=os.getpid(),worker=worker_state,worker_alive=alive,
                             protocol_sha256=protocol,time=time.time())
                notify.write(saved,state);report(state,worker_state)
                if status!='WATCHING':return
            except Exception as error:
                notify.write(OUT/'ERROR.json',dict(error=repr(error),time=time.time()))
                state.update(status='RETRYING',pid=os.getpid(),time=time.time());notify.write(saved,state)
            time.sleep(10)

if __name__=='__main__':run()
