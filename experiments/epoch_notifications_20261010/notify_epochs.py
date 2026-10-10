"""Report saved epoch results to this user's local desktop, without CUDA.

This is not an automatic chat sender. It reads immutable epoch receipts and
records the notification server's acknowledgement, not whether a human saw it.
"""
import argparse
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'local/epoch_notifications_20261010_v1'
SERIAL=ROOT/'local/five_candidates_20261010_v2'
SOURCES={
    'septda': ('SepTDA 참고 U-Net',ROOT/'local/septda_continuation_20261010_v1/septda'),
    'frozen_tf': ('본체 고정 시간·주파수 보강',SERIAL/'frozen_tf'),
    'ordered_context': ('시간 순서 게이트',SERIAL/'ordered_context'),
    'balanced_head': ('성분 상호작용',SERIAL/'balanced_head'),
    'wave_guard': ('파형 업데이트 보호',SERIAL/'wave_guard'),
}
INITIAL_SEEN={'septda':3,'frozen_tf':1,'ordered_context':1,'balanced_head':0,'wave_guard':1}


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(label,event,previous=None):
    assert 1<=event['epoch']<=50 and event['updates']==event['epoch']*75
    groups={g['count']:g for g in event['validation']['by_count']}
    assert set(groups)=={1,2,3} and all(g['cases']==210 for g in groups.values())
    fmt=lambda x:'미정의' if x is None else f'{x:.3f}'
    lines=[]
    for count in (2,3):
        g=groups[count]
        lines.append(f"{count}신호: NMSE {g['mean_nmse']:.4f} · SI-SDR {fmt(g['mean_si_sdr'])} dB · 최약NMSE {g['weakest_nmse']:.4f}")
    if previous:
        before={g['count']:g for g in previous['validation']['by_count']}
        delta=[groups[c]['mean_nmse']-before[c]['mean_nmse'] for c in (2,3)]
        lines.append(f'직전 대비 NMSE 변화: {delta[0]:+.4f} / {delta[1]:+.4f} (감소가 개선)')
    best=event['best']['epoch']
    lines.append('선택 모델: 기존 부모(e0)' if best==0 else f'선택 모델: e{best}')
    lines.append('동일 DEV 검증 · 독립 확인평가 아님')
    return f"{label} {event['epoch']}/50 epoch 완료",'\n'.join(lines)


def send(title,body):
    env=dict(os.environ,DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus',
             XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
    string=lambda value:json.dumps(value,ensure_ascii=False)
    result=subprocess.run(['/usr/bin/gdbus','call','--session',
        '--dest','org.freedesktop.Notifications','--object-path','/org/freedesktop/Notifications',
        '--method','org.freedesktop.Notifications.Notify',string('Drone RF 학습'),'0',
        string('utilities-system-monitor'),string(title),string(html.escape(body)),'[]','{}','0'],
        env=env,text=True,capture_output=True,timeout=15,check=True)
    match=re.fullmatch(r'\(uint32 (\d+),\)\s*',result.stdout)
    assert match,result.stdout
    return int(match.group(1))


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'STATE.json') if (root/'STATE.json').exists() else dict(seen=dict(INITIAL_SEEN),events=[])
        while True:
            try:
                for name,(label,folder) in SOURCES.items():
                    paths=sorted(folder.glob('EPOCH_*.json'))
                    for path in paths:
                        epoch=int(path.stem.split('_')[1])
                        if epoch<=state['seen'][name]: continue
                        assert epoch==state['seen'][name]+1,'Epoch gap: '+str(path)
                        event=read(path);assert event['epoch']==epoch
                        vp=folder/f'VALIDATION_{epoch:03d}.json'
                        if 'validation_sha256' in event: assert digest(vp)==event['validation_sha256']
                        prev_path=folder/f'EPOCH_{epoch-1:03d}.json'
                        title,body=render(label,event,read(prev_path) if prev_path.exists() else None)
                        notification_id=send(title,body)
                        receipt=dict(candidate=name,epoch=epoch,title=title,body=body,
                            notification_id=notification_id,epoch_sha256=digest(path),time=time.time(),
                            delivery='LOCAL_DESKTOP_SERVER_ACKNOWLEDGED',human_seen=False)
                        state['events'].append(receipt);state['seen'][name]=epoch
                        state.update(status='WATCHING',pid=os.getpid(),time=time.time())
                        write(root/'STATE.json',state)
                        print(json.dumps(receipt,ensure_ascii=False),flush=True)
                state.update(status='WATCHING',pid=os.getpid(),time=time.time())
                write(root/'STATE.json',state)
                if all(e==50 for e in state['seen'].values()):
                    state['status']='ALL_FIVE_REPORTED';write(root/'STATE.json',state);return
            except Exception as error:
                # Do not advance seen: retry delivery after a temporary desktop
                # disconnect. A crash after delivery before recording can repeat
                # one notification; this is at-least-once, not exactly-once.
                write(root/'ERROR.json',dict(type=type(error).__name__,message=str(error),time=time.time()))
                state.update(status='RETRYING',pid=os.getpid(),time=time.time())
                write(root/'STATE.json',state)
            time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args();run(args.run.resolve())
