"""Saved-probe desktop notification and readable table; no automatic chat."""
import fcntl
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/epoch_notifications_20261010'))
from notify_epochs import read,write,send,digest
RUN=ROOT/'local/tfgridnet_diversity_20261011_v1'
OUT=ROOT/'local/tfgridnet_diversity_notifications_20261011_v1'


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='WATCHING',pid=os.getpid(),seen=[],events=[],started=time.time())
        while time.time()-state['started']<5400:
            actual=read(RUN/'STATE.json');history=read(RUN/'HISTORY.json')['history'] if (RUN/'HISTORY.json').exists() else []
            lines=['# TF-GridNet 혼합 범위 확대 진행','',
                '전체 모델·128학습혼합·별도 index의 TRAIN30 진단. 원기록을 공유하므로 독립 기록/DEV 검증이 아니다.',
                '',f"현재 상태: {actual['status']}",'',
                '|추가 업데이트|두 NMSE|세 NMSE|두 복소SI-SDR dB|세 복소SI-SDR dB|두 최약NMSE|세 최약NMSE|',
                '|---:|---:|---:|---:|---:|---:|---:|']
            for point in history:
                a,b=point['by_count'];step=point['added_updates']
                lines.append(f"|{step}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}|{b['weakest_nmse']:.6f}|")
                if step not in state['seen']:
                    body=(f"추가 {step}/32업데이트 · TRAIN30 진단\nNMSE2/3: {a['mean_nmse']:.4f}/{b['mean_nmse']:.4f}\n"
                        f"복소SI-SDR2/3: {a['mean_si_sdr']:.2f}/{b['mean_si_sdr']:.2f}dB\n기존 DEV 최선 유지")
                    try:identifier=send('TF-GridNet 자료 범위 확대',body)
                    except Exception as e:
                        write(OUT/'ERROR.json',dict(message=str(e),time=time.time()));continue
                    state['seen'].append(step);state['events'].append(dict(added_updates=step,body=body,
                        notification_id=identifier,human_seen=False,receipt_sha256=digest(RUN/f'ADDED_{step:03d}.json'),time=time.time()))
            lines += ['', '[고정 규약](TFGRIDNET_DIVERSITY_PLAN_KO.md) · [전체 수치](TFGRIDNET_DIVERSITY_PROGRESS.json)']
            target=ROOT/'reports/2026-10-11/TFGRIDNET_DIVERSITY_LIVE_KO.md';tmp=target.with_suffix('.tmp')
            tmp.write_text('\n'.join(lines)+'\n');os.replace(tmp,target)
            state.update(worker=actual,time=time.time())
            if actual['status'] in ('COMPLETED','FAILED'):
                state.update(status=actual['status'],pid=None);write(OUT/'STATE.json',state);return
            write(OUT/'STATE.json',state);time.sleep(10)
        state.update(status='WATCH_TIMEOUT',pid=None);write(OUT/'STATE.json',state)


if __name__=='__main__':main()
