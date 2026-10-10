"""Local head-adaptation notifications; these do not send chat messages."""
import fcntl
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/epoch_notifications_20261010'))
from notify_epochs import read,write,send,digest
RUN=ROOT/'local/tfgridnet_head_adapt_20261011_v1'
OUT=ROOT/'local/tfgridnet_head_notifications_20261011_v1'


def main():
    OUT.mkdir(exist_ok=False)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='WATCHING',pid=os.getpid(),seen=[],events=[],started=time.time())
        while time.time()-state['started']<2400:
            if not (RUN/'STATE.json').exists():
                time.sleep(5)
                continue
            actual=read(RUN/'STATE.json')
            history=read(RUN/'HISTORY.json')['history'] if (RUN/'HISTORY.json').exists() else []
            lines=['# 전체 TF-GridNet 출력층 적응 진행','',
                '전체9,004,297파라미터 유지, 출력층9,224개만 학습. TRAIN30 반복 진단이며 독립 검증이 아니다.',
                '',f"현재 상태: {actual['status']}",'',
                '|추가 업데이트|두 NMSE|세 NMSE|두 복소SI-SDR dB|세 복소SI-SDR dB|두 최약NMSE|세 최약NMSE|',
                '|---:|---:|---:|---:|---:|---:|---:|']
            for point in history:
                a,b=point['by_count'];step=point['added_updates']
                lines.append(f"|{step}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}|{b['weakest_nmse']:.6f}|")
                if step not in state['seen']:
                    body=(f"출력층 적응 추가{step}회 · TRAIN30\nNMSE2/3 {a['mean_nmse']:.4f}/{b['mean_nmse']:.4f}\n"
                          f"복소SI-SDR {a['mean_si_sdr']:.2f}/{b['mean_si_sdr']:.2f}dB\n기존 DEV 최선 유지")
                    try:
                        identifier=send('TF-GridNet 출력층 적응',body)
                    except Exception as e:
                        write(OUT/'ERROR.json',dict(message=str(e),time=time.time()))
                        continue
                    state['seen'].append(step)
                    state['events'].append(dict(added_updates=step,body=body,notification_id=identifier,
                        receipt_sha256=digest(RUN/f'ADDED_{step:03d}.json'),human_seen=False,time=time.time()))
            target=ROOT/'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_LIVE_KO.md'
            tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');os.replace(tmp,target)
            state.update(worker=actual,time=time.time())
            if actual['status'] in ('COMPLETED','FAILED'):
                state.update(status=actual['status'],pid=None);write(OUT/'STATE.json',state);return
            write(OUT/'STATE.json',state);time.sleep(10)
        state.update(status='WATCH_TIMEOUT',pid=None);write(OUT/'STATE.json',state)


if __name__=='__main__':
    main()
