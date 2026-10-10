"""Local saved-update report and desktop notification; not a chat sender."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/epoch_notifications_20261010'))
from notify_epochs import send,digest,write,read

RUN=ROOT/'local/tfgridnet_continuation_20261011_v1'
OUT=ROOT/'local/tfgridnet_continuation_notifications_20261011_v1'
REPORT=ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_LIVE_KO.md'


def render(history,state):
    lines=['# TF-GridNet RF 고정 TRAIN4 진행 보고','',
        '같은 네 학습 혼합의 적합도 검사다. DEV·보류 I/Q는 읽지 않았고, 기존 최선 모델을 대체하지 않는다.',
        '',f"최종 확인 상태: {state['status']}. 실제 worker STATE와 대조한다.",'',
        '|업데이트|두 성분 NMSE|세 성분 NMSE|두 성분 SI-SDR dB|세 성분 SI-SDR dB|세 성분 최약NMSE|',
        '|---:|---:|---:|---:|---:|---:|']
    for point in history:
        a,b=point['by_count'];assert [a['count'],b['count']]==[2,3]
        lines.append(f"|{point['step']}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{b['weakest_nmse']:.6f}|")
    lines+=['','새 초기화6블록/D128/LSTM192/4heads. 전체512×500 격자. 전체최대64업데이트/추가40분의 진단이며 수렴 완료·새 기록 일반화·물리 드론 대수 추정의 증거가 아니다.',
        '', '[고정 규약](TFGRIDNET_CONTINUATION_PLAN_KO.md) · [초기화 추가 명세](TFGRIDNET_INITIALIZATION_NOTE_KO.md) · [전체 수치](TFGRIDNET_CONTINUATION_PROGRESS.json)']
    temp=REPORT.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');os.replace(temp,REPORT)


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='WATCHING',seen=[],events=[],pid=os.getpid(),started_at=time.time())
        if (OUT/'STATE.json').exists():state=read(OUT/'STATE.json');state['pid']=os.getpid()
        while time.time()-state['started_at']<7200:
            if not (RUN/'HISTORY.json').exists():
                time.sleep(5);continue
            progress=read(RUN/'STATE.json');history=read(RUN/'HISTORY.json')['history']
            render(history,progress)
            for point in history:
                step=point['step']
                if step<=32 or step in state['seen']:continue
                a,b=point['by_count']
                body=(f"TRAIN4 진단 {step}/64업데이트\n두/세 신호 NMSE: {a['mean_nmse']:.4f} / {b['mean_nmse']:.4f}\n"
                    f"복소 SI-SDR: {a['mean_si_sdr']:.2f} / {b['mean_si_sdr']:.2f}dB\n세 신호 최약NMSE: {b['weakest_nmse']:.4f}\n"
                    '고정 학습자료 진단 · DEV 성과 아님 · 기존 최선 유지')
                try:
                    notification=send('TF-GridNet RF 진단',body)
                except Exception as e:
                    write(OUT/'ERROR.json',dict(message=str(e),step=step,time=time.time()));continue
                state['seen'].append(step);state['events'].append(dict(step=step,body=body,
                    notification_id=notification,receipt_sha256=digest(RUN/f'STEP_{step:03d}.json'),
                    delivery='LOCAL_DESKTOP_SERVER_ACKNOWLEDGED',human_seen=False,time=time.time()))
            state.update(time=time.time(),worker=progress)
            if progress['status'] in ('COMPLETED','FAILED'):
                state.update(status=progress['status'],pid=None);write(OUT/'STATE.json',state);return
            write(OUT/'STATE.json',state);time.sleep(10)
        state.update(status='WATCH_TIMEOUT',pid=None,time=time.time());write(OUT/'STATE.json',state)


if __name__=='__main__':main()
