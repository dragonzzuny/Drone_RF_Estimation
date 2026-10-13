"""Local desktop milestones plus evidence files; never claims chat delivery."""
import json
import os
import subprocess
import time
from watch_wavenet import RUN, PUBLIC, read, refresh

RECEIPT=RUN/'DESKTOP_REPORT_RECEIPTS.json'


def notify(title, body):
    env=dict(os.environ, DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    result=subprocess.run(['/usr/bin/notify-send','-a','Drone RF 연구','-u','normal',title,body],
                          env=env,timeout=15,check=False,capture_output=True,text=True)
    return dict(time=time.time(),returncode=result.returncode,
                delivery='desktop_command_completed' if result.returncode==0 else 'desktop_command_failed',
                chat_message_sent=False)


def main():
    history=read(RECEIPT) if RECEIPT.exists() else {}
    while True:
        try:
            refresh()
            state=read(RUN/'STATE.json')
            reports=read(PUBLIC/'WAVENET100_FIVE_EPOCH_REPORTS.json')['completed_five_epoch_reports']
        except (json.JSONDecodeError,FileNotFoundError):
            time.sleep(1);continue
        if state.get('updates',0)>0 and 'training_start' not in history:
            history['training_start']=notify('WaveNet 100epoch 본학습 시작',
                f"GPU 실제 업데이트 확인: {state['updates']}회. 같은 B 분할·실효 batch32.")
        for row in reports:
            key=f"epoch_{row['epoch']:03d}"
            if key in history:continue
            a,b=row['wavenet']['2'],row['wavenet']['3']
            ua,ub=row['unet']['2'],row['unet']['3']
            si=lambda g:'미정의' if g['mean_si_sdr'] is None else f"{g['mean_si_sdr']:.3f}"
            body=(f"두/세 NMSE: {a['mean_nmse']:.4f} / {b['mean_nmse']:.4f}\n"
                  f"복소 SI-SDR: {si(a)} / {si(b)} dB\n"
                  f"최약 NMSE: {a['weakest_nmse']:.4f} / {b['weakest_nmse']:.4f}\n"
                  f"같은 epoch U-Net NMSE: {ua['mean_nmse']:.4f} / {ub['mean_nmse']:.4f}\n"
                  f"선택 가중치: e{row['wavenet_selected_epoch']} | 상세: WAVENET100_FIVE_EPOCH_REPORTS_KO.md")
            history[key]=notify(f"WaveNet {row['epoch']}/100epoch 완료·검산",body)
        terminal=state.get('status') in ('COMPLETE_AUDITED','FAILED','FAILED_STARTUP','STOPPED_WITH_RECOVERY')
        if terminal and 'terminal' not in history:
            history['terminal']=notify('WaveNet 실행 상태',
                f"{state['status']}, epoch {state.get('epoch','?')}, 업데이트 {state.get('updates','?')}")
        temporary=RECEIPT.with_suffix('.tmp')
        temporary.write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n');temporary.replace(RECEIPT)
        if terminal:break
        time.sleep(30)


if __name__=='__main__':main()
