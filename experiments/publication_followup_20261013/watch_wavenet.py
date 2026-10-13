"""Prepare five-epoch comparisons; this worker cannot send chat messages."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'local/wavenet_timeblock100_20261013_v1'
PUBLIC=ROOT/'reports/2026-10-13'


def read(p):return json.loads(p.read_text())


def verified_event(run,epoch):
    p=run/f'EPOCH_{epoch:03d}.json'
    audit=read(run/f'AUDIT_{epoch:03d}.json')
    assert audit['status']=='PASS' and audit['event_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    return read(p)


def stats(event):
    groups=event['validation']['by_count']
    return {str(row['count']):{k:row[k] for k in ('mean_nmse','mean_si_sdr','weakest_nmse','count_accuracy')}
            for row in groups}


def refresh():
    rows=[]
    for epoch in range(5,101,5):
        if not (RUN/f'AUDIT_{epoch:03d}.json').exists():break
        wave=verified_event(RUN,epoch)
        control=ROOT/('local/bandwidth_timeblock_20261011_v1' if epoch<=50
                      else 'local/timeblock_extend100_20261012_v1')
        unet=verified_event(control,epoch)
        assert wave['training_rows_sha256']==unet['training_rows_sha256']
        assert wave['validation']['rows_sha256']==unet['validation']['rows_sha256']
        rows.append(dict(epoch=epoch,updates=wave['updates'],wavenet=stats(wave),unet=stats(unet),
                         wavenet_selected_epoch=wave['best']['epoch'],
                         unet_selected_epoch=unet['best']['epoch']))
    result=dict(status='MONITORING',completed_five_epoch_reports=rows,
                chat_delivery_performed=False,scope='Exact epochs, same B630; not independent recordings')
    (PUBLIC/'WAVENET100_FIVE_EPOCH_REPORTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# WaveNet100: 완료5epoch 간격 비교','',
           '같은 B630,해당epoch의 실제 가중치 비교. 선택epoch와 마지막epoch는 구분한다.',
           '이 파일 갱신은 채팅 발송이 아니다. 실제 채팅 보고 여부는 별도로 확인한다.','',
           '|Epoch|모델|두/세 NMSE↓|두/세 복소SI-SDR↑ dB|두/세 최약NMSE↓|선택epoch|',
           '|---:|---|---:|---:|---:|---:|']
    for row in rows:
        for model in ('wavenet','unet'):
            a,b=row[model]['2'],row[model]['3']
            def si(x):return '미정의' if x['mean_si_sdr'] is None else f"{x['mean_si_sdr']:.3f}"
            lines.append(f"|{row['epoch']}|{model}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
                         f"{si(a)}/{si(b)}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|"
                         f"{row[model+'_selected_epoch']}|")
    if not rows:lines.append('\n완료5epoch 결과는 아직 없다.')
    (PUBLIC/'WAVENET100_FIVE_EPOCH_REPORTS_KO.md').write_text('\n'.join(lines)+'\n')
    return len(rows)


if __name__=='__main__':
    previous=-1
    while True:
        count=refresh()
        if count!=previous:
            print(f'Five-epoch report groups ready: {count}; chat delivery requires active assistant',flush=True)
            previous=count
        state=read(RUN/'STATE.json') if (RUN/'STATE.json').exists() else {}
        if state.get('status') in ('COMPLETE_AUDITED','FAILED','FAILED_STARTUP','STOPPED_WITH_RECOVERY'):
            print('Worker terminal state:',state.get('status'),flush=True);break
        time.sleep(30)
