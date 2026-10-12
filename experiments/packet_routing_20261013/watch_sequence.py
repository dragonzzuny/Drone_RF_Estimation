"""Read-only audited-epoch reports; a local file is not an in-chat notification."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/sequence_assignment_20261013_v1'
PUBLIC=ROOT/'reports/2026-10-13'


def read(path):
    return json.loads(path.read_text())


def atomic(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def collect():
    rows=[]
    for epoch in (1,2,3):
        for arm in ('shared','independent'):
            path=OUT/arm/f'EPOCH_{epoch:03d}.json';audit=OUT/arm/f'AUDIT_{epoch:03d}.json'
            if not audit.exists():
                continue
            receipt=read(audit)
            assert receipt['status']=='PASS'
            assert receipt['event_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            event=read(path)
            assert event['epoch']==epoch and event['arm']==arm and event['updates']==96*epoch
            rows.append(event)
    return rows


def render():
    events=collect();state=read(OUT/'STATE.json');initial=read(OUT/'INITIALIZATION.json')
    text=['# 연속 구간 출력 배정 비교: 검산된 epoch', '',
          '시작점은 기존 선정99epoch다.두 군 모두 같은768혼합묶음/96업데이트,새AdamW1e-5,',
          '전체32,142,859파라미터,같은C8검증이다.공통 순열군만 세 연속창에 하나의',
          '출력 대응을 적용한다.입력 길이나 신경망 연결을 늘린 학습은 아니다.', '',
          '|군|추가epoch|두 NMSE↓|세 NMSE↓|두 SI-SDR↑ dB|세 SI-SDR↑ dB|두 최약 NMSE↓|세 최약 NMSE↓|긴 구간 macroNMSE↓|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(label,epoch,short,long):
        g={v['count']:v for v in short['by_count']}
        return f"|{label}|{epoch}|{g[2]['mean_nmse']:.6f}|{g[3]['mean_nmse']:.6f}|{g[2]['mean_si_sdr']:.3f}|{g[3]['mean_si_sdr']:.3f}|{g[2]['weakest_nmse']:.6f}|{g[3]['weakest_nmse']:.6f}|{long:.6f}|"
    baseline=read(OUT/'INITIAL_LONG.json')
    text.append(row('부모99',0,initial['selected_validation'],baseline['merged']['selection_nmse']))
    for event in events:
        text.append(row('공통순열' if event['arm']=='shared' else '창별순열',event['epoch'],
                        event['short'],event['long_merged']['selection_nmse']))
    text+=['','B 같은 원녹음의 시간분할 진단이다.독립 원녹음/보류 기종 검증으로 표현하지 않는다.',
           '긴 구간은 사전 고정24개이며 원래 짧은DEV630과 별도로 해석한다.',
           '중간epoch로 방법의 일반적 우월성을 결정하지 않고 등록된3epoch씩의 최종 비교를 마친다.',
           '이 파일은 로컬 자동 보고이며 대화 보고를 보냈다는 뜻이 아니다.','']
    atomic(PUBLIC/'SEQUENCE_EPOCH_REPORTS_KO.md','\n'.join(text))
    now=time.strftime('%Y-%m-%d %H:%M:%S %Z')
    live=['# 출력 배정 비교 현재 상태','',f'확인 시각: {now}',
          f"현재 상태: `{state['status']}`, PID {state['pid']}",
          f"군: {state.get('arm','-')}, 추가 epoch: {state.get('epoch','-')}, 업데이트: {state.get('updates','-')}",
          f"완료·감사 통과 epoch: {len(events)}/6",'',
          '[검산된 epoch별 결과](SEQUENCE_EPOCH_REPORTS_KO.md)','']
    if 'bundles' in state:
        live.insert(5,f"처리 혼합묶음: {state['bundles']}/{state['total']}")
    elif 'case' in state:
        live.insert(5,f"검증 사례: {state['case']}/{state['total']}")
    atomic(PUBLIC/'SEQUENCE_LIVE_KO.md','\n'.join(live))
    return state,len(events)


if __name__=='__main__':
    while True:
        state,count=render()
        if state['status'] in ('FAILED','STOPPED_WITH_RECOVERY'):
            break
        if state['status']=='COMPLETE_AUDITED':
            assert count==6
            break
        time.sleep(20)
