"""Lightweight durable serial runner; avoid shadowing Python's queue module."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from workflow import ROOT, RUN, PUBLIC, ORDER, LABELS, SPECS, read, write, digest, verify


def completed(root,name):
    folder=Path(read(root/'PROTOCOL.json')['septda_run'])/'septda' if name=='septda' else root/name
    files=sorted(folder.glob('EPOCH_*.json'))
    events=[read(f) for f in files]
    assert [e['epoch'] for e in events]==list(range(1,len(events)+1))
    return events


def progress(root,status,active=None):
    p=read(root/'PROTOCOL.json');rows=[];histories={}
    lines=['# 후보5개 순차50구간 진행표','',
           '각 후보 총50구간. 기존 구간은 포함하며 공통 대조를 반복 학습하지 않는다. '
           '1구간=2,400합성·75업데이트,50구간=고정12,000합성10순회다. '
           '새 기종/미개봉 확인 평가 결과가 아닌 동일 DEV 추적이다.','',
           '|순서|후보|완료/목표|상태|최신 NMSE2/3 ↓|SI-SDR2/3 ↑ dB|최약NMSE2/3 ↓|선택 구간|',
           '|---:|---|---:|---|---|---|---|---:|']
    for i,name in enumerate(ORDER):
        events=completed(root,name);histories[name]=events
        marker=Path(p['septda_run'])/'SEPTDA_PRIORITY_COMPLETE.json' if name=='septda' else root/name/'COMPLETE.json'
        s='완료' if marker.exists() else ('진행' if name==active else '대기')
        if name=='septda' and not marker.exists():
            process=read(Path(p['septda_run'])/'STATE.json')
            alive=(Path('/proc')/str(process['pid'])).exists()
            s='진행' if process['status'] in ('TRAINING','VALIDATING') and alive else (process['status'] if alive else '프로세스 확인 필요')
        if events:
            last=events[-1];a,b=last['validation']['by_count'][1:]
            fmt=lambda v:'미정의' if v is None else f'{v:.3f}'
            cells=[f"{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}",f"{fmt(a['mean_si_sdr'])}/{fmt(b['mean_si_sdr'])}",
                   f"{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}",str(last['best']['epoch'])]
        else: cells=['—']*4
        rows.append(dict(candidate=name,completed=len(events),maximum=50,status=s))
        lines.append(f'|{i+1}|{LABELS[i]}|{len(events)}/50|{s}|'+'|'.join(cells)+'|')
    lines+=['','각 후보 전체 epoch 결과는 다음 JSON의histories와 로컬EPOCH/VALIDATION 파일에 누적한다. '
              '이 파일 자동 갱신은 채팅 자동 발송을 뜻하지 않는다.',
             f'','[실행 규약](FIVE_CANDIDATES_PLAN_KO.md) · [전체 누적 지표](FIVE_CANDIDATES_PROGRESS.json)']
    result=dict(status=status,active=active,rows=rows,histories=histories,queue_pid=os.getpid(),
                protocol_sha256=digest(root/'PROTOCOL.json'),time=time.time(),heldout_read=False)
    write(PUBLIC/'FIVE_CANDIDATES_PROGRESS.json',result)
    write(PUBLIC/'FIVE_CANDIDATES_PROGRESS_KO.md','\n'.join(lines)+'\n')


def check_complete(root,name,p):
    folder=Path(p['septda_run'])/'septda' if name=='septda' else root/name
    marker=folder.parent/'SEPTDA_PRIORITY_COMPLETE.json' if name=='septda' else folder/'COMPLETE.json'
    if not marker.exists(): return False
    value=read(marker);events=completed(root,name)
    assert len(events)==50 and value['epochs']==50 and value['updates']==3750
    assert events[-1]['last_sha256']==digest(folder/'LAST.pt')
    for e in events[1:]:
        assert e['updates']==75*e['epoch']
        assert digest(folder/f"VALIDATION_{e['epoch']:03d}.json")==e['validation_sha256']
    if name=='septda':
        assert value['status']=='SEPTDA_BUDGET_COMPLETE'
        assert value['amendment_sha256']==p['septda_execution_amendment_sha256']
    else:
        assert value['status']=='BUDGET_COMPLETE_AUDITED' and value['protocol_sha256']==digest(root/'PROTOCOL.json')
    return True


def run(root,python):
    p=read(root/'PROTOCOL.json');verify(root,p)
    ready=read(root/'READY.json');assert ready['protocol_sha256']==digest(root/'PROTOCOL.json')
    for name,sha in ready['preflight_sha256'].items():
        assert digest(root/name/'PREFLIGHT.json')==sha and read(root/name/'PREFLIGHT.json')['status']=='PASS'
    with (root/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'QUEUE_PID',str(os.getpid()))
        while not check_complete(root,'septda',p):
            for f in ('PRIORITY_FAILURE.json','PRIORITY_HANDOFF_FAILURE.json'):
                if (Path(p['septda_run'])/f).exists(): raise RuntimeError('SepTDA stopped: '+f)
            progress(root,'WAITING_SEPTDA_50','septda')
            write(root/'QUEUE_STATE.json',dict(status='WAITING_SEPTDA_50',pid=os.getpid(),time=time.time()))
            time.sleep(20)
        for name in ORDER[1:]:
            if check_complete(root,name,p): continue
            verify(root,p)
            assert not (root/name/'FAILURE.json').exists(),'Resolve failure before resuming queue'
            folder=root/name
            with (folder/'worker.log').open('a') as log:
                process=subprocess.Popen(['taskset','-c','14,15','nice','-n','10',python,
                    str(Path(__file__).with_name('train.py')),'--run',str(root),'--candidate',name],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                write(folder/'PID',str(process.pid))
                write(root/'QUEUE_STATE.json',dict(status='RUNNING_CANDIDATE',candidate=name,worker_pid=process.pid,pid=os.getpid(),time=time.time()))
                while process.poll() is None:
                    progress(root,'RUNNING_CANDIDATE',name);time.sleep(20)
                if process.returncode: raise RuntimeError(f'{name} exited {process.returncode}')
            assert check_complete(root,name,p)
            progress(root,'CANDIDATE_COMPLETE',None)
        verify(root,p)
        write(root/'COMPLETE.json',dict(status='ALL_FIVE_BUDGETS_COMPLETE',epochs_each=50,
            total_candidate_updates=18750,protocol_sha256=digest(root/'PROTOCOL.json'),time=time.time(),heldout_read=False))
        write(root/'QUEUE_STATE.json',dict(status='ALL_FIVE_BUDGETS_COMPLETE',pid=os.getpid(),time=time.time()))
        progress(root,'ALL_FIVE_BUDGETS_COMPLETE',None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--python',required=True);a=parser.parse_args();root=a.run.resolve()
    try: run(root,a.python)
    except Exception:
        write(root/'QUEUE_FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        write(root/'QUEUE_STATE.json',dict(status='FAILED',pid=os.getpid(),time=time.time()))
        progress(root,'FAILED',None)
        raise
