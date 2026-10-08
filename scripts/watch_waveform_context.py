"""Read-only training monitor; writes local Korean progress and matched results.

Does not alter training, choose new experiments, read I/Q, or publish results.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(value)
    os.replace(temporary,path)


def render(run,output):
    state=read(run/'RUN_STATE.json') or {}
    progress=read(run/'PROGRESS.json') or {}
    done=read(run/'COMPLETE.json')
    pid=state.get('pid')
    alive=False
    if pid:
        cmd=Path('/proc',str(pid),'cmdline')
        if cmd.exists():
            try:
                contents=cmd.read_bytes()
                alive=b'run_comparison.py' in contents and run.name.encode() in contents
            except FileNotFoundError:
                pass
    status='완료' if done else '실행 중' if alive else '프로세스 없음 — 상태 확인 필요'
    timestamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# 복소 문맥 대조 진행',f'갱신: {timestamp}',f'상태: {status}',
           '짧은 복소 0.63872ms / 긴 복소 10.48576ms. 같은 0.63872ms 정답 구간 채점.',
           '현재 단계: '+json.dumps(progress,ensure_ascii=False),
           '', '| 군 | 완료 epoch | 업데이트 | 현재 NMSE (2/3) | 선택 epoch |',
           '|---|---:|---:|---|---:|']
    records={}
    for arm in ('short_context','long_context'):
        paths=sorted((run/arm).glob('EPOCH_*.json'))
        if not paths:
            lines.append(f'| {arm} | 0 | 진행 파일 참조 | 평가 대기 | — |')
            continue
        row=read(paths[-1]); groups=row['metrics']['by_count']
        values=[g['mean_nmse'] for g in groups if g['count']>1]
        records[arm]=row
        lines.append(f"| {arm} | {row['epoch']} | {row['updates']} | {values[0]:.6f} / {values[1]:.6f} | {row['best']['epoch']} |")
    if len(records)==2:
        common=min(r['epoch'] for r in records.values())
        pair={a:read(run/a/f'EPOCH_{common:03d}.json') for a in records}
        for row in pair.values():
            if row['updates']!=300*common:
                raise RuntimeError('Unmatched update budget')
        summary=dict(updated_at=timestamp,common_epoch=common,updates_per_arm=300*common,
                     current_epoch=pair,independent_test=False,physical_aircraft_count=False)
        save(output/f'MATCHED_{common:03d}.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines.extend(['','고정 개발 검증 결과이며 한 seed다. 학습 손실 감소를 복원·일반화 개선으로 대체하지 않는다.',
                  '실험은 같은 원 RF 대역 중심 정렬 합성이며 원 중심주파수 간격은 보존하지 않는다.'])
    save(output/'STATUS_KO.md','\n'.join(lines)+'\n')
    return done is not None or (not alive and state.get('status') in ('FAILED','COMPLETED'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.nice(10)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2]))
    args.output.mkdir(parents=True,exist_ok=True)
    while True:
        if render(args.run,args.output):
            print('MONITOR_COMPLETE',flush=True);break
        time.sleep(30)
