"""Did the supervised embedding objective change at the saved endpoints?

CPU-only; same first two cases per count (2/3) in TRAIN and reused DEV.
Wait for the full three-arm training audit. This file is independent of the
already registered immutable training dependency snapshot.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import torch
from torch.nn import functional as F
from affinity import Capture,targets,loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from drone_rf.waveform import analyze
w=core.w
ARMS=('affinity_control','hard_affinity','soft_affinity')


def run(root,study):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    p=w.read(study/'PROTOCOL.json');old=p['original_protocol']
    sources=dict(p['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    libraries={role:core.base.worker.NativeMixtures(old['preparation'],role,1)
               for role in ('train_pack','validation_pack')}
    # Only construction metadata are used before checkpoint availability.
    indices={role:[int(i) for n in (2,3) for i in (data.rows['count']==n).nonzero()[0][:2]]
             for role,data in libraries.items()}
    protocol=dict(status='REGISTERED_CPU_AFFINITY_ENDPOINT_DIAGNOSTIC',indices=indices,sources=sources,
        predecessor=str(study),predecessor_protocol_sha256=w.digest(study/'PROTOCOL.json'),
        selection='First two existing examples of counts2/3 per role, chosen before candidate endpoints',
        inferences=24,optimizer_steps=0,heldout_read=False,dev_reused=True,gpu_use=False,
        registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol)
    audit=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_AUDIT.json'
    while not audit.exists():
        if (study/'FAILURE.json').exists():raise RuntimeError('Affinity predecessor failed')
        if (study/'SKIPPED.json').exists():raise RuntimeError('Affinity predecessor skipped')
        w.write(root/'STATE.json',dict(status='WAITING_FOR_AFFINITY_AUDIT',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    receipt=w.read(audit);assert receipt['status']=='PASS' and receipt['complete_sha256']==w.digest(study/'COMPLETE.json')
    torch.set_num_threads(2);rows=[];hashes={};started=time.time()
    for arm in ARMS:
        path=study/arm/'ACTUAL_001.pt';hashes[arm]=w.digest(path)
        assert hashes[arm]==receipt['hashes'][arm+'/ACTUAL_001.pt']
        net=core.base.worker.make_model('retained_unet',None).eval();capture=Capture(net)
        saved=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(saved['model']);del saved
        capture.enabled=True
        with torch.inference_mode():
            for role,data in libraries.items():
                for index in indices[role]:
                    item=core.batch(data[index]);prediction,_=core.base.worker.predict(net,item);embedding=capture.take()
                    power=F.avg_pool2d(analyze(item['references'][0]).abs().square()[None],2)
                    hy,hw=targets(power,'hard');sy,sw=targets(power,'soft');assert torch.equal(hw,sw)
                    hard=float(loss(embedding,hy,hw));soft=float(loss(embedding,sy,sw))
                    metric,_=core.metrics(prediction,item)
                    rows.append(dict(arm=arm,role=role,index=index,count=int(item['construction_count']),
                                     hard_affinity=hard,soft_affinity=soft,waveform=metric))
                    w.write(root/'STATE.json',dict(status='CPU_AFFINITY_ENDPOINT',inferences=len(rows),
                        total_inferences=24,pid=os.getpid(),time=time.time()))
        capture.close();del net,capture,item,prediction,embedding,power,hy,hw,sy,sw;gc.collect()
        assert w.digest(path)==hashes[arm]
    summary=[]
    for arm in ARMS:
        for role in libraries:
            for n in (2,3):
                selected=[r for r in rows if r['arm']==arm and r['role']==role and r['count']==n]
                assert len(selected)==2
                summary.append(dict(arm=arm,role=role,count=n,cases=2,
                    hard_affinity=statistics.mean(r['hard_affinity'] for r in selected),
                    soft_affinity=statistics.mean(r['soft_affinity'] for r in selected),
                    mean_nmse=statistics.mean(x for r in selected for x in r['waveform']['nmse'])))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(study/'PROTOCOL.json')==protocol['predecessor_protocol_sha256']
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),
        audit_sha256=w.digest(audit),checkpoint_sha256=hashes,rows=rows,summary=summary,
        optimizer_steps=0,inferences=24,gpu_use=False,heldout_read=False,dev_reused=True,
        seconds=time.time()-started,
        limitation='Two cases per count/role; supervised embedding diagnosis, not deployable assignment or independent test')
    public=ROOT/'reports/2026-10-10';w.write(root/'COMPLETE.json',result);w.write(public/'SOURCE_AFFINITY_ENDPOINT_RESULT.json',result)
    lines=['# 친화도 보조 목표와 실제 파형의 종점 진단','',
        '원규모 실제e1 세 군에서 개수별 앞2혼합씩TRAIN4+DEV4,총24회CPU추론이다. '
        '정답은 친화도 진단과파형평가에만 사용했다. 개발 자료의 결과로 추가 하이퍼파라미터를 선택하지 않는다.','',
        '| 자료 | 성분 수 | 군 | hard친화도 손실 ↓ | soft친화도 손실 ↓ | NMSE ↓ |',
        '|---|---:|---|---:|---:|---:|']
    for row in summary:lines.append(f"|{row['role']}|{row['count']}|{row['arm']}|{row['hard_affinity']:.6f}|{row['soft_affinity']:.6f}|{row['mean_nmse']:.6f}|")
    lines+=['','hard와soft는 서로 다른 목표이므로 각각 같은 열 안에서 군을 비교한다. '
        '보조 손실 감소만으로 복원 개선을 주장하지 않는다. 전체630개DEV의채택판정을 이 소수 사례로 대체하지 않는다.',
        '', '[전체24행](SOURCE_AFFINITY_ENDPOINT_RESULT.json) · [전체개발비교](SOURCE_AFFINITY_KO.md)']
    (public/'SOURCE_AFFINITY_ENDPOINT_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',inferences=24,pid=os.getpid(),time=time.time()))
    print(dict(summary=summary,seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True);a=parser.parse_args()
    try:run(a.run.resolve(),a.study.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
