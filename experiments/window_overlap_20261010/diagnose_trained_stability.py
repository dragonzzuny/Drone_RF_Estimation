"""Did paired consistency reduce window dependence at the saved endpoints?

CPU diagnosis on first two TRAIN and DEV examples per count; no optimization.
Source correspondence uses references for scoring only, never for inference.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
import numpy as np
import torch
import diagnose as core
from paired import paired_items,regions,canonical,consistency

ROOT=core.ROOT;w=core.w

def run(root):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    study=ROOT/'local/paired_window_20261010_v1';registered=w.read(study/'PROTOCOL.json')
    assert w.read(ROOT/'reports/2026-10-10/PAIRED_WINDOW_AUDIT.json')['status']=='PASS'
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    checkpoints={'parent':Path(old['parent_checkpoint']),
        **{arm:study/arm/'ACTUAL_001.pt' for arm in ('paired_supervision','paired_consistency')}}
    sources=dict(registered['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    libraries={role:core.base.worker.NativeMixtures(old['preparation'],role,1) for role in ('train_pack','validation_pack')}
    indices={role:[int(i) for count in (1,2,3) for i in np.flatnonzero(data.rows['count']==count)[:2]] for role,data in libraries.items()}
    p=dict(status='REGISTERED_CPU_TRAINED_WINDOW_STABILITY',indices=indices,sources=sources,
        checkpoint_sha256={key:w.digest(path) for key,path in checkpoints.items()},
        predecessor_complete_sha256=w.digest(study/'COMPLETE.json'),
        selection='First two existing examples of each count, same paired offset rule; no result-based selection',
        alignment='Independent full-window PIT to the same reference order; diagnostic only',
        inferences=72,updates=0,gpu_use=False,heldout_read=False,dev_reused=True,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p);torch.set_num_threads(2)
    rows=[];started=time.time();done=0
    for model,path in checkpoints.items():
        net=core.base.worker.make_model('retained_unet',path).eval()
        with torch.inference_mode():
            for role,data in libraries.items():
                for index in indices[role]:
                    raw_a,raw_b,offset=paired_items(data,index);a,b=core.batch(raw_a),core.batch(raw_b)
                    left,right=regions(data.length,offset)
                    for key in ('mixture','references'):assert torch.equal(a[key][...,left],b[key][...,right])
                    pa,la=core.base.worker.predict(net,a);pb,lb=core.base.worker.predict(net,b);done+=2
                    assert torch.equal(la,lb)
                    ma,assignment_a=core.metrics(pa,a);mb,assignment_b=core.metrics(pb,b)
                    ca=canonical(pa,assignment_a)[...,left];cb=canonical(pb,assignment_b)[...,right]
                    score=float(consistency(ca,cb,a['references'][...,left],a['active'],a['mixture'][...,left]))
                    mixnorm=float((ca-cb).abs().square().mean(-1).sum()/a['mixture'][...,left].abs().square().mean())
                    n=int(a['construction_count']);refpower=a['references'][...,left].abs().square().mean(-1)[0,:n].double()
                    error=(ca-cb).abs().square().mean(-1)[0,:n].double()
                    per_source=(error/refpower.clamp_min(1e-30)).tolist()
                    assert np.isfinite([score,mixnorm,*per_source]).all()
                    rows.append(dict(model=model,role=role,index=index,count=n,offset=offset,common_samples=left.stop-left.start,
                        consistency=score,drift_over_mixture_energy=mixnorm,per_source_relative_drift=per_source,
                        original_window=ma,shifted_window=mb))
                    w.write(root/'STATE.json',dict(status='CPU_TRAINED_STABILITY',model=model,role=role,
                        inferences=done,total_inferences=72,pid=os.getpid(),time=time.time()))
        del net;gc.collect()
    summaries=[]
    for model in checkpoints:
        for role in libraries:
            for count in (1,2,3):
                selected=[r for r in rows if r['model']==model and r['role']==role and r['count']==count]
                assert len(selected)==2
                summaries.append(dict(model=model,role=role,count=count,cases=2,
                    consistency=statistics.mean(r['consistency'] for r in selected),
                    drift_over_mixture_energy=statistics.mean(r['drift_over_mixture_energy'] for r in selected),
                    original_nmse=statistics.mean(v for r in selected for v in r['original_window']['nmse']),
                    shifted_nmse=statistics.mean(v for r in selected for v in r['shifted_window']['nmse'])))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    for name,path in checkpoints.items():assert w.digest(path)==p['checkpoint_sha256'][name]
    result=dict(status='COMPLETE_CHECKED',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summaries,
        seconds=time.time()-started,updates=0,gpu_use=False,heldout_read=False,dev_reused=True,
        limitation='Six TRAIN and six reused DEV cases, reference-aligned diagnostic; not population effect or independent test')
    public=ROOT/'reports/2026-10-10';w.write(root/'COMPLETE.json',result);w.write(public/'TRAINED_WINDOW_STABILITY_RESULT.json',result)
    lines=['# 일관성 학습이 실제 창 의존성을 줄였는가','',
        '원 부모와 두 창 대조/일관성 후보의 실제e1을 비교했다. 각 개수의 앞2혼합씩 TRAIN6+DEV6, 총72회 CPU 추론이다. '
        '원래 합성 일정과 ±163.84μs의 동일한 두 창 규칙을 사용했다. 공통474.88μs의 예측을 정답 순서로 대응한 값은 진단용이며 정답을 추론에 주지 않는다.','',
        '| 자료 | 성분 수 | 모델 | 일관성 항 ↓ | 창 간 차이/혼합 에너지 ↓ | 원창 NMSE ↓ | 이동창 NMSE ↓ |',
        '|---|---:|---|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['role']}|{s['count']}|{s['model']}|{s['consistency']:.6f}|{s['drift_over_mixture_energy']:.6f}|{s['original_nmse']:.6f}|{s['shifted_nmse']:.6f}|")
    lines+=['','창 간 예측이 비슷해지는 것과 정답에 가까워지는 것은 별개다. 개수마다2사례뿐이며 반복 사용한 DEV의 기전 진단이다. '
        '실제 모델 채택은 이미 완료한630개 전체 DEV 파형 지표를 기준으로 하며, 이 사후 진단으로 바꾸지 않는다.',
        '', '[모든36행](TRAINED_WINDOW_STABILITY_RESULT.json) · [전체 DEV 결과](PAIRED_WINDOW_KO.md)']
    (public/'TRAINED_WINDOW_STABILITY_KO.md').write_text('\n'.join(lines)+'\n')
    w.write(root/'STATE.json',dict(status='COMPLETE_CHECKED',inferences=72,pid=os.getpid(),time=time.time()))
    print(dict(summary=summaries,seconds=result['seconds']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    try:run(a.run.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
