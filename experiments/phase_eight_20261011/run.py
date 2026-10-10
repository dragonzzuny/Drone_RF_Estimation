"""Fixed one-shot C4/C8 DEV comparison after audited incumbent selection."""
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import core

ROOT=core.ROOT
sys.path.insert(0,str(ROOT/'experiments/ordered_branch_20261011'))
import architecture as model
import evaluation as stats
from drone_rf.waveform import waveform_metrics
from study import finite_values,value_status
w,worker=model.w,model.worker
OUT=ROOT/'local/phase_eight_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def state(status,**kw):
    w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))
    w.write(PUBLIC/'PHASE_EIGHT_LIVE_KO.md',f"# 8위상 추론 실제 상태\n\n{status}: {kw.get('case',0)}/630. "
        '가중치 학습0회, 같은 가중치의4위상 대8위상. 자동 채팅 보고가 아니다.\n')


def verify(p):
    for rel,h in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==h and w.digest(OUT/'source_snapshot'/rel)==h,rel
    for f,h in p['pinned_files'].items():assert w.digest(Path(f))==h,f


def register():
    OUT.mkdir(exist_ok=False)
    previous=ROOT/'local/ordered_branch_20261011_v1'
    audit=w.read(PUBLIC/'ORDERED_BRANCH_AUDIT.json');assert audit['status']=='PASS'
    result=w.read(previous/'COMPLETE.json');assert audit['candidate']==result['candidate']
    old=w.read(previous/'PROTOCOL.json')
    if result['candidate']:
        checkpoint=previous/'ACTUAL_001.pt';baseline=previous/'FOUR_PHASE_001.json';kind='ordered_branch'
    else:
        checkpoint=Path(old['parent_checkpoint']);baseline=Path(old['four_phase_baseline']);kind='native_parent'
    check=w.read(PUBLIC/'PHASE_EIGHT_CPU_CHECK.json');assert check['status']=='PASS'
    sources=dict(old['source_sha256'])
    for path in Path(__file__).parent.glob('*.py'):sources[str(path.relative_to(ROOT))]=w.digest(path)
    for rel,h in check['source_sha256'].items():assert sources[rel]==h
    for rel,h in sources.items():
        assert w.digest(ROOT/rel)==h
        dest=OUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    pinned=[checkpoint,baseline,PUBLIC/'ORDERED_BRANCH_AUDIT.json',previous/'COMPLETE.json',
        PUBLIC/'PHASE_EIGHT_CPU_CHECK.json',PUBLIC/'PHASE_EIGHT_PLAN_KO.md']
    data=worker.NativeMixtures(old['preparation'],'validation_pack',1)
    p=dict(model_kind=kind,checkpoint=str(checkpoint),checkpoint_sha256=w.digest(checkpoint),
        baseline=str(baseline),preparation=old['preparation'],dev_rows_sha256=data.rows_hash,
        source_sha256=sources,pinned_files={str(f):w.digest(f) for f in pinned},
        cases=630,forward_passes=5040,optimizer_updates=0,angles=[0,45,90,135,180,225,270,315],
        precision='FP32 TF32off',maximum_seconds=900,
        criterion='NMSE2/3 lower, complexSI2/3 higher, weakestNMSE2/3 nonworse',
        heldout_read=False,independent_test=False,registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p);verify(p)
    return p,data


@torch.no_grad()
def infer(p,data):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if p['model_kind']=='native_parent':net=worker.make_model('retained_unet',Path(p['checkpoint']))
    else:
        old=w.read(ROOT/'local/ordered_branch_20261011_v1/PROTOCOL.json')
        net=model.build(Path(old['parent_checkpoint']))
        net.load_state_dict(torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['model'])
    net.cuda().eval().requires_grad_(False)
    baseline=w.read(Path(p['baseline']));rows4=[];rows8=[];started=time.time()
    maximum_nmse_difference=0.;maximum_si_difference=0.
    for i in range(len(data)):
        if i%25==0:state('GPU_INFERENCE',case=i,total=630,seconds=time.time()-started)
        item=worker.fit.base.batch([data[i]])
        four,eight,logits,diag=core.predict_eight(net,worker.predict,item)
        for estimates,destination in ((four,rows4),(eight,rows8)):
            m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
            active=item['active'][0];power=m['reference_power'][0][active]
            row=copy.deepcopy(baseline['rows'][i])
            assert row['index']==i and row['count']==int(item['construction_count'][0])
            assert row['reference_power']==power.tolist()
            si=m['si_sdr'][0][active];gain=si-m['input_si_sdr'][0][active]
            row.update(assignment=m['assignment'][0].tolist(),nmse=finite_values(m['nmse'][0][active]),
                si_sdr=finite_values(si),si_status=value_status(si),si_sdr_gain=finite_values(gain),gain_status=value_status(gain),
                predicted_count=int(logits.argmax(-1)[0])+1,inactive_leak=float(m['inactive_leak'][0].sum()),
                background_nmse=float(m['background_nmse'][0]),sum_relative_error=float(m['sum_relative_error'][0]),
                phase_details=diag if destination is rows8 else diag['four_phase'])
            assert row['predicted_count']==baseline['rows'][i]['predicted_count']
            if destination is rows4:
                a=baseline['rows'][i]
                np.testing.assert_allclose(row['nmse'],a['nmse'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(row['si_sdr'],a['si_sdr'],rtol=1e-5,atol=1e-4)
                maximum_nmse_difference=max(maximum_nmse_difference,max(abs(x-y) for x,y in zip(row['nmse'],a['nmse'])))
                maximum_si_difference=max(maximum_si_difference,max(abs(x-y) for x,y in zip(row['si_sdr'],a['si_sdr'])))
            destination.append(row)
    four,eight=stats.summarize(rows4,epoch=0),stats.summarize(rows8,epoch=0)
    checks=stats.compare(eight,four)
    candidate=all(all(v for k,v in row.items() if k!='count') for row in checks)
    verify(p)
    r=dict(status='COMPLETE',four_phase=four,eight_phase=eight,checks=checks,candidate=candidate,
        model_kind=p['model_kind'],checkpoint_sha256=p['checkpoint_sha256'],
        seconds=time.time()-started,forward_passes=5040,optimizer_updates=0,
        maximum_baseline_nmse_difference=maximum_nmse_difference,maximum_baseline_si_difference=maximum_si_difference,
        protocol_sha256=w.digest(OUT/'PROTOCOL.json'),heldout_read=False,independent_test=False)
    w.write(OUT/'COMPLETE.json',r);w.write(PUBLIC/'PHASE_EIGHT_RESULT.json',r)
    del net;torch.cuda.empty_cache()


def main():
    p,data=register()
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);signal.alarm(900)
        assert torch.cuda.is_available();infer(p,data);signal.alarm(0)
    state('CPU_AUDITING',case=630)
    subprocess.run(['taskset','-c','12,13',sys.executable,str(Path(__file__).with_name('audit.py'))],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    state('COMPLETE_AUDITED',case=630)


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('15minute inference cap')))
    try:main()
    except Exception:
        if OUT.exists():
            w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED')
        raise
