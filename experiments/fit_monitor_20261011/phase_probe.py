"""Reuse reference-free phase averaging on the fixed TF-GridNet64 model."""
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch
import head_adapt as head

ROOT, fit, w, worker, study = head.ROOT, head.fit, head.w, head.worker, head.study
OUT = ROOT/'local/tfgridnet_phase_probe_20261011_v1'
PHASE_SOURCE = ROOT/'experiments/rfuav_phase_average_20261009/phase_core.py'
spec = importlib.util.spec_from_file_location('gridnet_phase_probe_core',PHASE_SOURCE)
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


def state(status,**fields):
    w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**fields))


def summarize(rows):
    groups=[]
    for count in (2,3):
        chosen=[r for r in rows if r['count']==count]
        nmse=[v for r in chosen for v in r['nmse']]
        groups.append(dict(count=count,cases=len(chosen),mean_nmse=float(np.mean(nmse)),
            median_nmse=float(np.median(nmse)),mean_si_sdr=float(np.mean([v for r in chosen for v in r['si_sdr']])),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in chosen])),
            max_source_nmse=max(nmse),source_nmse_ge_one=sum(v>=1 for v in nmse)))
    return dict(rows=rows,by_count=groups)


def main():
    assert w.read(ROOT/'reports/2026-10-11/TFGRIDNET_HEAD_ADAPT_AUDIT.json')['status']=='PASS'
    assert w.read(head.OUT/'COMPLETE.json')['selected']['added_updates']==0
    OUT.mkdir(exist_ok=False)
    p=copy.deepcopy(w.read(head.OUT/'PROTOCOL.json'))
    fit.verify(head.OUT,p)
    checkpoint=Path(p['parent_checkpoint'])
    for source in [Path(__file__),PHASE_SOURCE]:
        p['source_sha256'][str(source.relative_to(ROOT))]=w.digest(source)
    for rel in p['source_sha256']:
        target=OUT/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    plan=ROOT/'reports/2026-10-11/TFGRIDNET_PHASE_PROBE_PLAN_KO.md'
    p['pinned_files'][str(plan)]=w.digest(plan)
    p.update(optimizer_updates=0,training_parameters=0,inference_angles=[0,90,180,270],
             forward_passes=120,maximum_seconds=720,checkpoint_selection='fixed original64 only',
             success='NMSE2/3 down, SI2/3 up, weakest and median NMSE2/3 nonworse',
             selection='No checkpoint/model selection or incumbent replacement; fixed inference diagnostic',
             stop_rule='Complete all30 paired cases or fail at12minutes; no optimizer',registered_at=time.time())
    w.write(OUT/'PROTOCOL.json',p);ph=w.digest(OUT/'PROTOCOL.json');fit.verify(OUT,p)
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    original=w.read(head.COMPARATOR/'ADDED_000.json')
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
        signal.alarm(720);assert torch.cuda.is_available();torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net=head.base.build().cuda().eval();net.load_state_dict(saved['model']);net.requires_grad_(False);del saved
        for module in net.modules():
            if isinstance(module,head.base.ChunkedLSTM):module.chunk=p['lstm_chunk']
        baseline=[];averaged=[];diagnostics=[];began=time.time()
        for case,index in enumerate(p['probe_indices']):
            state('GPU_INFERENCE',case=case+1,total=30,seconds=time.time()-began)
            item=worker.fit.base.batch([data[index]])
            with torch.no_grad():
                single,mean,logits,diagnostic=phase.phase_predictions(net,worker.predict,item)
                for output,destination in [(single,baseline),(mean,averaged)]:
                    m=study.waveform_metrics(output,item['references'],item['active'],item['mixture'])
                    active=item['active'][0]
                    row=copy.deepcopy(original['rows'][case]);assert row['index']==index
                    row.update(nmse=m['nmse'][0][active].tolist(),si_sdr=m['si_sdr'][0][active].tolist(),
                        assignment=m['assignment'][0].tolist(),sum_relative_error=float(m['sum_relative_error'][0]),
                        predicted_count=int(logits.argmax(-1)[0])+1)
                    assert row['reference_power']==m['reference_power'][0][active].tolist()
                    assert np.isfinite(row['nmse']+row['si_sdr']).all() and row['sum_relative_error']<1e-9
                    if destination is baseline:
                        np.testing.assert_allclose(row['nmse'],original['rows'][case]['nmse'],rtol=1e-5,atol=1e-6)
                        np.testing.assert_allclose(row['si_sdr'],original['rows'][case]['si_sdr'],rtol=1e-5,atol=1e-4)
                    destination.append(row)
            diagnostics.append(dict(index=index,**diagnostic))
            w.write(OUT/'PARTIAL.json',dict(baseline=baseline,averaged=averaged,diagnostics=diagnostics))
        first,last=summarize(baseline),summarize(averaged)
        checks=head.compare(first,last)
        candidate=all(c['nmse'] and c['si'] and c['weak'] and c['median'] for c in checks)
        fit.verify(OUT,p)
        result=dict(status='COMPLETE',baseline=first,averaged=last,diagnostics=diagnostics,checks=checks,
            inference_improvement_candidate=candidate,protocol_sha256=ph,
            checkpoint_sha256=w.digest(checkpoint),seconds=time.time()-began,forward_passes=len(baseline)*4,
            optimizer_updates=0,validation_read=False,heldout_read=False,incumbent_replaced=False)
        w.write(OUT/'COMPLETE.json',result)
        w.write(ROOT/'reports/2026-10-11/TFGRIDNET_PHASE_PROBE_RESULT.json',result)
        state('COMPLETED',cases=len(baseline),inference_improvement_candidate=candidate)


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Phase probe12minute cap')))
    try:
        main()
    except Exception:
        if OUT.exists():
            w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED')
        raise
