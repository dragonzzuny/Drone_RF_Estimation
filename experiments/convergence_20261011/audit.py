"""CPU reproduction of the final bounded TRAIN4 diagnosis, never heldout."""
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import run as study

ROOT=study.ROOT
RUN=ROOT/'local/convergence_20261011_v1'
OUT=ROOT/'local/convergence_audit_20261011_v1'
w=study.w


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    began=time.time()
    w.write(OUT/'PROTOCOL.json',dict(worker_sha256=w.digest(Path(__file__)),
        scope='CPU final-checkpoint and all ten TRAIN components reproduction',
        nmse_rtol=1e-4,nmse_atol=1e-6,si_rtol=1e-4,si_atol_db=1e-3,
        optimizer_updates=0,heldout_read=False,registered_at=began))
    while not (RUN/'COMPLETE.json').exists():
        if (RUN/'FAILURE.json').exists():raise RuntimeError('Training failed')
        if time.time()-began>3600:raise TimeoutError('Training receipt absent')
        w.write(OUT/'STATE.json',dict(status='WAITING_TRAINING',pid=os.getpid(),time=time.time()))
        time.sleep(15)
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    w.write(OUT/'STATE.json',dict(status='CPU_AUDIT',pid=os.getpid(),time=time.time()))
    p=w.read(RUN/'PROTOCOL.json');complete=w.read(RUN/'COMPLETE.json')
    study.verify(RUN,p)
    assert complete['protocol_sha256']==w.digest(RUN/'PROTOCOL.json')
    assert complete['history_sha256']==w.digest(RUN/'HISTORY.json')
    assert complete['checkpoint_sha256']==w.digest(RUN/'LAST.pt')
    ck=torch.load(RUN/'LAST.pt',map_location='cpu',weights_only=False)
    assert ck['updates']==complete['updates'] and ck['protocol_sha256']==complete['protocol_sha256']
    assert {int(x['step']) for x in ck['optimizer']['state'].values()}=={complete['updates']}
    for s in ck['optimizer']['state'].values():
        for value in s.values():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    net=study.original.make_model('balanced',Path(p['parent_checkpoint'])).eval()
    net.load_state_dict(ck['model']);del ck
    assert all(torch.isfinite(x).all() for x in net.parameters())
    assert sum(x.numel() for x in net.parameters())==p['parameters']
    data=study.worker.NativeMixtures(p['preparation'],'train_pack',1)
    items=[]
    for index in p['train_indices']:
        raw=data[index]
        items.append({k:torch.as_tensor(raw[k])[None] for k in ('mixture','references','active','context_features','crop_start','construction_count')})
    actual=study.evaluate(net,items,data,p['train_indices'])
    delta={'nmse':0.,'si_sdr':0.}
    for a,b in zip(actual['rows'],complete['final']['rows']):
        for key in ('index','count','categories','pack_ids','weakest_index','assignment','predicted_count'):
            assert a[key]==b[key],key
        np.testing.assert_allclose(a['reference_power'],b['reference_power'],rtol=1e-12,atol=1e-15)
        for key,tol,atol in [('nmse',1e-4,1e-6),('si_sdr',1e-4,1e-3)]:
            np.testing.assert_allclose(a[key],b[key],rtol=tol,atol=atol)
            delta[key]=max(delta[key],float(np.max(np.abs(np.array(a[key])-b[key]))))
    for point in complete['history']:
        assert sum(len(x['nmse']) for x in point['rows'])==10
        for group in point['by_count']:
            rows=[r for r in point['rows'] if r['count']==group['count']]
            expected=dict(mean_nmse=np.mean([v for r in rows for v in r['nmse']]),
                mean_si_sdr=np.mean([v for r in rows for v in r['si_sdr']]),
                weakest_nmse=np.mean([r['nmse'][r['weakest_index']] for r in rows]),
                max_source_nmse=max(v for r in rows for v in r['nmse']))
            for key,value in expected.items():assert abs(value-group[key])<1e-12
    study.verify(RUN,p)
    result=dict(status='PASS',updates=complete['updates'],parameters=p['parameters'],
        cpu_final=actual,gpu_cpu_max_abs=delta,all_history_summaries_recomputed=True,
        checkpoint_sha256=complete['checkpoint_sha256'],training_protocol_sha256=complete['protocol_sha256'],
        audit_protocol_sha256=w.digest(OUT/'PROTOCOL.json'),optimizer_updates=0,
        heldout_read=False,validation_read=False,time=time.time())
    w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/CONVERGENCE_AUDIT.json',result)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
