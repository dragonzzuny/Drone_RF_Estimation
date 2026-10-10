"""No-update TRAIN-only application check outside four repeatedly fitted mixtures."""
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tfgridnet_rf_20261011'))
import fit
from model import build,ChunkedLSTM
w,study,worker=fit.w,fit.convergence,fit.worker
TRAIN=ROOT/'local/tfgridnet_continuation_20261011_v1'
ROUTING=ROOT/'local/tfgridnet_routing_20261011_v2'
OUT=ROOT/'local/tfgridnet_outside_fit_20261011_v1'


def state(status,**values):
    w.write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**values))


def groups(rows):
    result=[]
    for subset in ('fit4','outside_fit4'):
        for count in (2,3):
            selected=[r for r in rows if r['subset']==subset and r['count']==count]
            if not selected:continue
            nmse=[n for r in selected for n in r['nmse']]
            result.append(dict(subset=subset,count=count,mixtures=len(selected),sources=len(nmse),
                mean_nmse=float(np.mean(nmse)),mean_si_sdr=float(np.mean([v for r in selected for v in r['si_sdr']])),
                mean_si_sdri=float(np.mean([v for r in selected for v in r['si_sdri']])),
                weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in selected])),
                max_source_nmse=max(nmse),source_nmse_ge_one=sum(v>=1 for v in nmse)))
    return result


def main():
    OUT.mkdir(exist_ok=False)
    plan=ROOT/'reports/2026-10-11/TFGRIDNET_OUTSIDE_FIT_PLAN_KO.md'
    pins={str(p):w.digest(p) for p in [plan,Path(__file__)]}
    w.write(OUT/'REGISTRATION.json',dict(files=pins,indices_rule='first48 TRAIN schedule1; count2 or3',
        fit_indices=[4,5,2,11],optimizer_updates=0,validation_read=False,heldout_read=False,time=time.time()))
    while not all((p/'COMPLETE.json').exists() for p in [TRAIN,ROUTING]):
        for p in [TRAIN,ROUTING]:
            if (p/'FAILURE.json').exists():raise RuntimeError(f'Prerequisite failed: {p.name}')
        state('WAITING_PREREQUISITES');time.sleep(15)
    with worker.fit.base.LOCK.open('r') as lock:
        state('WAITING_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX);signal.alarm(1200)
        assert torch.cuda.is_available();torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        p=w.read(TRAIN/'PROTOCOL.json');fit.verify(TRAIN,p)
        chosen=w.read(TRAIN/'COMPLETE.json')['selected'];ckpath=Path(chosen['checkpoint'])
        assert w.digest(ckpath)==chosen['checkpoint_sha256'];pins[str(ckpath)]=w.digest(ckpath)
        data=worker.NativeMixtures(p['preparation'],'train_pack',1)
        indices=[i for i in range(48) if int(data.rows[i]['count']) in (2,3)]
        assert set(p['train_indices']).issubset(indices)
        w.write(OUT/'PROTOCOL.json',dict(files=pins,indices=indices,fit_indices=p['train_indices'],
            source_rows_sha256=data.rows_hash,selected_update=chosen['step'],selected_checkpoint_sha256=chosen['checkpoint_sha256'],
            preparation=p['preparation'],parameters=p['parameters'],chunk=p['lstm_chunk'],
            validation_read=False,heldout_read=False,optimizer_updates=0,time=time.time()))
        net=build().cuda().eval();ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);assert ck['updates']==chosen['step'];del ck
        for module in net.modules():
            if isinstance(module,ChunkedLSTM):module.chunk=p['lstm_chunk']
        rows=[];began=time.time()
        for i,index in enumerate(indices):
            state('GPU_INFERENCE',case=i+1,total=len(indices),index=index,selected_update=chosen['step'])
            item=worker.fit.base.batch([data[index]])
            with torch.no_grad():
                out,_=worker.predict(net,item)
                m=study.waveform_metrics(out,item['references'],item['active'],item['mixture'])
            active=item['active'][0];count=int(item['construction_count'])
            nm=m['nmse'][0][active].tolist();si=m['si_sdr'][0][active].tolist()
            input_si=m['input_si_sdr'][0][active].tolist();powers=m['reference_power'][0][active].tolist()
            assert np.isfinite(nm+si+input_si).all() and float(m['sum_relative_error'][0])<1e-9
            clips=[data.library.clips[int(k)] for k in data.rows[index]['indices'][:count]]
            row=dict(index=index,count=count,subset='fit4' if index in p['train_indices'] else 'outside_fit4',
                categories=[c['category'] for c in clips],pack_ids=[c['pack_id'] for c in clips],
                nmse=nm,si_sdr=si,input_si_sdr=input_si,si_sdri=[a-b for a,b in zip(si,input_si)],
                reference_power=powers,weakest_index=int(np.argmin(powers)),
                assignment=m['assignment'][0].tolist(),sum_relative_error=float(m['sum_relative_error'][0]))
            if row['subset']=='fit4':
                reference=next(r for r in chosen['rows'] if r['index']==index)
                np.testing.assert_allclose(nm,reference['nmse'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(si,reference['si_sdr'],rtol=1e-5,atol=1e-4)
            rows.append(row);w.write(OUT/'PARTIAL.json',dict(rows=rows,groups=groups(rows)))
        for path,sha in pins.items():assert w.digest(Path(path))==sha
        fit.verify(TRAIN,p)
        result=dict(status='COMPLETE',rows=rows,groups=groups(rows),seconds=time.time()-began,
            selected_update=chosen['step'],selected_checkpoint_sha256=chosen['checkpoint_sha256'],
            protocol_sha256=w.digest(OUT/'PROTOCOL.json'),optimizer_updates=0,validation_read=False,heldout_read=False,
            scope='TRAIN mixtures outside four fitted examples; recording/sample overlap possible; not new-recording validation')
        w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_OUTSIDE_FIT_RESULT.json',result)
        state('COMPLETED',cases=len(rows))


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Outside-fit check time budget')))
    signal.alarm(3600)
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
