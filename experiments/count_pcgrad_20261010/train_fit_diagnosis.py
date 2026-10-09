"""CPU-only fixed TRAIN48 fit for completed PCGrad/CAGrad checkpoints.

Reuse the prior TRAIN48 selection, parent/control rows and audit. No new subset
selection, model updates or heldout access. These are development diagnostics,
not independent samples or a generalization-gap estimator.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('count_training_for_fit',HERE/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from drone_rf.waveform import waveform_metrics
w=base.w;ROOT=base.ROOT


def evaluate(net,raw):
    item={k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
          ('mixture','references','active','context_features','crop_start','construction_count')}
    with torch.inference_mode():
        estimates,logits=base.worker.predict(net,item)
        m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
    n=int(raw['construction_count']);power=m['reference_power'][0,:n].tolist()
    result=dict(index=int(raw['index']),count=n,nmse=m['nmse'][0,:n].tolist(),
        si_sdr=m['si_sdr'][0,:n].tolist(),reference_power=power,
        weakest_index=int(np.argmin(power)),predicted_count=int(logits.argmax(-1))+1,
        sum_relative_error=float(m['sum_relative_error'][0]))
    assert np.isfinite(result['nmse']+result['si_sdr']).all() and result['sum_relative_error']<1e-10
    return result


def run(root,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    prior_path=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    prior=w.read(prior_path);audit=w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT_AUDIT.json')
    assert prior['status']=='COMPLETE' and audit['status']=='PASS'
    studies=[]
    for key in ('pcgrad','cagrad'):
        folder=ROOT/f'local/count_{key}_20261010_v1';p=w.read(folder/'PROTOCOL.json')
        a=w.read(ROOT/f'reports/2026-10-10/COUNT_{key.upper()}_FINAL_AUDIT.json')
        assert a['status']=='PASS' and a['complete_sha256']==w.digest(folder/'COMPLETE.json')
        studies.append(dict(arm=key,folder=str(folder),protocol=p,
            protocol_sha256=w.digest(folder/'PROTOCOL.json'),checkpoint_sha256=w.digest(folder/'ACTUAL_001.pt')))
    sources=dict(studies[0]['protocol']['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_CPU_COUNT_TRAIN_FIT',indices=prior['train_indices'],
        source_sha256=sources,prior_sha256=w.digest(prior_path),
        checkpoints={s['arm']:s['checkpoint_sha256'] for s in studies},
        selection='Unchanged previous TRAIN48',updates=0,heldout_read=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    torch.set_num_threads(2)
    preparation=studies[0]['protocol']['preparation']
    assert preparation==studies[1]['protocol']['preparation']
    train=base.worker.NativeMixtures(preparation,'train_pack',1)
    validation=base.worker.NativeMixtures(preparation,'validation_pack',1)
    rows=[r for r in prior['rows'] if r['model'] in ('parent/e0','retained_unet/e1')]
    backend=[];started=time.time();done=0
    for study in studies:
        folder=Path(study['folder']);saved=torch.load(folder/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==study['protocol_sha256'] and saved['epoch']==1
        net=base.worker.make_model('retained_unet').eval();net.load_state_dict(saved['model']);del saved
        gpu=w.read(folder/'VALIDATION_001.json')['rows']
        for count in (1,2,3):
            expected=next(r for r in gpu if r['count']==count);actual=evaluate(net,validation[expected['index']])
            assert actual['predicted_count']==expected['predicted_count']
            error=max(abs(a-b) for a,b in zip(actual['nmse'],expected['nmse']))
            si_error=max(abs(a-b) for a,b in zip(actual['si_sdr'],expected['si_sdr']))
            assert error<2e-5 and si_error<2e-3
            backend.append(dict(model=study['arm'],index=expected['index'],nmse_max_error=error,si_sdr_max_error=si_error))
        for index in protocol['indices']:
            value=evaluate(net,train[index]);value['model']=study['arm']+'/e1';rows.append(value);done+=1
            w.write(root/'PARTIAL.json',dict(rows=rows,backend_checks=backend))
            w.write(root/'STATE.json',dict(status='CPU_TRAIN_FIT',arm=study['arm'],cases=done,total=96,pid=os.getpid(),time=time.time()))
        del net
    summaries=[]
    for model in ('parent/e0','retained_unet/e1','pcgrad/e1','cagrad/e1'):
        for count in (1,2,3):
            part=[r for r in rows if r['model']==model and r['count']==count]
            summaries.append(dict(model=model,count=count,cases=len(part),
                mean_nmse=statistics.mean(v for r in part for v in r['nmse']),
                mean_si_sdr=statistics.mean(v for r in part for v in r['si_sdr']),
                weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in part)))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),
        rows=rows,summaries=summaries,backend_checks=backend,seconds=time.time()-started,
        model_updates=0,heldout_read=False,independent_test=False,
        limitation='Fixed small TRAIN48 diagnosis; different power/activity distribution from DEV630, not a pure generalization-gap estimate')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=96,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds'],summaries=summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.public.resolve())
