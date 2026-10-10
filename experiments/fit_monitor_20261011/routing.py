"""Reference-assisted output-role diagnosis; never an inference correction."""
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
from model import build
w,study,worker=fit.w,fit.convergence,fit.worker
RUN=ROOT/'local/tfgridnet_fit_20261011_v1'
OUT=ROOT/'local/tfgridnet_routing_20261011_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    p=w.read(RUN/'PROTOCOL.json');fit.verify(RUN,p)
    pins={str(path):w.digest(path) for path in [RUN/'LAST.pt',Path(__file__),
        ROOT/'reports/2026-10-11/TFGRIDNET_ROUTING_PLAN_KO.md']}
    w.write(OUT/'PROTOCOL.json',dict(files=pins,indices=p['train_indices'],optimizer_updates=0,
        validation_read=False,heldout_read=False,maximum_seconds=900,registered_at=time.time()))
    ck=torch.load(RUN/'LAST.pt',map_location='cpu',weights_only=False);assert ck['updates']==32
    net=build().eval();net.load_state_dict(ck['model']);del ck
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    published=w.read(RUN/'COMPLETE.json')['final']['rows'];rows=[];maxnm=maxsi=0.
    for index,prior in zip(p['train_indices'],published):
        w.write(OUT/'STATE.json',dict(status='CPU_INFERENCE',index=index,pid=os.getpid(),time=time.time()))
        raw=data[index];item={k:torch.as_tensor(raw[k])[None] for k in (
            'mixture','references','active','context_features','crop_start','construction_count')}
        with torch.no_grad():out,_=worker.predict(net,item)
        options=[]
        for background in range(4):
            order=[j for j in range(4) if j!=background]+[background];permuted=out[:,order]
            with torch.no_grad():
                loss=study.original.prior.pit_waveform_loss(permuted,item['references'],item['active'],item['mixture'])
                metrics=study.waveform_metrics(permuted,item['references'],item['active'],item['mixture'])
            assignment=[order[int(j)] for j in metrics['assignment'][0]]
            options.append(dict(background_output=background,reference_outputs=assignment,
                loss=float(loss['loss']),nmse=metrics['nmse'][item['active']].tolist(),
                si_sdr=metrics['si_sdr'][item['active']].tolist(),sum_relative_error=float(metrics['sum_relative_error'][0])))
        original=options[3]
        np.testing.assert_allclose(original['nmse'],prior['nmse'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(original['si_sdr'],prior['si_sdr'],rtol=1e-5,atol=1e-4)
        maxnm=max(maxnm,max(abs(a-b) for a,b in zip(original['nmse'],prior['nmse'])))
        maxsi=max(maxsi,max(abs(a-b) for a,b in zip(original['si_sdr'],prior['si_sdr'])))
        count=int(item['construction_count']);e=out[0].to(torch.complex128)
        r=item['references'][0,:count].to(torch.complex128)
        e=e-e.mean(-1,keepdim=True);r=r-r.mean(-1,keepdim=True)
        coh=(e@r.conj().T).abs().square()/(e.abs().square().sum(-1)[:,None]*r.abs().square().sum(-1)[None]).clamp_min(1e-16)
        chosen=min(options,key=lambda v:v['loss'])
        rows.append(dict(index=index,count=count,categories=prior['categories'],options=options,
            original=original,oracle_role_choice=chosen,role_changed=chosen['background_output']!=3,
            complex_coherence_outputs_by_references=coh.tolist(),output_power=out[0].abs().square().mean(-1).tolist()))
        w.write(OUT/'PARTIAL.json',rows)
    for path,sha in pins.items():assert w.digest(Path(path))==sha
    fit.verify(RUN,p)
    result=dict(status='COMPLETE',rows=rows,maximum_cpu_gpu_nmse_difference=maxnm,
        maximum_cpu_gpu_si_difference_db=maxsi,checkpoint_sha256=pins[str(RUN/'LAST.pt')],
        protocol_sha256=w.digest(OUT/'PROTOCOL.json'),independent_full_length_cpu_inference=True,
        optimizer_updates=0,validation_read=False,heldout_read=False,
        scope='reference-assisted TRAIN4 output-role diagnosis; not deployable inference',time=time.time())
    w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_ROUTING_RESULT.json',result)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('15 minute CPU budget')))
    signal.alarm(900)
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        w.write(OUT/'STATE.json',dict(status='FAILED',pid=None,time=time.time()));raise
