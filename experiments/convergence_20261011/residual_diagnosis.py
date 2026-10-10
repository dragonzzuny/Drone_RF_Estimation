"""Orthogonal reference-assisted error diagnosis of the completed TRAIN4 fit.

These projections use references and are diagnostic only, never inference.
Other references are orthogonalized to the target before projection, so the
three error-energy terms sum without double-counting correlated components.
"""
from pathlib import Path
import os
import time
import traceback
import torch
import run as study

ROOT=study.ROOT;w=study.w
OUT=ROOT/'local/convergence_residual_20261011_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    p=w.read(ROOT/'local/convergence_20261011_v1/PROTOCOL.json')
    final=ROOT/'local/convergence_20261011_v1/LAST.pt'
    hashes={str(Path(__file__)):w.digest(Path(__file__)),str(final):w.digest(final),p['parent_checkpoint']:w.digest(Path(p['parent_checkpoint']))}
    w.write(OUT/'PROTOCOL.json',dict(indices=p['train_indices'],files=hashes,
        labels=['target-direction error','orthogonal other-reference span','unexplained residual'],
        interpretation='reference-assisted least-squares decomposition; not causal physical leakage attribution',
        heldout_read=False,validation_read=False,optimizer_updates=0,time=time.time()))
    data=study.worker.NativeMixtures(p['preparation'],'train_pack',1);rows=[]
    for arm in ('parent','fit256'):
        net=study.original.make_model('balanced',Path(p['parent_checkpoint'])).eval()
        if arm=='fit256':
            ck=torch.load(final,map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);del ck
        for index in p['train_indices']:
            w.write(OUT/'STATE.json',dict(status='CPU_DIAGNOSIS',arm=arm,index=index,pid=os.getpid(),time=time.time()))
            raw=data[index]
            item={k:torch.as_tensor(raw[k])[None] for k in ('mixture','references','active','context_features','crop_start','construction_count')}
            with torch.no_grad():out,_=study.worker.predict(net,item)
            score=study.waveform_metrics(out,item['references'],item['active'],item['mixture'])
            count=int(item['construction_count']);r=item['references'][0,:count].to(torch.complex128).T
            estimates=out[0,:3][score['assignment'][0]][:count].to(torch.complex128).T
            for source in range(count):
                target=r[:,source];estimate=estimates[:,source];energy=target.abs().square().sum()
                gain=torch.vdot(target,estimate)/energy
                others=r[:,[j for j in range(count) if j!=source]]
                orthogonal=others-target[:,None]*((target.conj()[:,None]*others).sum(0)/energy)[None]
                rest=estimate-gain*target
                solution=torch.linalg.lstsq(orthogonal,rest[:,None],driver='gelsd').solution
                leakage=(orthogonal@solution)[:,0];residual=rest-leakage
                terms=[float((gain-1).abs().square()),float(leakage.abs().square().sum()/energy),float(residual.abs().square().sum()/energy)]
                nmse=float((estimate-target).abs().square().sum()/energy)
                assert abs(sum(terms)-nmse)<1e-8
                cross=float(abs(torch.vdot(leakage,residual))/energy)
                assert cross<1e-8
                rows.append(dict(arm=arm,index=index,count=count,source=source,nmse=nmse,
                    target_gain_abs=float(gain.abs()),target_gain_phase_deg=float(gain.angle()*180/torch.pi),
                    target_direction_error=terms[0],orthogonal_other_reference_energy=terms[1],
                    unexplained_error=terms[2],sum_error=abs(sum(terms)-nmse),cross=cross))
        del net
    for path,sha in hashes.items():assert w.digest(Path(path))==sha
    result=dict(status='COMPLETE',rows=rows,protocol_sha256=w.digest(OUT/'PROTOCOL.json'),
        optimizer_updates=0,validation_read=False,heldout_read=False,
        scope='same fixed TRAIN4; reference-assisted diagnostic only, not inferred corrections or physical noise floor',time=time.time())
    w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/CONVERGENCE_RESIDUAL_DIAGNOSIS.json',result)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
