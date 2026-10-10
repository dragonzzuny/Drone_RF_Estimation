"""CPU-only output-space loss diagnosis on the same four TRAIN examples."""
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.losses import pit_waveform_loss
from drone_rf.waveform import waveform_metrics
w=worker.watch
OUT=ROOT/'local/convergence_geometry_20261011_v1'


def cosine(a,b):
    return float((a.conj()*b).real.sum()/(a.abs().square().sum().sqrt()*b.abs().square().sum().sqrt()).clamp_min(1e-30))


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    p=w.read(ROOT/'local/convergence_20261011_v1/PROTOCOL.json')
    w.write(OUT/'STATE.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).eval()
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    rows=[];start=time.time()
    for index in p['train_indices']:
        raw=data[index]
        item={k:torch.as_tensor(raw[k])[None] for k in ('mixture','references','active','context_features','crop_start','construction_count')}
        with torch.no_grad():out,logits=worker.predict(net,item)
        output=out.detach().requires_grad_(True)
        refs,active,mix=item['references'],item['active'],item['mixture']
        score=pit_waveform_loss(output,refs,active,mix)
        assignment=score['assignment'];aligned=output[:,:3].gather(1,assignment[...,None].expand_as(refs))
        mix_power=mix.abs().square().mean(-1).clamp_min(1e-8)
        powers=refs.abs().square().mean(-1)
        floor=torch.maximum(powers,1e-6*mix_power[:,None])
        denominator=torch.where(active,floor,mix_power[:,None])
        nmse=((aligned-refs).abs().square().mean(-1)/denominator).mean()+score['background_loss']
        ec=aligned-aligned.mean(-1,keepdim=True);rc=refs-refs.mean(-1,keepdim=True)
        cross=(ec*rc.conj()).mean(-1).abs().square()
        var=ec.abs().square().mean(-1)*rc.abs().square().mean(-1)
        meaningful=active & (powers>1e-6*mix_power[:,None])
        coherence=((1-(cross/var.clamp_min(1e-16)).clamp(0,1))*meaningful).mean()
        torch.testing.assert_close(nmse+coherence,score['loss'],rtol=1e-5,atol=1e-6)
        gn,=torch.autograd.grad(nmse,output,retain_graph=True)
        gc,=torch.autograd.grad(coherence,output)
        pn=gn-gn.mean(1,keepdim=True);pc=gc-gc.mean(1,keepdim=True)
        perms=list(itertools.permutations(range(3)))
        costs=[float(((output[:,:3][:,q]-refs).abs().square().mean(-1)/denominator).mean().detach()) for q in perms]
        best=perms[min(range(6),key=costs.__getitem__)]
        metric=waveform_metrics(out,refs,active,mix)
        count=int(item['construction_count'])
        rows.append(dict(index=index,count=count,nmse=metric['nmse'][active].tolist(),si_sdr=metric['si_sdr'][active].tolist(),
            reference_power=metric['reference_power'][active].tolist(),loss_nmse_background=float(nmse.detach()),
            loss_coherence=float(coherence.detach()),output_gradient_cosine=cosine(gn,gc),
            sum_preserving_gradient_cosine=cosine(pn,pc),
            nmse_gradient_norm=float(pn.abs().square().sum().sqrt()),coherence_gradient_norm=float(pc.abs().square().sum().sqrt()),
            nmse_only_assignment=list(best),training_assignment=assignment[0].tolist(),
            assignments_equal=list(best)==assignment[0].tolist()))
        w.write(OUT/'STATE.json',dict(status='RUNNING',cases=len(rows),total=4,pid=os.getpid(),time=time.time()))
        w.write(OUT/'PARTIAL.json',rows)
    assert not torch.cuda.is_initialized()
    result=dict(status='COMPLETE',rows=rows,cases=4,optimizer_updates=0,heldout_read=False,
        validation_iq_read=False,gpu_initialized=False,parent_checkpoint_sha256=w.digest(Path(p['parent_checkpoint'])),
        source_sha256=w.digest(Path(__file__)),scope='output-space geometry only; not parameter-gradient conflicts or DEV causality',
        seconds=time.time()-start,time=time.time())
    w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/CONVERGENCE_CPU_GEOMETRY.json',result)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',cases=4,pid=None,time=time.time()))


if __name__=='__main__':
    try:run()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
