"""Check loss-gradient interaction in actual parent parameter space, CPU only."""
from pathlib import Path
import os
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.losses import pit_waveform_loss
w=worker.watch
OUT=ROOT/'local/convergence_parameter_geometry_20261011_v1'


def relation(a,b):
    an=float(a.double().norm());bn=float(b.double().norm())
    return dict(cosine=float(torch.dot(a.double(),b.double())/max(an*bn,1e-30)),
                nmse_norm=an,coherence_norm=bn,coherence_to_nmse_norm=bn/max(an,1e-30))


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    p=w.read(ROOT/'local/convergence_20261011_v1/PROTOCOL.json')
    files={str(Path(__file__)):w.digest(Path(__file__)),p['parent_checkpoint']:w.digest(Path(p['parent_checkpoint']))}
    protocol=dict(indices=p['train_indices'],parent_sha256=files[p['parent_checkpoint']],
        sources=p['source_sha256'],files=files,optimizer_updates=0,device='cpu',
        purpose='local parameter gradients of normalized MSE+background versus coherence; count CE excluded',
        heldout_read=False,validation_iq_read=False,time=time.time())
    w.write(OUT/'PROTOCOL.json',protocol)
    net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).eval()
    params=list(net.parameters());data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    rows=[];sn=sc=None;began=time.time()
    for index in p['train_indices']:
        w.write(OUT/'STATE.json',dict(status='RUNNING',pid=os.getpid(),cases=len(rows),index=index,time=time.time()))
        raw=data[index]
        item={k:torch.as_tensor(raw[k])[None] for k in ('mixture','references','active','context_features','crop_start','construction_count')}
        out,_=worker.predict(net,item);refs,active,mix=item['references'],item['active'],item['mixture']
        score=pit_waveform_loss(out,refs,active,mix)
        aligned=out[:,:3].gather(1,score['assignment'][...,None].expand_as(refs))
        mixp=mix.abs().square().mean(-1).clamp_min(1e-8);powers=refs.abs().square().mean(-1)
        denominator=torch.where(active,torch.maximum(powers,1e-6*mixp[:,None]),mixp[:,None])
        nmse=((aligned-refs).abs().square().mean(-1)/denominator).mean()+score['background_loss']
        ec=aligned-aligned.mean(-1,keepdim=True);rc=refs-refs.mean(-1,keepdim=True)
        coh=((1-((ec*rc.conj()).mean(-1).abs().square()/
            (ec.abs().square().mean(-1)*rc.abs().square().mean(-1)).clamp_min(1e-16)).clamp(0,1))*
            (active & (powers>1e-6*mixp[:,None]))).mean()
        torch.testing.assert_close(nmse+coh,score['loss'],rtol=1e-5,atol=1e-6)
        gn=torch.autograd.grad(nmse,params,retain_graph=True,allow_unused=True)
        a=torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(t).flatten() for g,t in zip(gn,params)])
        del gn
        gc=torch.autograd.grad(coh,params,allow_unused=True)
        b=torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(t).flatten() for g,t in zip(gc,params)])
        del gc
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if sn is None:sn=torch.zeros_like(a);sc=torch.zeros_like(b)
        sn.add_(a/4);sc.add_(b/4)
        rows.append(dict(index=index,count=int(item['construction_count']),**relation(a,b)))
        w.write(OUT/'PARTIAL.json',rows)
        del a,b,out,score,nmse,coh,aligned,ec,rc
    for path,sha in files.items():assert w.digest(Path(path))==sha
    for path,sha in p['source_sha256'].items():assert w.digest(ROOT/path)==sha
    result=dict(status='COMPLETE',rows=rows,batch=relation(sn,sc),parameters=sum(x.numel() for x in params),
        protocol_sha256=w.digest(OUT/'PROTOCOL.json'),optimizer_updates=0,heldout_read=False,validation_iq_read=False,
        seconds=time.time()-began,scope='initial local TRAIN4 gradients; no causal generalization claim',time=time.time())
    w.write(OUT/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-11/CONVERGENCE_PARAMETER_GEOMETRY.json',result)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    try:run()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
