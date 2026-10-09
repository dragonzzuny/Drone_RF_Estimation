"""Algebra, gradient and target invariance checks without recorded I/Q."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from affinity import targets,loss


def run(output):
    torch.manual_seed(0);torch.set_num_threads(2)
    p=torch.rand(2,3,4,5,dtype=torch.float64);p[0,2]=0;p[:,:,0,0]=0
    raw=torch.randn(2,20,7,dtype=torch.float64,requires_grad=True)
    e=F.normalize(raw,dim=-1);checks=[]
    for kind in ('hard','soft'):
        y,w=targets(p,kind)
        assert torch.allclose(w.sum(-1),torch.ones(2,dtype=w.dtype),atol=1e-14)
        assert torch.equal(w[:,0],torch.zeros_like(w[:,0]))
        low=loss(e,y,w)
        full=(((e@e.transpose(1,2)-y@y.transpose(1,2))**2)*w[:,:,None]*w[:,None,:]).sum((1,2)).mean()
        gl=torch.autograd.grad(low,raw,retain_graph=True)[0]
        gf=torch.autograd.grad(full,raw,retain_graph=True)[0]
        assert abs(float(low-full))<1e-12 and torch.allclose(gl,gf,atol=1e-12,rtol=1e-10)
        yp,wp=targets(p[:,[2,0,1]],kind)
        assert abs(float(loss(e,yp,wp)-low))<1e-12
        ys,ws=targets(p*7.3,kind)
        assert abs(float(loss(e,ys,ws)-low))<1e-12
        ypad,wpad=targets(torch.cat((p,torch.zeros_like(p[:,:1])),1),kind)
        assert abs(float(loss(e,ypad,wpad)-low))<1e-12
        assert abs(float(loss(y,y,w)))<1e-12
        checks.append(dict(kind=kind,low_rank_loss=float(low.detach()),value_error=abs(float((low-full).detach())),
            gradient_max_error=float((gl-gf).abs().max()),source_permutation_invariant=True,
            common_scale_invariant=True,inactive_padding_invariant=True,zero_when_embedding_matches=True))
    # Soft labeling retains an overlapping weak component that hard argmax loses.
    example=torch.tensor([[[[.99]],[[.01]]]],dtype=torch.float64)
    y,_=targets(example,'soft');hard,_=targets(example,'hard')
    assert abs(float(y[0,0,1])-.1)<1e-12 and hard[0,0,1]==0
    empty=torch.zeros_like(p);ey,ew=targets(empty,'soft');assert loss(e,ey,ew)==0
    paths=[Path(__file__),Path(__file__).with_name('affinity.py')]
    repo=Path(__file__).resolve().parents[2]
    result=dict(status='PASS',checks=checks,overlap_example_soft_weak_coefficient=.1,
        no_recorded_iq=True,gpu_use=False,model_updates=0,
        source_sha256={str(q.relative_to(repo)):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths})
    output.write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
