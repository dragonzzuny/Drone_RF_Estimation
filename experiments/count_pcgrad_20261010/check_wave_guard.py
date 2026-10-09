"""Check removal of auxiliary gradients and actual-displacement constraints."""
import hashlib
import json
from pathlib import Path
import torch
from wave_update_guard import WaveAccumulator,correct
from update_projection import check as check_projection
from check import run as check_accumulation


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.context_encoder=torch.nn.Linear(3,4)
        self.wave=torch.nn.Linear(4,2);self.count_head=torch.nn.Linear(4,3)
    def forward(self,x):
        encoded=self.context_encoder(x);return self.wave(encoded),self.count_head(encoded)


def run():
    torch.manual_seed(0);torch.set_num_threads(2)
    base_check=check_accumulation();projection_check=check_projection()
    net=Toy().double();acc=WaveAccumulator(net.named_parameters());params=list(net.parameters())
    x,y=torch.randn(3,dtype=torch.float64),torch.randn(2,dtype=torch.float64)
    wave,count=net(x);wl=(wave-y).square().mean();ce=.1*torch.nn.functional.cross_entropy(count[None],torch.tensor([1]))
    target=torch.autograd.grad(wl/32,params,retain_graph=True,allow_unused=True)
    target=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(target,params)])
    ce_grad=torch.autograd.grad(ce/32,acc.ce_parameters,retain_graph=True,allow_unused=True)
    acc.collect_ce(2,ce_grad);((wl+ce)/32).backward();acc.collect(2)
    error=float((acc.waveform_groups()[0]-target).abs().max());assert error<1e-14
    p=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64));before=p.detach().clone()
    with torch.no_grad():p.copy_(torch.tensor([1.,2.,3.,4.],dtype=torch.float64))
    h=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]],dtype=torch.float64)
    actual,receipt=correct([p],before,h)
    assert torch.allclose(actual,torch.tensor([0.,0.,3.,4.],dtype=torch.float64))
    assert torch.equal(actual,p.detach())
    # Ordinary optimizer state is not touched by the projection helper.
    assert receipt['protected_dot_after_fp32']==[0.,0.]
    here=Path(__file__).resolve().parent;root=here.parents[1];hashes=dict(base_check['source_sha256'])
    for path in (here/'wave_update_guard.py',here/'update_projection.py',Path(__file__)):
        hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(status='PASS',waveform_gradient_recovery_max_error=error,
        actual_displacement_projection_checked=True,projection_contract=projection_check,
        source_sha256=hashes,gpu_use=False,recorded_iq_reads=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
