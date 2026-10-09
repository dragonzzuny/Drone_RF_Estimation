"""Check complex quadratic identity and split-backward shared-weight gradients."""
import hashlib
import json
from pathlib import Path
import torch
from paired import consistency,regions,canonical


def run():
    torch.set_num_threads(2);torch.manual_seed(0)
    x=torch.randn(2,4,257,dtype=torch.complex128)
    y=torch.randn_like(x);target=torch.randn_like(x)
    lhs=((x-target).abs().square()+(y-target).abs().square()).mean()/2
    rhs=((x+y)/2-target).abs().square().mean()+(x-y).abs().square().mean()/4
    assert abs(float(lhs-rhs))<1e-12
    reference=torch.randn(2,3,257,dtype=torch.complex128)
    reference[0,2]=0;active=torch.tensor([[True,True,False],[True,True,True]])
    mix=reference.sum(1)
    weight=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
    def prediction(t):
        return torch.einsum('ij,bjt->bit',weight.to(torch.complex128),t)
    a,b=prediction(x),prediction(y)
    direct=consistency(a,b,reference,active,mix)
    grad_direct,=torch.autograd.grad(direct,weight)
    a,b=prediction(x),prediction(y)
    left=consistency(a,b.detach(),reference,active,mix)
    grad_left,=torch.autograd.grad(left,weight)
    a,b=prediction(x),prediction(y)
    right=consistency(a.detach(),b,reference,active,mix)
    grad_right,=torch.autograd.grad(right,weight)
    relative=float((grad_direct-grad_left-grad_right).norm()/grad_direct.norm())
    assert relative<1e-12
    # A source permutation is harmless when supervised assignments transform.
    assignment=torch.tensor([[2,1,0],[0,2,1]])
    ordered=canonical(x,assignment)
    inverse=torch.argsort(torch.tensor([1,2,0]))
    assert torch.equal(ordered,canonical(x[:,[1,2,0,3]],inverse[assignment]))
    for off in (-16384,16384):
        first,second=regions(63872,off)
        u=torch.arange(200000)
        assert torch.equal(u[50000:113872][first],u[50000+off:113872+off][second])
    assert float(consistency(x,x,reference,active,mix))==0
    sources={str(Path(__file__).relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str(Path(__file__).with_name('paired.py').relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(Path(__file__).with_name('paired.py').read_bytes()).hexdigest()}
    return dict(status='PASS',complex_mse_identity_error=abs(float(lhs-rhs)),
        sequential_gradient_relative_error=relative,permutation_and_overlap_coordinates_checked=True,
        source_sha256=sources,recorded_iq_reads=0,gpu_use=False,training_run=False,
        limitation='Algebra and synthetic shared-weight gradient check, not a full-separator training or performance result')


if __name__=='__main__':
    result=run();out=Path(__file__).resolve().parents[2]/'reports/2026-10-10/PAIRED_WINDOW_ALGEBRA_CHECK.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result))
