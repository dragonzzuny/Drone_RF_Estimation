"""Check reference-blind selection and exact two-pass gradient algebra."""
import hashlib
import json
from pathlib import Path
import torch
from successive import predict,strongest_source

def run():
    torch.manual_seed(0)
    x=torch.randn(2,37,dtype=torch.complex128)
    a=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    seen=[]
    def base(parameter,item):
        assert set(item)=={'mixture','context_features','crop_start'}
        seen.append(item['mixture'])
        z=item['mixture']
        # First source is strictly largest for this parameter range.
        return torch.stack((parameter*z,.1*z,.05*z,(.85-parameter)*z),1),torch.zeros(2,3,dtype=torch.float64)
    batch=dict(mixture=x,context_features=torch.zeros(2,65,255),crop_start=torch.zeros(2,dtype=torch.long),
        references='forbidden',active='forbidden',construction_count='forbidden')
    output,_,trace=predict(a,batch,base)
    assert len(seen)==2 and seen[1].grad_fn is not None
    assert float((output.sum(1)-x).detach().abs().max())<1e-14
    expected=torch.stack((a*x,a*(1-a)*x,(1-a).square()*x,torch.zeros_like(x)),1)
    assert float((output-expected).detach().abs().max())<1e-14
    target=torch.randn(2,4,37,dtype=torch.complex128)
    objective=(output-target).abs().square().sum()
    gradient=torch.autograd.grad(objective,a)[0]
    derivative=torch.stack((x,(1-2*a)*x,-2*(1-a)*x,torch.zeros_like(x)),1)
    analytic=(2*(expected-target).conj()*derivative).real.sum()
    difference=float((gradient-analytic).detach().abs())
    assert difference<1e-10
    # Permuting candidate source slots leaves the selected waveform invariant;
    # a background with larger energy must never be chosen.
    p=torch.stack((.2*x,3*x,.5*x,20*x),1)
    chosen,index=strongest_source(p)
    changed,newindex=strongest_source(p[:,[2,0,1,3]])
    assert torch.equal(chosen,changed) and bool((index==1).all()) and bool((newindex==2).all())
    result=dict(status='PASS',two_pass_gradient_absolute_error=difference,
        analytic_gradient=float(analytic.detach()),autograd_gradient=float(gradient),
        exact_additive_conservation=True,reference_blind_whitelist=True,
        output_permutation_invariance_away_from_ties=True,background_excluded=True,
        optimizer_steps=0,full_model_test=False,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('successive.py')]})
    out=Path(__file__).resolve().parents[2]/'reports/2026-10-10/SUCCESSIVE_ALGEBRA_CHECK.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':run()
