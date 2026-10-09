"""Synthetic CPU proof checks, followed by one full-size model backward."""
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from canonical import CanonicalPhaseSeparator, phase_anchor


def main():
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(73)
    z = torch.randn(2,32,48,dtype=torch.complex128)
    # An intentionally non-equivariant function: canonicalization, not the
    # underlying network, must provide equivariance.
    class Toy(torch.nn.Module):
        def forward(self,z,context,start):
            raw=torch.stack((z.real+2j,z.imag+.3j,z.square(),torch.ones_like(z)),1)
            estimates=raw+(z-raw.sum(1))[:,None]/4
            return dict(estimates=estimates,count_logits=context.mean(-1)[:,:3])
    toy=CanonicalPhaseSeparator(Toy())
    context=torch.randn(2,65,8,dtype=torch.float64)
    start=torch.zeros(2,dtype=torch.long)
    baseline=toy(z,context,start)
    for angle in (0.1,0.9,1.5707963267948966,3.0,5.7):
        rotation=torch.exp(torch.tensor(angle*1j,dtype=torch.complex128))
        result=toy(z*rotation,context,start)
        torch.testing.assert_close(result['estimates'],baseline['estimates']*rotation,rtol=1e-10,atol=1e-11)
        torch.testing.assert_close(result['count_logits'],baseline['count_logits'],rtol=0,atol=0)
    for magnitude in (0.,1e-30,1.,1e20):
        value=z*magnitude
        phase,valid,_=phase_anchor(value)
        assert torch.isfinite(phase).all()
        torch.testing.assert_close(value/phase*phase,value,rtol=1e-12,atol=1e-40 if magnitude<1 else 1e-10)
        if magnitude==0:
            assert not valid.any() and torch.count_nonzero(toy(value,context,start)['estimates'])==0
    del toy,baseline,result,z
    refs=torch.randn(1,3,63872,dtype=torch.complex64)
    item=dict(mixture=refs.sum(1),references=refs,active=torch.ones(1,3,dtype=torch.bool),
        context_features=torch.randn(1,65,255),crop_start=torch.tensor([4097]))
    model=CanonicalPhaseSeparator(worker.make_model('retained_unet')).eval()
    assert sum(p.numel() for p in model.parameters())==32142859
    with torch.no_grad():
        expected,counts=worker.predict(model,item)
        rotation=torch.exp(torch.tensor(.731j,dtype=torch.complex64))
        rotated=dict(item,mixture=item['mixture']*rotation)
        actual,new_counts=worker.predict(model,rotated)
        error=float((actual-expected*rotation).abs().square().sum()/expected.abs().square().sum())
        assert error<1e-9
        torch.testing.assert_close(counts,new_counts,rtol=0,atol=0)
    del expected,actual,new_counts,counts,rotated
    model.train()
    output,logits=worker.predict(model,item)
    loss=worker.pit_waveform_loss(output,refs,item['active'],item['mixture'])['loss']+.1*torch.nn.functional.cross_entropy(logits,torch.tensor([2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    sum_error=float(((output.sum(1)-item['mixture']).abs().square().sum()/item['mixture'].abs().square().sum()).detach())
    assert sum_error<1e-9
    paths=(Path(__file__),Path(__file__).with_name('canonical.py'))
    result=dict(status='PASS',source_sha256={str(p.relative_to(ROOT)):worker.watch.digest(p) for p in paths},
        parameters=32142859,added_parameters=0,analytic_non_equivariant_toy_rotations_checked=5,
        finite_zero_and_extreme_anchors=True,full_63872_sample_forward_backward=True,
        full_model_rotation_relative_error=error,sum_relative_error=sum_error,
        all_parameter_gradients_finite=True,count_logits_rotation_invariant=True,
        device='cpu',recorded_iq_reads=0,parent_checkpoint_reads=0,model_updates=0,
        seconds=time.time()-started,limitation='Synthetic correctness check; no RF improvement demonstrated')
    worker.watch.write(ROOT/'reports/2026-10-10/CANONICAL_PHASE_CPU_CHECK.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
