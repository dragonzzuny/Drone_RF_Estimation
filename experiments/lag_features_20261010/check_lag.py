"""Synthetic RF-lag identities and full-model backward, no recorded data."""
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.waveform import analyze
from lag_input import lag_features,augment,LAGS


def main():
    started=time.time();torch.set_num_threads(2);torch.manual_seed(41)
    length=63872
    t=torch.arange(length,dtype=torch.float64)
    omega=2*torch.pi*17/512
    tone=torch.exp(1j*omega*t)[None]
    z=analyze(tone);power,phase=lag_features(z)
    rotated=lag_features(z*torch.exp(torch.tensor(.872j,dtype=torch.complex128)))
    for a,b in zip((power,phase),rotated):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    # A positive time lag produces exp(+j omega tau) in Z conj(Z_delayed).
    for j,lag in enumerate(LAGS):
        frame=(lag+512)//128+2
        actual=torch.complex(phase[0,2*j,17,frame],phase[0,2*j+1,17,frame])
        expected=torch.exp(torch.tensor(1j*omega*lag,dtype=torch.complex128))
        torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
        assert torch.count_nonzero(phase[:,:,:,:(LAGS[0]+256+127)//128])==0
        assert torch.count_nonzero(phase[:,2*j:2*j+2,:,:((lag+256+127)//128)])==0
    zero=lag_features(torch.zeros_like(z))
    assert all(torch.isfinite(v).all() and torch.count_nonzero(v)==0 for v in zero)
    del z,power,phase,rotated,zero,tone,t
    refs=torch.randn(1,3,length,dtype=torch.complex64)
    item=dict(mixture=refs.sum(1),references=refs,active=torch.ones(1,3,dtype=torch.bool),
        context_features=torch.randn(1,65,255),crop_start=torch.tensor([4097]))
    model=worker.make_model('retained_unet').eval()
    with torch.no_grad():initial,counts=worker.predict(model,item)
    augment(model,'lag_power_phase')
    assert sum(q.numel() for q in model.parameters())==32149771
    with torch.no_grad():actual,newcounts=worker.predict(model,item)
    torch.testing.assert_close(actual,initial,rtol=0,atol=0)
    torch.testing.assert_close(newcounts,counts,rtol=0,atol=0)
    model.train();output,logits=worker.predict(model,item)
    loss=worker.pit_waveform_loss(output,refs,item['active'],item['mixture'])['loss']+.1*torch.nn.functional.cross_entropy(logits,torch.tensor([2]))
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    branch=model.down[0][0]
    assert branch.power.weight.grad.norm()>0 and branch.phase.weight.grad.norm()>0
    sum_error=float(((output.sum(1)-item['mixture']).abs().square().sum()/item['mixture'].abs().square().sum()).detach())
    assert sum_error<1e-9
    result=dict(status='PASS',source_sha256={str(p.relative_to(ROOT)):worker.watch.digest(p) for p in (Path(__file__),Path(__file__).with_name('lag_input.py'))},
        lags_samples=LAGS,lags_us=[x/100 for x in LAGS],complex_parameters=32149771,power_control_parameters=32146315,
        analytic_tone_phase_sign_checked=True,common_phase_invariant_features=True,zero_features_finite=True,
        unavailable_lag_and_reflect_edges_masked=True,exact_parent_predictions_at_zero_initialization=True,
        full_input_backward_finite=True,power_gradient_norm=float(branch.power.weight.grad.norm()),phase_gradient_norm=float(branch.phase.weight.grad.norm()),
        sum_relative_error=sum_error,model_updates=0,recorded_iq_reads=0,gpu_use=False,seconds=time.time()-started,
        limitation='Synthetic implementation checks only; no drone-separation performance demonstrated')
    worker.watch.write(ROOT/'reports/2026-10-10/LAG_FEATURE_CPU_CHECK.json',result);print(result,flush=True)


if __name__=='__main__':main()
