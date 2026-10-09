"""Analytic checks and a full-capacity synthetic CPU backward; no recorded IQ."""
from pathlib import Path
import time
import torch
from objective import objective, spectral_terms, worker, ROOT


def main():
    start=time.time(); torch.set_num_threads(2); torch.manual_seed(83)
    refs=torch.randn(2,3,2048,dtype=torch.complex128)
    active=torch.tensor([[True,True,True],[True,False,False]])
    refs=refs*active[...,None]; mix=refs.sum(1)
    item=dict(references=refs,active=active,mixture=mix,construction_count=active.sum(1))
    ids=torch.arange(3).expand(2,-1)
    exact=torch.cat((refs,torch.zeros_like(refs[:,:1])),1)
    perfect=spectral_terms(exact,refs,active,mix,ids)
    assert perfect['loss']==0
    shrink=spectral_terms(exact*.5,refs,active,mix,ids)
    torch.testing.assert_close(shrink['relative_l1'][active],torch.full((4,),.5,dtype=torch.float64))
    torch.testing.assert_close(shrink['magnitude_nmse'][active],torch.full((4,),.25,dtype=torch.float64))
    rotated=spectral_terms(exact*1j,refs,active,mix,ids)
    assert float(rotated['loss'])<1e-12
    torch.testing.assert_close(rotated['spectral_complex_nmse'][active],torch.full((4,),2.,dtype=torch.float64))
    estimate=(exact+.2*torch.randn_like(exact)).requires_grad_();logits=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    result=objective(estimate,logits,item)
    loss0=objective(estimate,logits,item,0)['loss']
    original=worker.pit_waveform_loss(estimate,refs,active,mix)['loss']+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
    torch.testing.assert_close(loss0,original,rtol=0,atol=0)
    g0=torch.autograd.grad(loss0,(estimate,logits),retain_graph=True)
    g1=torch.autograd.grad(original,(estimate,logits),retain_graph=True)
    for a,b in zip(g0,g1):torch.testing.assert_close(a,b,rtol=0,atol=0)
    permuted=estimate[:,[2,0,1,3]]
    torch.testing.assert_close(objective(permuted,logits,item)['loss'],result['loss'],rtol=1e-12,atol=1e-12)
    terms=spectral_terms(estimate,refs,active,mix,result['assignment'])
    torch.testing.assert_close((terms['magnitude_nmse']+terms['phase_interaction'])[active],terms['spectral_complex_nmse'][active],rtol=1e-12,atol=1e-12)
    scaled=spectral_terms(estimate*7,refs*7,active,mix*7,result['assignment'])
    torch.testing.assert_close(scaled['loss'],terms['loss'],rtol=1e-12,atol=1e-12)
    result['loss'].backward();assert torch.isfinite(estimate.grad).all()
    refs=torch.randn(1,3,63872,dtype=torch.complex64)
    item=dict(mixture=refs.sum(1),references=refs,active=torch.ones(1,3,dtype=torch.bool),
              context_features=torch.randn(1,65,255),crop_start=torch.tensor([4097]),construction_count=torch.tensor([3]))
    net=worker.make_model('retained_unet').train()
    assert sum(p.numel() for p in net.parameters())==32142859
    output,logits=worker.predict(net,item);loss=objective(output,logits,item);loss['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    paths=(Path(__file__),Path(__file__).with_name('objective.py'))
    report=dict(status='PASS',source_sha256={str(p.relative_to(ROOT)):worker.watch.digest(p) for p in paths},
        identity_zero=True,half_amplitude_relative_l1=.5,half_amplitude_magnitude_nmse=.25,
        quarter_turn_magnitude_loss_zero=True,quarter_turn_spectral_complex_nmse=2.,
        inactive_sources_excluded=True,zero_weight_original_loss_and_gradients_exact=True,
        whole_window_permutation_invariant=True,spectral_error_decomposition_exact=True,
        common_gain_invariance=True,full_parameters=32142859,full_samples=63872,
        all_gradients_finite=True,recorded_iq_reads=0,gpu_use=False,updates=0,seconds=time.time()-start)
    worker.watch.write(ROOT/'reports/2026-10-10/MAGNITUDE_CPU_CHECK.json',report);print(report,flush=True)


if __name__=='__main__':main()
