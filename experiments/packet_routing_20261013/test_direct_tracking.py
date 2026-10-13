"""Check exact expected I/Q cost, permutation symmetries, and common initialization."""
from pathlib import Path
import numpy as np
import torch
from direct_tracking_model import DirectTracking,waveform_cost,expected_waveform_loss,decode
from learned_tracking_model import TrackingEmbedding
from routing_statistics import collect
from spectral_routing import PERMS,WINDOW
from evaluate_parent_full_context import ROOT,write,sha


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(0);base=TrackingEmbedding()
    torch.manual_seed(0);net=DirectTracking()
    assert all(torch.equal(value,net.encoder.state_dict()[key]) for key,value in base.state_dict().items())
    assert sum(p.numel() for p in net.parameters())==875905
    rng=np.random.default_rng(73);checks=[]
    for k in (1,2,3):
        n=2*WINDOW+7
        refs=(rng.standard_normal((3,n))+1j*rng.standard_normal((3,n))).astype(np.complex64)
        refs[k:]=0;mixture=refs.sum(0)
        output=np.zeros((4,n),np.complex64)
        output[:3]=refs*.7+(rng.standard_normal((3,n))+1j*rng.standard_normal((3,n)))*.2
        output[3]=mixture-output[:3].sum(0)
        cache=collect(output,refs,mixture);cost=waveform_cost(cache,k)
        choice=[4,1,5];logits=torch.full((1,3,6),-40.)
        for t,q in enumerate(choice):logits[0,t,q]=40
        actual=float(expected_waveform_loss(logits,torch.from_numpy(cost)[None]))
        routed=np.empty_like(output[:3])
        for t,(a,b) in enumerate(cache['bounds']):routed[:,a:b]=output[list(PERMS[choice[t]]),a:b]
        values=[]
        for mapping in PERMS:
            value=0.
            for j,r in enumerate(mapping):
                error=np.sum(abs(routed[j].astype(np.complex128)-refs[r])**2)
                value+=error/(k*np.sum(abs(refs[r].astype(np.complex128))**2)) if r<k else .1*error/((3-k)*np.sum(abs(mixture.astype(np.complex128))**2))
            values.append(value)
        np.testing.assert_allclose(actual,min(values),rtol=1e-6,atol=1e-7)
        checks.append(dict(count=k,expected_hard_choice_loss=actual,direct_iq_loss=float(min(values))))
    torch.manual_seed(5);observed=torch.randn(2,33,3,513)
    net.eval();logits=net(observed)
    slot_order=np.array([2,0,1]);permuted=net(observed[:,:,slot_order])
    original_orders,_=decode(logits);new_orders,_=decode(permuted)
    np.testing.assert_array_equal(slot_order[new_orders],original_orders)
    # Global reference order does not affect the PIT minimum (all sources active).
    small=rng.random((33,3,3));stats=dict(sse=small,energy_r=np.ones((33,3)),energy_m=np.ones(33)*3,
        bounds=np.zeros((33,2),dtype=int))
    target=waveform_cost(stats,3)
    changed=dict(stats,sse=small[:,:,[2,0,1]],energy_r=stats['energy_r'][:,[2,0,1]])
    loss=expected_waveform_loss(logits,torch.from_numpy(np.stack([target,target])))
    perm_loss=expected_waveform_loss(logits,torch.from_numpy(np.stack([waveform_cost(changed,3)]*2)))
    torch.testing.assert_close(loss,perm_loss)
    loss.backward()
    assert torch.isfinite(net.identity.weight.grad).all() and net.identity.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    write(ROOT/'reports/2026-10-13/DIRECT_TRACKING_CPU_CHECK.json',dict(status='PASS',parameters=875905,
        common_encoder_initialization_exact=True,hard_choice_cost_matches_direct_iq=checks,
        input_slot_permutation_equivariant=True,global_reference_order_invariant=True,
        finite_nonzero_head_gradient=True,source_sha256=sha(Path(__file__).with_name('direct_tracking_model.py')),
        check_source_sha256=sha(Path(__file__))))
    print('DIRECT_TRACKING_CPU_CHECK PASS',checks)


if __name__=='__main__':main()
