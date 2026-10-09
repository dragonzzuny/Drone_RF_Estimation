"""Check the sole loss change; unchanged full model was already CPU-validated."""
from pathlib import Path
import torch
import objective as original
import selective


def main():
    torch.set_num_threads(2);torch.manual_seed(19)
    n=4096;refs=torch.randn(3,3,n,dtype=torch.complex128)
    active=torch.tensor([[1,0,0],[1,1,0],[1,1,1]],dtype=torch.bool)
    refs=refs*active[...,None];mix=refs.sum(1)
    estimates=torch.randn(3,4,n,dtype=torch.complex128,requires_grad=True)
    logits=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    item=dict(references=refs,active=active,mixture=mix,construction_count=active.sum(-1))
    total=selective.objective(estimates,logits,item);parts=[]
    for i in range(3):
        one={k:v[i:i+1] for k,v in item.items()}
        old=original.objective(estimates[i:i+1],logits[i:i+1],one)
        new=selective.objective(estimates[i:i+1],logits[i:i+1],one)
        expected=old['main'] if i==0 else old['loss']
        torch.testing.assert_close(new['loss'],expected,rtol=0,atol=0)
        actual_grad=torch.autograd.grad(new['loss'],(estimates,logits),retain_graph=True)
        expected_grad=torch.autograd.grad(expected,(estimates,logits),retain_graph=True)
        for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)
        parts.append(new['loss'])
    torch.testing.assert_close(total['loss'],torch.stack(parts).mean(),rtol=1e-12,atol=1e-12)
    assert torch.isfinite(total['loss'])
    p=Path(__file__);w=selective.w
    result=dict(status='PASS',source_sha256={str(x.relative_to(selective.base.ROOT)):w.digest(x) for x in (p,p.with_name('selective.py'))},
        single_source_exact_original_loss_and_gradients=True,two_three_exact_all_count_magnitude_loss_and_gradients=True,
        heterogeneous_microbatch_averaging_checked=True,parameters_changed=0,model_updates=0,
        device='cpu',recorded_iq_reads=0,heldout_read=False,
        full_model_check_reused='MAGNITUDE_CPU_CHECK.json; no architecture or inference change')
    w.write(selective.CHECK,result);print(result,flush=True)


if __name__=='__main__':main()
