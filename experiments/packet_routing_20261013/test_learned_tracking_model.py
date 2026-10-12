"""Preparation checks, not an RF training result."""
from pathlib import Path
import torch
from learned_tracking_model import TrackingEmbedding,affinity_loss
from sequence_data import ROOT,write_json,sha256


def run():
    torch.set_num_threads(1);torch.manual_seed(0)
    net=TrackingEmbedding().eval();x=torch.rand(2,7,3,513)
    orders=torch.stack([torch.randperm(3) for _ in range(7)])
    with torch.no_grad():
        expected=net(x);changed=net(x[:,torch.arange(7)[:,None],orders])
        torch.testing.assert_close(changed,expected[:,torch.arange(7)[:,None],orders],rtol=1e-5,atol=1e-6)
        assert torch.allclose(expected.norm(dim=-1),torch.ones(2,7,3),atol=1e-6)
    labels=torch.arange(3)[None,None,:].expand(2,7,3).clone()
    weights=torch.ones(2,7,3)/21
    a=affinity_loss(net(x),labels,weights)
    renamed=labels.clone();renamed[labels==0]=2;renamed[labels==2]=0
    b=affinity_loss(net(x),renamed,weights);torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert any(p.grad.abs().max()>0 for p in net.parameters())
    opt=torch.optim.AdamW(net.parameters(),lr=2.5e-4);opt.step()
    write_json(ROOT/'reports/2026-10-13/LEARNED_TRACKING_MODEL_CPU_CHECK.json',
        dict(status='PASS',parameters=sum(p.numel() for p in net.parameters()),
             within_block_permutation_equivariance=True,source_label_renaming_invariance=True,
             finite_gradient_all_parameters=True,synthetic_update_discarded=True,rf_training_updates=0,
             source_sha256=sha256(Path(__file__).with_name('learned_tracking_model.py')),test_sha256=sha256(__file__)))
    print('TRACKING_MODEL_CPU_PASS',sum(p.numel() for p in net.parameters()),flush=True)


if __name__=='__main__':run()
