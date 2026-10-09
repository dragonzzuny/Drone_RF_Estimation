"""Synthetic full-model preflight of the balanced input candidate."""
import copy
import itertools
from pathlib import Path
import time
import torch
from torch.nn import functional as F
import train_comparison as worker
from balanced_source_head import BalancedSourceInteractionHead, augment
from drone_rf.losses import pit_waveform_loss

w = worker.watch


def main():
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(327)
    head = BalancedSourceInteractionHead(torch.nn.Conv2d(64, 8, 1), chunk_points=17).double()
    f = torch.randn(19, 64, dtype=torch.float64, requires_grad=True)
    r = torch.randn(19, 3, 2, dtype=torch.float64, requires_grad=True)
    torch.nn.init.normal_(head.readout.weight, std=.03)
    expected = head.correction(f, r)
    for order in itertools.permutations(range(3)):
        torch.testing.assert_close(head.correction(f, r[:, list(order)]), expected[:, list(order)], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(expected.sum(1), torch.zeros_like(expected[:, 0]), rtol=0, atol=1e-12)
    for magnitude in (0., 1e-30, 1., 1e6):
        tiny = (r.detach()*magnitude).requires_grad_()
        y = head.correction(f, tiny)
        grad, = torch.autograd.grad(y.square().mean(), tiny)
        assert torch.isfinite(y).all() and torch.isfinite(grad).all()
    h = torch.randn(2, 64, 5, 9, dtype=torch.float64, requires_grad=True)
    other = copy.deepcopy(head);other.checkpoint_chunks=False;other.chunk_points=1000
    h2 = h.detach().clone().requires_grad_()
    a, b = head(h), other(h2)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(a[:, 6:], head.base(h)[:, 6:], rtol=0, atol=0)
    a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-8, atol=1e-12)
    for (name, p), (name2, q) in zip(head.named_parameters(), other.named_parameters()):
        assert name == name2 and p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-12)
    del head, other, h, h2, a, b, f, r, expected
    torch.manual_seed(19)
    refs = torch.randn(1, 3, 63872, dtype=torch.complex64)
    refs *= torch.tensor([1., .316227766, .1])[None, :, None]
    item = dict(mixture=refs.sum(1), references=refs, active=torch.ones(1,3,dtype=torch.bool),
        context_features=torch.randn(1,65,255), crop_start=torch.tensor([4097]))
    net=worker.make_model('retained_unet').eval()
    with torch.no_grad():
        baseline, logits0 = worker.predict(net,item)
    augment(net)
    assert sum(p.numel() for p in net.parameters())==32_180_747
    with torch.no_grad():
        actual, logits = worker.predict(net,item)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(logits, logits0, rtol=0, atol=0)
    del baseline, actual, logits, logits0
    net.train()
    estimate, logits = worker.predict(net,item)
    loss=pit_waveform_loss(estimate,refs,item['active'],item['mixture'])['loss']+.1*F.cross_entropy(logits,torch.tensor([2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.output.readout.weight.grad.norm()>0
    assert net.output.qkv.weight.grad.norm()==0
    relative=float(((estimate.sum(1)-item['mixture']).abs().square().sum()/item['mixture'].abs().square().sum()).detach())
    assert relative<1e-9
    result=dict(status='PASS', source_sha256={str(p.relative_to(w.ROOT)):w.digest(p) for p in (Path(__file__),Path(__file__).with_name('balanced_source_head.py'))},
        parameters=32_180_747, additional_parameters=37_888, matches_unbalanced_head_parameter_count=True,
        exact_initial_parent_waveform_and_count_logits=True, nonzero_readout_source_permutations=6,
        zero_sum_correction=True, background_unchanged=True, chunk_output_and_gradients_match=True,
        finite_zero_and_extreme_raw_source_gradients=True, all_full_model_gradients_finite=True,
        first_readout_gradient_norm=float(net.output.readout.weight.grad.norm()),
        first_qkv_gradient_norm=float(net.output.qkv.weight.grad.norm()), sum_relative_error=relative,
        model_updates=0, recorded_iq_reads=0, parent_checkpoint_reads=0, device='cpu',
        seconds=time.time()-started, rf_performance_evaluated=False)
    w.write(w.ROOT/'reports/2026-10-10/BALANCED_SOURCE_HEAD_CPU_CHECK.json', result)
    print(result,flush=True)

if __name__=='__main__':
    main()
