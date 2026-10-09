"""Full-capacity synthetic preflight; no recorded RF or parent weights read."""
import copy
import gc
import itertools
import resource
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/architecture_audit_20261009'))
import fit_diagnostic
import watch_epochs as watch
from models import build, predict
from drone_rf.losses import pit_waveform_loss
from source_head import SourceInteractionHead, augment


def main():
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(327)
    head = SourceInteractionHead(torch.nn.Conv2d(64, 8, 1),
                                 chunk_points=17).double()
    h = torch.randn(2, 64, 5, 9, dtype=torch.float64, requires_grad=True)
    base = head.base(h)
    torch.testing.assert_close(head(h), base, rtol=0, atol=0)
    # Nonzero readout: otherwise permutation and gradient tests would be vacuous.
    torch.nn.init.normal_(head.readout.weight, std=.03)
    flat = torch.randn(19, 64, dtype=torch.float64, requires_grad=True)
    sources = torch.randn(19, 3, 2, dtype=torch.float64, requires_grad=True)
    original = head.correction(flat, sources)
    assert original.abs().sum() > 0
    for order in itertools.permutations(range(3)):
        changed = head.correction(flat, sources[:, list(order)])
        torch.testing.assert_close(changed, original[:, list(order)], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(original.sum(1), torch.zeros_like(original[:, 0]),
                               rtol=0, atol=1e-12)
    # Checkpointed chunking must agree with one unchunked computation, including
    # gradients. Chunking changes memory use, never the mathematical attention.
    other = copy.deepcopy(head)
    other.chunk_points = 1000
    other.checkpoint_chunks = False
    h2 = h.detach().clone().requires_grad_()
    y, y2 = head(h), other(h2)
    torch.testing.assert_close(y, y2, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(y[:, 6:], base[:, 6:], rtol=0, atol=0)
    torch.testing.assert_close(y[:, :6].reshape(2, 3, 2, 5, 9).sum(1),
        base[:, :6].reshape(2, 3, 2, 5, 9).sum(1), rtol=1e-10, atol=1e-12)
    y.square().mean().backward(); y2.square().mean().backward()
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-9, atol=1e-12)
    for (name, p), (name2, q) in zip(head.named_parameters(), other.named_parameters()):
        assert name == name2 and p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-12)
    assert head.qkv.weight.grad.norm() > 0
    del head, other, h, h2, y, y2, base, flat, sources, original, changed
    gc.collect()

    # Retain the ACTUAL full 64/128/256/512/1024 U-Net and actual 63872 samples.
    torch.manual_seed(19)
    refs = torch.randn(1, 3, 63872, dtype=torch.complex64)
    refs *= torch.tensor([1., .316227766, .1])[None, :, None]
    item = dict(mixture=refs.sum(1), references=refs,
        active=torch.ones(1, 3, dtype=torch.bool),
        context_features=torch.randn(1, 65, 255), crop_start=torch.tensor([4097]))
    net = build('unet_mean').eval()
    baseline_parameters = sum(p.numel() for p in net.parameters())
    assert baseline_parameters == 32_142_859
    with torch.no_grad():
        expected, expected_count = predict(net, item)
    augment(net)
    parameters = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        actual, actual_count = predict(net, item)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_count, expected_count, rtol=0, atol=0)
    del actual, actual_count, expected, expected_count
    net.train()
    estimates, logits = predict(net, item)
    objective = pit_waveform_loss(estimates, refs, item['active'], item['mixture'])['loss']
    objective = objective + .1 * F.cross_entropy(logits, torch.tensor([2]))
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    readout_norm = float(net.output.readout.weight.grad.norm())
    assert readout_norm > 0
    assert any(p.grad.abs().sum() > 0 for p in net.down.parameters())
    # Before the first readout update, upstream branch gradients are expected
    # to be zero. Report that deliberately, not as fully trained new features.
    upstream_first_gradient = float(net.output.qkv.weight.grad.norm())
    assert upstream_first_gradient == 0
    relative = float(((estimates.sum(1)-item['mixture']).abs().square().sum()
                     /item['mixture'].abs().square().sum()).detach())
    assert relative < 1e-9
    paths = [Path(__file__), HERE/'source_head.py', Path(fit_diagnostic.__file__)]
    result = dict(status='PASS', source_sha256={str(p.relative_to(ROOT)):watch.digest(p) for p in paths},
        baseline_parameters=baseline_parameters, parameters=parameters,
        additional_parameters=parameters-baseline_parameters,
        full_complex_input_samples=63872, source_feature_width=64,
        exact_parent_prediction_at_initialization=True, count_logits_unchanged_at_initialization=True,
        nonzero_head_all_six_permutations=True, correction_complex_sum_zero=True,
        raw_background_unchanged=True, chunked_checkpoint_output_and_gradient_equivalence=True,
        nonzero_readout_has_nonzero_attention_gradient=True,
        finite_full_model_gradients=True, first_readout_gradient_norm=readout_norm,
        first_upstream_branch_gradient_norm=upstream_first_gradient,
        sum_relative_error=relative, seconds=time.time()-started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        recorded_iq_reads=0, checkpoint_reads=0, optimizer_updates=0,
        device='cpu', rf_performance_evaluated=False,
        limitation='Synthetic preflight only. No longer temporal context or complex phase equivariance is added. Not a SepTDA reproduction.')
    watch.write(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_CPU_CHECK.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
