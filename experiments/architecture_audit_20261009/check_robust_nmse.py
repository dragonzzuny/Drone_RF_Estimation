"""Synthetic complex-I/Q invariants for the prospective loss comparison."""
import argparse
import gc
import itertools
from pathlib import Path
import torch
import robust_nmse as loss
import watch_epochs as watch
from models import build, predict
from torch.nn import functional as F


def full_input_check():
    torch.manual_seed(19)
    references = torch.randn(1, 3, 63872, dtype=torch.complex64)
    references[:, 1] *= .316227766
    references[:, 2] = 0
    mixture = references.sum(1)
    batch = dict(mixture=mixture, references=references, active=torch.tensor([[True, True, False]]),
                 context_features=torch.randn(1, 65, 255), crop_start=torch.tensor([4097]),
                 construction_count=torch.tensor([2]))
    results = []
    for mode in loss.MODES:
        torch.manual_seed(0)
        net = build('unet_mean').train()
        parameters = sum(p.numel() for p in net.parameters())
        assert parameters == 32_142_859
        estimates, logits = predict(net, batch)
        objective = loss.objective(estimates, references, batch['active'], mixture, mode)['loss']
        objective = objective + .1*F.cross_entropy(logits, batch['construction_count']-1)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        assert float(norm) > 0 and net.output.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.down.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        relative = float((estimates.sum(1)-mixture).abs().square().sum()/mixture.abs().square().sum())
        assert relative < 1e-9
        results.append(dict(mode=mode, parameters=parameters, full_complex_samples=63872,
            finite_full_network_gradients=True, head_and_encoder_gradients=True,
            preclip_norm=float(norm), sum_relative_error=relative, optimizer_updates=0))
        del net, estimates, logits, objective, norm
        gc.collect()
    return results


def run():
    torch.set_num_threads(2)
    torch.manual_seed(827)
    references = torch.randn(3, 3, 257, dtype=torch.complex128)
    active = torch.arange(3)[None] < torch.tensor([1, 2, 3])[:, None]
    references *= active[..., None]
    # Include a finite very weak source; never replace it by an inactive label.
    references[1, 1] *= 1e-3
    mixture = references.sum(1)
    estimates = torch.randn(3, 4, 257, dtype=torch.complex128, requires_grad=True)
    outcomes = {}
    for mode in loss.MODES:
        value = loss.objective(estimates, references, active, mixture, mode)
        gradient, = torch.autograd.grad(value['loss'], estimates)
        assert torch.isfinite(value['loss']) and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        for permutation in itertools.permutations(range(3)):
            transformed = torch.cat((estimates[:, list(permutation)], estimates[:, -1:]), 1)
            alt = loss.objective(transformed, references, active, mixture, mode)
            torch.testing.assert_close(value['loss'], alt['loss'], rtol=1e-10, atol=1e-10)
        # Overall complex gain is a units transformation, not oracle output scaling.
        scale = 2.3 * torch.exp(torch.tensor(.37j, dtype=torch.complex128))
        alt = loss.objective(estimates*scale, references*scale, active, mixture*scale, mode)
        torch.testing.assert_close(value['loss'], alt['loss'], rtol=1e-10, atol=1e-10)
        exact = torch.cat((references, torch.zeros_like(mixture[:, None])), 1).requires_grad_(True)
        perfect = loss.objective(exact, references, active, mixture, mode)
        perfect_gradient, = torch.autograd.grad(perfect['loss'], exact)
        assert torch.isfinite(perfect_gradient).all()
        torch.testing.assert_close(perfect['loss'], torch.zeros_like(perfect['loss']), atol=1e-10, rtol=0)
        outcomes[mode] = dict(finite_loss_and_gradient=True, all_six_output_permutations=True,
                              common_complex_gain_invariance=True, perfect_reconstruction_finite=True)
    original = loss.objective(estimates, references, active, mixture, 'original')
    robust = loss.objective(estimates, references, active, mixture, 'log1p_nmse')
    assert torch.equal(original['assignment'], robust['assignment'])
    # Independent reference-aligned formula, using the exact existing denominator.
    aligned = estimates[:, :3].gather(1, original['assignment'][..., None].expand(-1, -1, 257))
    pm = mixture.abs().square().mean(-1).clamp_min(1e-8)
    pr = references.abs().square().mean(-1)
    divisor = torch.where(active, torch.maximum(pr, pm[:, None]*1e-6), pm[:, None])
    error = (aligned-references).abs().square().mean(-1)/divisor
    expected_difference = (torch.log1p(error)-error).mean()
    torch.testing.assert_close(robust['loss']-original['loss'], expected_difference, rtol=1e-10, atol=1e-9)
    # d log(1+u)/du = 1/(1+u): changing gradient weights is the intervention.
    u = torch.tensor([0., .1, 1., 100.], dtype=torch.float64, requires_grad=True)
    derivative, = torch.autograd.grad(torch.log1p(u).sum(), u)
    torch.testing.assert_close(derivative, 1/(1+u))
    full = full_input_check()
    paths = [Path(__file__), Path(loss.__file__), Path(loss.diagnostic.__file__)]
    return dict(status='PASS', source_sha256={str(p.relative_to(watch.ROOT)):watch.digest(p) for p in paths},
                modes=outcomes, original_assignment_preserved=True, independent_delta_formula=True,
                scalar_derivative_verified=True, full_input_checks=full,
                recorded_iq_reads=0, trained=False, heldout_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = run(); watch.write(args.output, result)
    print(result)
