"""Feature algebra, zero initialization, gradient and hook lifetime checks."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from conditioning import augment, features, conditioned_spectrum

ROOT = Path(__file__).resolve().parents[2]


def run():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    x = torch.randn(2, 24, 31, dtype=torch.complex128)
    r = torch.randn_like(x)
    f = features(x, r)
    assert f.shape == (2, 4, 24, 31) and torch.isfinite(f).all()
    assert torch.count_nonzero(features(torch.zeros_like(x), torch.zeros_like(r))) == 0
    assert torch.allclose(f, features(3 * x, 3 * r), atol=1e-12, rtol=1e-12)
    phase = torch.tensor(.71, dtype=torch.float64)
    rot = torch.exp(1j * phase)
    fr = features(x * rot, r * rot)
    assert torch.allclose(fr[:, 2:], f[:, 2:], atol=1e-12, rtol=1e-12)
    assert torch.allclose(torch.complex(fr[:, 0], fr[:, 1]),
                          torch.complex(f[:, 0], f[:, 1]) * rot, atol=1e-12, rtol=1e-12)
    # Degenerate spectra must have finite input derivatives as well.
    for a, b in ((x, r), (x, torch.zeros_like(r)), (torch.zeros_like(x), torch.zeros_like(r))):
        a, b = a.clone().requires_grad_(), b.clone().requires_grad_()
        ga, gb = torch.autograd.grad(features(a, b).square().sum(), (a, b))
        assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    class HookHarness(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 64, 3, padding=1))])
        def forward(self, z, context, position):
            return self.down[0](features(z, z).float())
    net = HookHarness()
    rng = torch.get_rng_state().clone()
    augment(net)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in net.original_conditioning.parameters()) == 2304
    z = x.to(torch.complex64); rest = r.to(torch.complex64)
    expected = net(rest, None, None)
    actual = conditioned_spectrum(net, rest, z, None, None)
    assert torch.equal(expected, actual) and len(net.down[0][0]._forward_hooks) == 0
    actual.square().mean().backward()
    assert torch.isfinite(net.original_conditioning.weight.grad).all()
    assert float(net.original_conditioning.weight.grad.norm()) > 0
    # Cleanup must occur even if downstream code raises.
    h = net.down[0].register_forward_hook(lambda *args: (_ for _ in ()).throw(RuntimeError('intentional')))
    try:
        try: conditioned_spectrum(net, rest, z, None, None)
        except RuntimeError as e: assert str(e) == 'intentional'
        else: raise AssertionError('Exception not observed')
        assert len(net.down[0][0]._forward_hooks) == 0
    finally: h.remove()
    result = dict(status='PASS',feature_scale_invariance=True,feature_phase_covariance=True,
        model_phase_equivariance_claimed=False,degenerate_derivatives_finite=True,
        zero_input_features_zero=True,scale_invariance_above_numerical_floor_only=True,
        zero_adapter_forward_exact=True,adapter_gradient_nonzero=True,hook_cleanup=True,
        cpu_rng_preserved=True,extra_parameters=2304,optimizer_steps=0,
        full_model_test=False,heldout_read=False,sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name('conditioning.py'))})
    out = ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_ALGEBRA_CHECK.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': run()
