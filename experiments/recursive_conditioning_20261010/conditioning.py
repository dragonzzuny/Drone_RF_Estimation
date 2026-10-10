"""Original complex mixture conditioning for a successive second extraction.

Separate development candidate; never changes the registered successive trial.
The adapter is exactly zero initially. It adds 2304 parameters to the first
convolution of the full 32M separator, during the second pass only.
"""
import torch
from torch import nn


def features(original, remaining):
    """Bounded-scale original RI plus magnitude and a relative residual level.

Both arguments are complex STFTs [B,F,T] of observed/predicted waveforms.
q = sqrt(mean|X|^2 + mean|R|^2); original RI/q retain phase information.
RMS(R)/q records the relative level omitted by independent normalization.
No target, class, count, or recovered source is needed to compute this input.
The feature map is not itself equivariant to phase rotation of a learned net.
"""
    if original.shape != remaining.shape or original.ndim != 3:
        raise ValueError('Expected equal [B,F,T] spectra')
    if not original.is_complex() or not remaining.is_complex():
        raise ValueError('Complex I/Q spectra required')
    xp = original.abs().square().mean((1, 2), keepdim=True)
    rp = remaining.abs().square().mean((1, 2), keepdim=True)
    q = (xp + rp).clamp_min(1e-16).sqrt()
    z = original / q
    level = torch.where(rp > 0, rp.clamp_min(1e-16).sqrt() / q, 0.)
    return torch.stack((z.real, z.imag, torch.log1p(z.abs()),
                        level.expand_as(z.real)), 1)


def augment(net):
    if hasattr(net, 'original_conditioning'):
        raise ValueError('Already augmented')
    first = net.down[0][0]
    if (first.in_channels, first.out_channels, first.kernel_size) != (4, 64, (3, 3)):
        raise ValueError('Expected original full-width first convolution')
    # Restore only the CPU RNG; never reset the active GPU worker RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        adapter = nn.Conv2d(4, 64, 3, padding=1, bias=False)
        nn.init.zeros_(adapter.weight)
    net.original_conditioning = adapter.to(device=first.weight.device, dtype=first.weight.dtype)
    return net


def conditioned_spectrum(net, remaining, original, context, position):
    """A scoped hook is recreated during checkpoint backward recomputation."""
    extra = features(original, remaining)
    calls = []
    def inject(module, args, output):
        calls.append(True)
        return output + net.original_conditioning(extra).to(output.dtype)
    handle = net.down[0][0].register_forward_hook(inject)
    try:
        result = net(remaining, context, position)
        assert len(calls) == 1
        return result
    finally:
        handle.remove()


def predict(net, item, base_predict, analyze, synthesize, strongest_source,
            checkpoint_fn=None):
    """Whitelisted two-stage forward with no detach or reference input.

Zero adapter: numerically the same function as the registered successive
candidate. The first/count paths remain unconditioned. Three output sources,
last residual, zero background; this limitation is deliberately preserved.
"""
    original = {key: item[key] for key in ('mixture', 'context_features', 'crop_start')}
    def first_call(x, context, position):
        return base_predict(net, dict(mixture=x, context_features=context, crop_start=position))
    def second_call(r, x, context, position):
        result = conditioned_spectrum(net, analyze(r), analyze(x), context, position)
        return synthesize(result['estimates'], r.shape[-1]), result['count_logits']
    invoke = (lambda f, *args: f(*args)) if checkpoint_fn is None else checkpoint_fn
    first, logits = invoke(first_call, original['mixture'], original['context_features'], original['crop_start'])
    s1, slot1 = strongest_source(first)
    remaining = original['mixture'] - s1
    second, _ = invoke(second_call, remaining, original['mixture'], original['context_features'], original['crop_start'])
    s2, slot2 = strongest_source(second)
    s3 = remaining - s2
    output = torch.stack((s1, s2, s3, torch.zeros_like(s3)), 1)
    return output, logits, dict(remaining=remaining, first_predictions=first,
                               first_slot=slot1, second_slot=slot2)
