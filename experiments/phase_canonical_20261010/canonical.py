"""Mixture-only global phase canonicalization around an unchanged separator.

For nonzero z choose the first maximum-magnitude STFT coefficient a(z).
u(z)=a/|a| and G(z)=u F(z/u). In exact arithmetic a common phase rotation
does not change that index; G(exp(j phi)z)=exp(j phi)G(z), slot by slot.
Near magnitude ties the chosen anchor can be numerically discontinuous.
This adds no parameters, no source labels and no ground-truth count.
"""
import torch
from torch import nn


def phase_anchor(z):
    if z.ndim != 3 or not z.is_complex():
        raise ValueError('Expected complex [B,F,T]')
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite STFT')
    flat = z.flatten(1)
    magnitudes = flat.abs()
    index = magnitudes.argmax(-1, keepdim=True)
    anchor = flat.gather(1, index)
    amplitude = anchor.abs()
    nonzero = amplitude > 0
    phase = anchor / torch.where(nonzero, amplitude, torch.ones_like(amplitude))
    phase = torch.where(nonzero, phase, torch.ones_like(phase))
    return phase[:, :, None], nonzero[:, :, None], index


class CanonicalPhaseSeparator(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, z, context_features, crop_start):
        phase, nonzero, _ = phase_anchor(z)
        result = self.base(z / phase, context_features, crop_start)
        result = dict(result)
        output = result['estimates'] * phase[:, None]
        result['estimates'] = torch.where(nonzero[:, None], output, torch.zeros_like(output))
        return result
