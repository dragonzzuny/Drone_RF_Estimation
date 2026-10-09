"""Prepared full U-Net control vs complex-mask output parameterization.

Both retain all 32,142,859 parameters and identical modules/initial tensors.
The head is zeroed in BOTH arms to start from the same mixture/4 outputs.
Mapping predicts normalized complex STFT values. Masking predicts a complex
factor multiplying each observed mixture STFT bin. Both enforce output sum.
This is standard complex masking as an inductive bias, not a novel architecture
or a guarantee of phase equivariance, invertibility, or perfect separation.
"""
import torch
from torch import nn
from torch.nn import functional as F

import fit_diagnostic  # Pinned native/vendor imports.
from drone_rf.context_model import ContextualSeparator

PARAMETERS = 32_142_859
MODES = ('mapping', 'masking')


class OutputUNet(ContextualSeparator):
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('Use mapping or masking')
        super().__init__('tcn', 'mean')
        self.mode = mode
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('Full backbone capacity changed')

    def _forward(self, z, bottleneck_context=None):
        # Same full encoder/decoder and normalization as the frozen parent.
        if not z.is_complex() or z.ndim != 3 or min(z.shape[-2:]) < 16:
            raise ValueError('Expected complex [B,F,T], F,T>=16')
        scale = z.abs().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        norm = z / scale
        x = torch.stack([norm.real, norm.imag, torch.full_like(norm.real, .5),
                         torch.log1p(norm.abs())], 1)
        skips = []
        for i, layer in enumerate(self.down):
            if i:
                x = F.avg_pool2d(x, 2)
            x = layer(x)
            skips.append(x)
        if bottleneck_context is not None:
            if bottleneck_context.shape != (x.shape[0], x.shape[1], x.shape[3]):
                raise ValueError('Context misalignment')
            x = x + bottleneck_context[:, :, None, :]
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[-2:],
                      mode='bilinear', align_corners=False), skip], 1))
        raw = self.output(x).float()
        factors = torch.complex(raw[:, 0::2], raw[:, 1::2])
        estimates = factors * (scale[:, None] if self.mode == 'mapping' else z[:, None])
        return estimates + (z - estimates.sum(1))[:, None] / (self.max_sources + 1)


def build(mode):
    torch.manual_seed(0)
    return OutputUNet(mode)
