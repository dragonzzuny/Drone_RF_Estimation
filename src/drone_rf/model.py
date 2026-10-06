"""Full-width complex U-Net with independently supervised unordered outputs.

Adapted from this project's no-NMF model99/model101 (2026-10-06).
This is a new, untrained candidate, not a published-model reproduction.
The extra background output makes K=0 compatible with mixture consistency.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(8, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(8, outgoing), nn.SiLU())


class ComplexSeparator(nn.Module):
    """Keep the original 64/128/256/512/1024 backbone; output K slots + background.

    Input: complex STFT [batch, frequency, time]. Output has K+1 complex streams.
    No source identities, reference waveforms, or source count enter forward().
    The last stream is an explicitly supervised background/residual stream.
    """
    def __init__(self, max_sources=2):
        super().__init__()
        if max_sources not in (2, 4):
            raise ValueError('This experiment supports maximum 2 or 4 sources')
        self.max_sources = max_sources
        widths = (64, 128, 256, 512, 1024)
        self.down = nn.ModuleList(Block(4 if i==0 else widths[i-1], c)
                                  for i,c in enumerate(widths))
        self.up = nn.ModuleList(Block(widths[i+1]+widths[i], widths[i])
                                for i in (3,2,1,0))
        self.output = nn.Conv2d(64, 2*(max_sources+1), 1)
        # Break slot symmetry without tying outputs to aircraft identities.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, z):
        if not z.is_complex() or z.ndim!=3 or min(z.shape[-2:])<16:
            raise ValueError('Expected complex [B,F,T] with F,T >= 16')
        scale = z.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
        norm = z/scale
        x = torch.stack([norm.real, norm.imag, torch.full_like(norm.real,.5),
                         torch.log1p(norm.abs())],1)
        skips=[]
        for i, layer in enumerate(self.down):
            if i: x=F.avg_pool2d(x,2)
            x=layer(x); skips.append(x)
        for layer,skip in zip(self.up,reversed(skips[:-1])):
            x=layer(torch.cat([F.interpolate(x,size=skip.shape[-2:],
                     mode='bilinear',align_corners=False),skip],1))
        raw=self.output(x).float()
        residual=torch.complex(raw[:,0::2],raw[:,1::2])*scale[:,None]
        # This projection enforces only the sum, not the accuracy of any source.
        return residual+(z-residual.sum(1))[:,None]/(self.max_sources+1)
