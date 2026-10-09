"""Prepared candidate: phase-preserving long I/Q context with a full WaveNet.

Packing 32 adjacent complex samples into 64 real channels is a permutation,
not resampling, averaging or magnitude conversion. The full 30x128 backbone
processes 65,280 tokens for 20.8896 ms at 100 MS/s. It decodes only the requested
63,872-sample crop. Two matched scopes differ only in whether out-of-crop I/Q
is visible. Both retain identical full-mixture RMS and mean power features.

This module is NOT connected to the currently running three-arm comparison.
No performance improvement or full-input GPU feasibility is assumed.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cycle15 import build_cycle15

PACK = 32
SAMPLES = 2_088_960
FINE = 63_872
PARAMETERS = 4_641_795


def pack_iq(waveform, packing=PACK):
    if waveform.ndim != 2 or not waveform.is_complex() or waveform.shape[-1] % packing:
        raise ValueError('Complex [B,L], length divisible by packing required')
    b, length = waveform.shape
    return torch.stack((waveform.real, waveform.imag), 1).reshape(
        b, 2, length//packing, packing).permute(0, 1, 3, 2).reshape(b, 2*packing, length//packing)


def unpack_iq(channels, sources=1, packing=PACK):
    if channels.ndim != 3 or channels.shape[1] != 2*sources*packing or channels.is_complex():
        raise ValueError('Real [B,2*sources*packing,tokens] required')
    b, _, tokens = channels.shape
    values = channels.reshape(b, sources, 2, packing, tokens)
    return torch.complex(values[:, :, 0], values[:, :, 1]).transpose(-1, -2).reshape(
        b, sources, tokens*packing)


def visible_iq(long_mixture, crop_start, fine_samples, scope):
    if scope == 'long':
        return long_mixture
    if scope != 'local':
        raise ValueError('Use local or long scope')
    position = torch.arange(long_mixture.shape[-1], device=long_mixture.device)[None]
    mask = (position >= crop_start[:, None]) & (position < crop_start[:, None] + fine_samples)
    return long_mixture * mask


def decode_crop(head, h, crop_start, fine_samples=FINE):
    needed = (fine_samples+2*PACK-2)//PACK
    positions = torch.arange(needed, device=h.device)[None] + (crop_start//PACK)[:, None]
    # At the exact right edge the extra token is discarded, so repeating its
    # feature does not enter any returned sample. Handle unaligned crops too.
    selected = h.gather(2, positions.clamp_max(h.shape[-1]-1)[:, None].expand(-1, h.shape[1], -1))
    local = unpack_iq(head(selected).float(), sources=4)
    offsets = (crop_start % PACK)[:, None] + torch.arange(fine_samples, device=h.device)[None]
    return local.gather(2, offsets[:, None].expand(-1, 4, -1))


class PhasePackedWaveNet(nn.Module):
    def __init__(self, scope):
        super().__init__()
        if scope not in ('local', 'long'):
            raise ValueError('Use local or long scope')
        self.scope = scope
        self.net = build_cycle15()
        self.net.input_projection = nn.Conv1d(2*PACK, 128, 1)
        self.net.output = nn.Conv1d(128, 8*PACK, 1)
        nn.init.kaiming_normal_(self.net.input_projection.weight)
        nn.init.normal_(self.net.output.weight, std=1e-3)
        nn.init.zeros_(self.net.output.bias)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('Backbone capacity changed')

    def forward(self, long_mixture, context_features, crop_start, fine_samples=FINE):
        if (long_mixture.ndim != 2 or not long_mixture.is_complex()
                or long_mixture.shape[-1] != SAMPLES
                or context_features.shape != (long_mixture.shape[0], 65, 255)
                or crop_start.shape != (long_mixture.shape[0],)
                or crop_start.dtype != torch.int64 or fine_samples != FINE
                or torch.any(crop_start < 0) or torch.any(crop_start+fine_samples > SAMPLES)
                or not torch.isfinite(long_mixture).all()
                or not torch.isfinite(context_features).all()):
            raise ValueError('Expected native full 20.8896ms complex context and valid crop')
        # Common normalization/auxiliary information in both scopes. Only the
        # additional complex samples differ, so no new target information enters.
        scale = long_mixture.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        visible = visible_iq(long_mixture, crop_start, fine_samples, self.scope)
        features = context_features.mean(-1, keepdim=True).expand_as(context_features)
        encoded = self.net.context_encoder(features)
        conditioning = self.net.context_projection(encoded.mean(-1, keepdim=True))
        x = F.relu(self.net.input_projection(pack_iq(visible/scale)) + .1*conditioning)
        skips = 0
        for start in range(0, 30, 5):
            fn = lambda value, start=start: self.net.segment(start, start+5, value)
            if self.net.checkpoint_blocks and self.training and torch.is_grad_enabled():
                x, delta = checkpoint(fn, x, use_reentrant=False)
            else:
                x, delta = fn(x)
            skips = skips + delta
        h = F.relu(self.net.skip_projection(skips/math.sqrt(30)))
        estimates = decode_crop(self.net.output,h,crop_start,fine_samples)*scale[:, None]
        fine_positions = crop_start[:, None] + torch.arange(fine_samples, device=h.device)[None]
        mixture = long_mixture.gather(1, fine_positions)
        estimates = estimates + (mixture-estimates.sum(1))[:, None]/4
        return dict(estimates=estimates, count_logits=self.net.count_head(encoded.mean(-1)))
