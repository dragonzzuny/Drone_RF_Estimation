"""Full local I/Q U-Net plus an independently encoded complex context branch.

Both arms inherit the SAME selected short-context e5 checkpoint. The only arm
difference is masking outside-target samples in the new context branch.
No new inference labels or long-reference supervision are introduced.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PARENT_DIR = Path(__file__).resolve().parents[1] / 'rfuav_waveform_20261008'
sys.path.insert(0, str(PARENT_DIR))
from waveform_models import WaveformUNet, pack_iq, unpack_iq, LEGACY, PACK, LONG_SAMPLES, TARGET_SAMPLES
from drone_rf.context_model import TemporalResidual, align_context

ARMS = ('local_only', 'local_global')
PARENT = Path(__file__).resolve().parents[2] / 'local/waveform_context_20261008_v1/short_context/BEST.pt'
PARENT_SHA256 = '21c957015455c652576df4a1ac82a1be9c141e09eae088940de6330ef955f25f'
CONTEXT_STEP = PACK * 4**4
CONTEXT_FIRST = (PACK - 1) / 2


def align_iq_context(tokens, frames, first_center, stride):
    """Align learned context token centers to local feature sample coordinates.

    Four odd symmetric stride-4 convolutions keep the first center at 7.5
    native samples; their step is 4096 samples. No approximate resize is used.
    """
    positions = first_center + torch.arange(frames, device=tokens.device, dtype=torch.float32) * stride
    index = (positions - CONTEXT_FIRST) / CONTEXT_STEP
    horizontal = 2 * (index + .5) / tokens.shape[-1] - 1
    grid = torch.stack((horizontal, torch.zeros_like(horizontal)), -1)[None, None]
    return F.grid_sample(tokens[:, :, None], grid.expand(tokens.shape[0], -1, -1, -1),
                         padding_mode='border', align_corners=False)[:, :, 0]


class ComplexContext(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (64, 128, 128, 128)
        layers = []
        for i, width in enumerate(widths):
            kernel = 17 if i == 0 else 9
            layers.extend((nn.Conv1d(32 if i == 0 else widths[i - 1], width, kernel,
                                    stride=4, padding=kernel // 2), nn.GroupNorm(8, width), nn.SiLU()))
        self.front = nn.Sequential(*layers)
        # Ordered 256 tokens; two convs at each dilation span the full context.
        self.body = nn.Sequential(*(TemporalResidual(128, d) for d in (1, 2, 4, 8, 16, 32, 64)))
        self.norm = nn.LayerNorm(128)

    def forward(self, observed):
        x = self.body(self.front(pack_iq(observed)))
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class LocalGlobalUNet(WaveformUNet):
    def __init__(self, use_long):
        super().__init__(long_context=False)
        self.use_long = use_long
        self.fusion_enabled = True  # diagnostic bypass; always True in training
        self.iq_context = ComplexContext()
        self.fine_film = nn.Conv1d(128, 128, 1)
        self.coarse_film = nn.Conv1d(128, 2048, 1)
        # Small NONZERO projections permit encoder gradients from the first step.
        for layer in (self.fine_film, self.coarse_film):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    def condition(self, x, tokens, offset, coarse=False):
        if tokens is None:
            return x
        stride = CONTEXT_STEP if coarse else PACK
        first = (CONTEXT_STEP - 1) / 2 if coarse else CONTEXT_FIRST
        layer = self.coarse_film if coarse else self.fine_film
        gamma, beta = align_iq_context(layer(tokens), x.shape[-1], first, stride).chunk(2, 1)
        centers = first + torch.arange(x.shape[-1], device=x.device)[None] * stride
        # Condition scored-area features only; local raw-I/Q support stays fixed.
        mask = ((centers >= offset) & (centers < offset + TARGET_SAMPLES))[:, None]
        return x * (1 + .1 * torch.tanh(gamma) * mask) + .1 * beta * mask

    def forward(self, mixture, long_mixture, context_features, crop_start, long_start):
        if (mixture.ndim != 2 or not mixture.is_complex() or mixture.shape[-1] != TARGET_SAMPLES
                or long_mixture.shape != (mixture.shape[0], LONG_SAMPLES) or not long_mixture.is_complex()):
            raise ValueError('Complex short/long mixture geometry changed')
        if (context_features.shape != (mixture.shape[0], 65, 256)
                or crop_start.shape != (mixture.shape[0],) or torch.any(crop_start < 0)
                or long_start.shape != crop_start.shape or torch.any(long_start < 0)
                or torch.any(long_start + LONG_SAMPLES > 2097152)
                or torch.any(crop_start < long_start)
                or torch.any(crop_start + TARGET_SAMPLES > long_start + LONG_SAMPLES)
                or not torch.isfinite(context_features).all() or not torch.isfinite(long_mixture).all()):
            raise ValueError('Invalid mixture-only input geometry or values')
        scale = mixture.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        positions = torch.arange(LONG_SAMPLES, device=mixture.device)[None]
        offset = (crop_start - long_start)[:, None]
        observed_local = long_mixture * ((positions >= offset) & (positions < offset + TARGET_SAMPLES))
        observed_context = long_mixture if self.use_long else observed_local
        tokens = self.iq_context(observed_context / scale) if self.fusion_enabled else None
        x = pack_iq(observed_local / scale)
        features = context_features.mean(-1, keepdim=True).expand_as(context_features)
        encoded = self.context_encoder(features)
        skips = []
        for i, layer in enumerate(self.down):
            if i:
                x = F.avg_pool1d(x, 4)
            x = layer(x)
            if i == 0:
                x = self.condition(x, tokens, offset)
            skips.append(x)
        aligned = align_context(encoded, long_start + (PACK - 1) / 2, LONG_SAMPLES // PACK,
                                fine_hop=PACK, pooling_stride=256)
        x = x + .1 * self.context_projection(aligned)
        x = self.condition(x, tokens, offset, coarse=True)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat((F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False), skip), 1))
        raw = self.output(x).float()
        full = unpack_iq(raw.reshape(raw.shape[0] * 4, 2 * PACK, raw.shape[-1])).reshape(raw.shape[0], 4, LONG_SAMPLES) * scale[:, None]
        indices = (offset + torch.arange(TARGET_SAMPLES, device=mixture.device)[None]).long()
        estimates = full.gather(2, indices[:, None].expand(-1, 4, -1))
        estimates = estimates + (mixture - estimates.sum(1))[:, None] / 4
        return estimates, self.count_head(encoded.mean(-1))


def build(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(0)
    net = LocalGlobalUNet(use_long=arm == 'local_global')
    saved = torch.load(PARENT, map_location='cpu', weights_only=False)
    if saved['best']['epoch'] != 5:
        raise ValueError('Parent selection changed')
    incompatible = net.load_state_dict(saved['model'], strict=False)
    added = ('iq_context.', 'fine_film.', 'coarse_film.')
    if incompatible.unexpected_keys or any(not k.startswith(added) for k in incompatible.missing_keys):
        raise ValueError('Parent load did not exactly preserve the full local network')
    return net


def predict(model, item):
    return model(item['mixture'], item['long_mixture'], item['context_features'], item['crop_start'], item['long_start'])


def stack_batch(items, device):
    keys = ('mixture', 'long_mixture', 'long_start', 'references', 'active', 'context_features', 'crop_start', 'construction_count')
    return {k: torch.as_tensor(np.stack([item[k] for item in items]), device=device) for k in keys}
