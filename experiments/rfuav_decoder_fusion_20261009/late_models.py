"""Full-capacity local waveform U-Net with late complex-context feature fusion.

Two arms differ only in the I/Q observed by the added context encoder. The
fine/coarse FiLM layers of the preceding trial are NOT part of this model.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PRIOR_FUSION_DIR = Path(__file__).resolve().parents[1] / 'rfuav_local_global_20261008'
sys.path.insert(0, str(PRIOR_FUSION_DIR))
from fusion_models import (ComplexContext, align_iq_context, CONTEXT_FIRST, CONTEXT_STEP,
    ARMS, PARENT, PARENT_SHA256, PARENT_DIR, LEGACY, WaveformUNet, PACK, LONG_SAMPLES, TARGET_SAMPLES)
from drone_rf.context_model import align_context
from waveform_models import pack_iq, unpack_iq


class DecoderFusionUNet(WaveformUNet):
    def __init__(self, use_long):
        super().__init__(long_context=False)
        self.use_long = use_long
        self.fusion_enabled = True
        self.iq_context = ComplexContext()
        self.decoder_fusion = nn.Sequential(
            nn.Conv1d(64 + 128, 64, 3, padding=1), nn.SiLU(), nn.Conv1d(64, 64, 1))
        # Small but nonzero: near-parent outputs and nonzero encoder gradients.
        nn.init.normal_(self.decoder_fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder_fusion[-1].bias)

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
            skips.append(x)
        aligned = align_context(encoded, long_start + (PACK - 1) / 2, LONG_SAMPLES // PACK,
                                fine_hop=PACK, pooling_stride=256)
        x = x + .1 * self.context_projection(aligned)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat((F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False), skip), 1))
        if tokens is not None:
            aligned_iq = align_iq_context(tokens, x.shape[-1], CONTEXT_FIRST, PACK)
            centers = CONTEXT_FIRST + torch.arange(x.shape[-1], device=x.device)[None] * PACK
            mask = ((centers >= offset) & (centers < offset + TARGET_SAMPLES))[:, None]
            # Correct local decoded features immediately before the I/Q head.
            x = x + self.decoder_fusion(torch.cat((x, aligned_iq), 1)) * mask
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
    net = DecoderFusionUNet(use_long=arm == 'local_global')
    saved = torch.load(PARENT, map_location='cpu', weights_only=False)
    if saved['best']['epoch'] != 5:
        raise ValueError('Parent selection changed')
    incompatible = net.load_state_dict(saved['model'], strict=False)
    added = ('iq_context.', 'decoder_fusion.')
    if incompatible.unexpected_keys or any(not k.startswith(added) for k in incompatible.missing_keys):
        raise ValueError('Parent load did not exactly preserve the full local network')
    return net


def predict(model, item):
    return model(item['mixture'], item['long_mixture'], item['context_features'], item['crop_start'], item['long_start'])


def stack_batch(items, device):
    keys = ('mixture', 'long_mixture', 'long_start', 'references', 'active', 'context_features', 'crop_start', 'construction_count')
    return {k: torch.as_tensor(np.stack([item[k] for item in items]), device=device) for k in keys}
