"""Native RFUAV geometry adapter for the full RF Challenge-style WaveNet.

30 layers, 128 residual channels, three complex source slots + background.
Only input geometry handling differs from the prepared historical adapter.
This is our multi-output adaptation, not reproduction of a paper's scores.
"""
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DENSE = Path(__file__).resolve().parents[1] / 'rfuav_dense_gated_20261008'
sys.path.insert(0, str(DENSE))
from models import WaveNetSeparator, build


class NativeWaveNet(WaveNetSeparator):
    def forward(self, mixture, context_features, crop_start):
        if mixture.ndim != 2 or not mixture.is_complex():
            raise ValueError('Expected complex [B,T]')
        if (context_features.ndim != 3
                or context_features.shape[:2] != (mixture.shape[0], 65)
                or context_features.shape[-1] < 2
                or crop_start.shape != (mixture.shape[0],)
                or not torch.isfinite(mixture).all()
                or not torch.isfinite(context_features).all()
                or not torch.isfinite(crop_start).all()
                or torch.any(crop_start < 0)
                or torch.any(crop_start + mixture.shape[-1] > context_features.shape[-1]*8192)):
            raise ValueError('Invalid native long context or crop')
        features = context_features.mean(-1, keepdim=True).expand_as(context_features)
        encoded = self.context_encoder(features)
        conditioning = self.context_projection(encoded.mean(-1, keepdim=True))
        scale = mixture.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        normalized = mixture / scale
        x = F.relu(self.input_projection(torch.stack((normalized.real, normalized.imag), 1))
                   + .1 * conditioning)
        skips = 0
        for start in range(0, 30, 5):
            fn = lambda value, start=start: self.segment(start, start + 5, value)
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                x, s = checkpoint(fn, x, use_reentrant=False)
            else:
                x, s = fn(x)
            skips = skips + s
        raw = self.output(F.relu(self.skip_projection(skips / math.sqrt(30)))).float()
        estimates = torch.complex(raw[:, 0::2], raw[:, 1::2]) * scale[:, None]
        estimates = estimates + (mixture - estimates.sum(1))[:, None] / 4
        return dict(estimates=estimates, count_logits=self.count_head(encoded.mean(-1)))


def build_native_wavenet():
    baseline = build('wavenet')
    model = NativeWaveNet()
    model.load_state_dict(baseline.state_dict(), strict=True)
    return model
