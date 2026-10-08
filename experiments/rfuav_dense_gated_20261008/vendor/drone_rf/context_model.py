"""Full complex U-Net with aligned mixture-only temporal context.

TCN, Transformer and bidirectional LSTM are candidates, not trained results.
All use the same 128-wide interface and retain the complete fine U-Net.
The long-context count head classifies 1/2/3; zero-source rejection is not yet
covered. Its prediction never hard-masks or orders waveform outputs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import ComplexSeparator


class TemporalResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 3, padding=dilation,
                                             dilation=dilation) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))

    def forward(self, x):
        residual = x
        for layer, norm in zip(self.layers, self.norms):
            x = F.silu(norm(layer(x).transpose(1, 2))).transpose(1, 2)
        return x + residual


class TemporalEncoder(nn.Module):
    def __init__(self, kind='tcn', input_features=65, width=128):
        super().__init__()
        if kind not in ('tcn', 'transformer', 'lstm') or width % 4:
            raise ValueError('Use tcn/transformer/lstm and a width divisible by four')
        self.kind, self.width = kind, width
        self.input = nn.Conv1d(input_features, width, 1)
        if kind == 'tcn':
            self.body = nn.Sequential(*(TemporalResidual(width, d) for d in (1, 2, 4, 8, 16, 32)))
        elif kind == 'transformer':
            # Construct layers independently rather than sharing identical
            # initialization via cloned TransformerEncoder layers.
            self.body = nn.ModuleList(nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=4 * width,
                dropout=.1, batch_first=True, norm_first=True) for _ in range(2))
        else:
            self.body = nn.LSTM(width, width // 2, num_layers=2, dropout=.1,
                                batch_first=True, bidirectional=True)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, feature):
        x = self.input(feature)
        if self.kind == 'tcn':
            x = self.body(x).transpose(1, 2)
        else:
            x = x.transpose(1, 2)
            if self.kind == 'transformer':
                position = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None]
                rates = torch.exp(torch.arange(0, self.width, 2, device=x.device,
                                               dtype=torch.float32) * (-math.log(10000.) / self.width))
                pe = torch.zeros(x.shape[1], self.width, device=x.device)
                pe[:, 0::2], pe[:, 1::2] = torch.sin(position * rates), torch.cos(position * rates)
                x = x + pe.to(x.dtype)[None]
                for layer in self.body:
                    x = layer(x)
            else:
                x, _ = self.body(x)
        return self.output_norm(x).transpose(1, 2)


def align_context(encoded, crop_start, fine_frames, fine_hop=128,
                  first_center=4095.5, context_step=8192, pooling_stride=16):
    """Align to centers of four 2x U-Net time pools, not an arbitrary resize.

    Fine STFT must use center=True; frame t then centers at crop_start+t*hop.
    Each bottleneck frame centers at (16*t+7.5)*hop. Context edge coordinates
    use border replication. This is an offline, noncausal context model.
    """
    if encoded.ndim != 3 or fine_frames < pooling_stride or min(fine_hop, context_step, pooling_stride) < 1:
        raise ValueError('Invalid alignment geometry')
    batch, _, tokens = encoded.shape
    if crop_start.shape != (batch,) or not torch.isfinite(crop_start).all() or torch.any(crop_start < 0):
        raise ValueError('Nonnegative [B] crop positions required')
    center = torch.arange(fine_frames // pooling_stride, device=encoded.device,
                          dtype=torch.float32) * pooling_stride + (pooling_stride - 1) / 2.
    samples = crop_start.to(device=encoded.device, dtype=torch.float32)[:, None] + center[None] * fine_hop
    index = (samples - first_center) / context_step
    horizontal = 2 * (index + .5) / tokens - 1
    grid = torch.stack([horizontal, torch.zeros_like(horizontal)], dim=-1)[:, None]
    # grid_sample on CPU does not support every autocast dtype.
    return F.grid_sample(encoded.float()[:, :, None], grid,
                         padding_mode='border', align_corners=False)[:, :, 0].to(encoded.dtype)


class ContextualSeparator(ComplexSeparator):
    """Three unordered waveform slots + background and a separate count head.

    ``mean`` is a matched-capacity ablation that removes temporal variation and
    crop-specific localization. It does not isolate periodicity alone. Train it
    independently; a test-time ablation alone also causes distribution shift.
    """
    def __init__(self, context_kind='tcn', context_mode='ordered', input_features=65,
                 fine_hop=128, context_step=8192, first_center=4095.5):
        super().__init__(max_sources=3)
        if context_mode not in ('ordered', 'mean'):
            raise ValueError('Context mode must be ordered or mean')
        self.context_mode = context_mode
        self.fine_hop, self.context_step, self.first_center = fine_hop, context_step, first_center
        self.context_encoder = TemporalEncoder(context_kind, input_features)
        self.context_projection = nn.Conv1d(128, 1024, 1, bias=False)
        self.count_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, z, context_features, crop_start):
        if z.ndim != 3 or not z.is_complex():
            raise ValueError('Expected complex fine STFT [B,F,T]')
        if context_features.ndim != 3 or context_features.shape[0] != z.shape[0] or context_features.shape[-1] < 2:
            raise ValueError('Expected mixture-only long context [B,features,tokens]')
        if not torch.isfinite(context_features).all():
            raise ValueError('Nonfinite context features')
        # When length is hop-divisible, center=True includes a final frame
        # centered exactly on the right sample boundary (reflect padded).
        end = crop_start + (z.shape[-1] - 1) * self.fine_hop
        if torch.any(end > context_features.shape[-1] * self.context_step):
            raise ValueError('Fine window lies outside its long context')
        if self.context_mode == 'mean':
            context_features = context_features.mean(-1, keepdim=True).expand_as(context_features)
        encoded = self.context_encoder(context_features)
        aligned = align_context(encoded, crop_start, z.shape[-1], self.fine_hop,
                                self.first_center, self.context_step)
        estimates = self._forward(z, .1 * self.context_projection(aligned))
        return dict(estimates=estimates, count_logits=self.count_head(encoded.mean(-1)))
