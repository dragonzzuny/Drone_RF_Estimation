"""Full-capacity architecture candidates; no target or source count in inference.

The gated convolutions follow the RF Challenge WaveNet architecture described
in https://arxiv.org/html/2409.08839v3 (30 layers, 128 residual channels).
This is a multi-output RFUAV adaptation, NOT a reproduction of their results.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from drone_rf.context_model import ContextualSeparator, TemporalEncoder
from drone_rf.waveform import analyze, synthesize


ARMS = ('unet_mean', 'unet_gated', 'wavenet')


class GatedResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.filter_gate = nn.Conv1d(width, 2 * width, 3,
                                     padding=dilation, dilation=dilation)
        self.project = nn.Conv1d(width, 2 * width, 1)
        for layer in (self.filter_gate, self.project):
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        gate, value = self.filter_gate(x).chunk(2, dim=1)
        residual, skip = self.project(gate.sigmoid() * value.tanh()).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2), skip


class BottleneckGates(nn.Module):
    """Process time at EACH frequency cell; do not average away complex features.

    The U-Net remains 64/128/256/512/1024 wide. A 256-channel adapter changes
    processing inside its bottleneck, not the backbone width. Zero final weights
    make the augmented model exactly the baseline at initialization.
    """
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv1d(1024, 256, 1)
        self.blocks = nn.ModuleList(GatedResidual(256, d) for d in (1, 2, 4, 8))
        self.expand = nn.Conv1d(256, 1024, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        b, c, f, t = x.shape
        h = self.reduce(x.permute(0, 2, 1, 3).reshape(b * f, c, t))
        skips = 0
        for block in self.blocks:
            h, s = block(h)
            skips = skips + s
        delta = self.expand(skips / math.sqrt(len(self.blocks)))
        return x + delta.reshape(b, f, c, t).permute(0, 2, 1, 3)


class GatedUNet(ContextualSeparator):
    def __init__(self):
        super().__init__(context_kind='tcn', context_mode='mean')
        # Preserve original parameter names for a verifiable baseline copy.
        self.gates = BottleneckGates()

    def _forward(self, z, bottleneck_context=None):
        # This is the pinned project's forward with ONE bottleneck insertion.
        if not z.is_complex() or z.ndim != 3 or min(z.shape[-2:]) < 16:
            raise ValueError('Expected complex [B,F,T], F,T >= 16')
        scale = z.abs().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        norm = z / scale
        x = torch.stack((norm.real, norm.imag, torch.full_like(norm.real, .5),
                         torch.log1p(norm.abs())), dim=1)
        skips = []
        for i, layer in enumerate(self.down):
            if i:
                x = F.avg_pool2d(x, 2)
            x = layer(x)
            skips.append(x)
        if bottleneck_context is not None:
            if bottleneck_context.shape != (x.shape[0], x.shape[1], x.shape[3]):
                raise ValueError('Context shape mismatch')
            x = x + bottleneck_context[:, :, None, :]
        x = self.gates(x)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat((F.interpolate(x, size=skip.shape[-2:],
                      mode='bilinear', align_corners=False), skip), dim=1))
        raw = self.output(x).float()
        residual = torch.complex(raw[:, 0::2], raw[:, 1::2]) * scale[:, None]
        return residual + (z - residual.sum(1))[:, None] / 4


class WaveNetSeparator(nn.Module):
    """Noncausal real/imaginary WaveNet, three unordered sources + background.

    The waveform path's convolutional receptive field is 6139 samples. Global
    mixture RMS and the mean long-context feature are additional conditioning.
    This is NOT 20.97 ms of phase-preserving waveform context.
    """
    def __init__(self, checkpoint_blocks=True):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.context_encoder = TemporalEncoder('tcn')
        self.count_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))
        self.context_projection = nn.Conv1d(128, 128, 1, bias=False)
        self.input_projection = nn.Conv1d(2, 128, 1)
        self.blocks = nn.ModuleList(GatedResidual(128, 2 ** (i % 10)) for i in range(30))
        self.skip_projection = nn.Conv1d(128, 128, 1)
        self.output = nn.Conv1d(128, 8, 1)
        for layer in (self.input_projection, self.skip_projection):
            nn.init.kaiming_normal_(layer.weight)
        # Independent small slot weights, not four identical zero estimates.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def segment(self, start, stop, x):
        skips = 0
        for i in range(start, stop):
            x, s = self.blocks[i](x)
            skips = skips + s
        return x, skips

    def forward(self, mixture, context_features, crop_start):
        if mixture.ndim != 2 or not mixture.is_complex():
            raise ValueError('Expected complex [B,T]')
        if (context_features.shape != (mixture.shape[0], 65, 256)
                or crop_start.shape != (mixture.shape[0],)
                or torch.any(crop_start < 0)
                or torch.any(crop_start + mixture.shape[-1] > 2097152)):
            raise ValueError('Invalid long context or crop')
        features = context_features.mean(-1, keepdim=True).expand_as(context_features)
        encoded = self.context_encoder(features)
        conditioning = self.context_projection(encoded.mean(-1, keepdim=True))
        scale = mixture.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        normalized = mixture / scale
        x = F.relu(self.input_projection(torch.stack((normalized.real, normalized.imag), 1))
                   + .1 * conditioning)
        skips = 0
        for start in range(0, 30, 5):
            # Bind start in the closure; backward must recompute the SAME segment.
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


def build(arm):
    """Fresh seed-0 models. U-Net shared parameters are identical; no warm start."""
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(0)
    if arm == 'unet_mean':
        return ContextualSeparator('tcn', 'mean')
    if arm == 'unet_gated':
        return GatedUNet()
    net = WaveNetSeparator()
    # Same initial count/context subnetwork as the U-Net arms. No held-out model.
    torch.manual_seed(0)
    baseline = ContextualSeparator('tcn', 'mean')
    net.context_encoder.load_state_dict(baseline.context_encoder.state_dict())
    net.count_head.load_state_dict(baseline.count_head.state_dict())
    del baseline
    return net


def predict(net, batch):
    # Deliberate whitelist. References, aircraft names and true counts never enter.
    mixture = batch['mixture']
    args = (batch['context_features'], batch['crop_start'])
    if isinstance(net, WaveNetSeparator):
        result = net(mixture, *args)
        return result['estimates'], result['count_logits']
    result = net(analyze(mixture), *args)
    return synthesize(result['estimates'], mixture.shape[-1]), result['count_logits']
