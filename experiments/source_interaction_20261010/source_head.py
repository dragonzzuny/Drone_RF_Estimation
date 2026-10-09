"""Source-interaction output head for the existing full-width RF U-Net.

An adaptation hypothesis inspired by SepTDA's source-axis processing, not a
SepTDA reproduction. It adds neither longer temporal input nor an oracle count.
All original backbone and count-head parameters are retained. A zero final
readout preserves the parent prediction exactly at initialization.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SourceInteractionHead(nn.Module):
    """Shared nonlinear features and attention over THREE predicted sources.

    Input features [B,64,F,T] and the original head's real/imaginary pairs
    [B,8,F,T]. No identity embeddings: permuting the first three original pairs
    permutes the correction. Background is not a fourth exchangeable source.
    The three complex corrections sum to zero before mixture projection.
    """

    def __init__(self, base_head, width=64, chunk_points=4096,
                 checkpoint_chunks=True):
        super().__init__()
        if base_head.in_channels != 64 or base_head.out_channels != 8:
            raise ValueError('Expected the full parent 64-to-8 output head')
        if width < 1 or chunk_points < 1:
            raise ValueError('Positive feature width and chunk length required')
        self.base = base_head
        self.width = width
        self.chunk_points = chunk_points
        self.checkpoint_chunks = checkpoint_chunks
        self.input = nn.Linear(66, width)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.feedforward = nn.Sequential(nn.LayerNorm(width),
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        # A shared readout bias would cancel exactly in the zero-sum projection.
        self.readout = nn.Linear(width, 2, bias=False)
        nn.init.zeros_(self.readout.weight)

    def correction(self, features, raw_sources):
        """Point batch [P,64], [P,3,2] -> [P,3,2], never mixes RF samples."""
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError('Expected [points,64] shared features')
        if raw_sources.shape != (features.shape[0], 3, 2):
            raise ValueError('Expected three complex predicted sources')
        shared = features[:, None].expand(-1, 3, -1)
        h = self.input(torch.cat((shared, raw_sources), -1))
        q, k, v = self.qkv(self.norm(h)).chunk(3, -1)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.width), -1)
        h = h + self.attention_output(weights @ v)
        h = h + self.feedforward(h)
        delta = self.readout(h)
        return delta - delta.mean(1, keepdim=True)

    def forward(self, features):
        raw = self.base(features)
        b, _, f, t = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, 64)
        sources = raw[:, :6].permute(0, 2, 3, 1).reshape(-1, 3, 2)
        pieces = []
        for start in range(0, flat.shape[0], self.chunk_points):
            feature_piece = flat[start:start + self.chunk_points]
            source_piece = sources[start:start + self.chunk_points]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                delta = checkpoint(self.correction, feature_piece, source_piece,
                                   use_reentrant=False)
            else:
                delta = self.correction(feature_piece, source_piece)
            pieces.append(delta)
        delta = torch.cat(pieces).reshape(b, f, t, 6).permute(0, 3, 1, 2)
        return torch.cat((raw[:, :6] + delta, raw[:, 6:]), dim=1)


def augment(net, *, checkpoint_chunks=True):
    """Wrap a loaded parent, preserving its existing output weights and bias."""
    if isinstance(net.output, SourceInteractionHead):
        raise ValueError('Head already augmented')
    net.output = SourceInteractionHead(net.output,
                                      checkpoint_chunks=checkpoint_chunks)
    return net
