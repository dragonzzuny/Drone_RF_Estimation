"""Preserve the full parent and progressively admit ordered power context.

This is an information-flow ablation, not a SepFormer/SepTDA reproduction.
All three complex outputs remain independently estimated by the full U-Net.
"""
import torch
from torch import nn


class OrderedContext(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.order_gate = nn.Parameter(torch.zeros(base.input.in_channels))

    def forward(self, features):
        mean = features.mean(-1, keepdim=True).expand_as(features)
        # Signed bounded coefficients; not probabilities or source counts.
        ordered = mean + self.order_gate.tanh()[None, :, None] * (features - mean)
        return self.base(ordered)


def augment(net):
    if net.context_mode != 'mean' or isinstance(net.context_encoder, OrderedContext):
        raise ValueError('A loaded mean-context parent is required')
    net.context_encoder = OrderedContext(net.context_encoder)
    # Mean operation is now inside OrderedContext; raw time order reaches it.
    net.context_mode = 'ordered'
    return net


def parent_key(key):
    if key == 'context_encoder.order_gate':
        return None
    return key.replace('context_encoder.base.', 'context_encoder.', 1)
