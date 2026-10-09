"""Supervised source-affinity auxiliary; waveform inference is unchanged.

Chimera/deep-clustering inspired, not their reproduction. Soft targets are
sqrt of per-source power fractions, not a mixture-power decomposition.
The auxiliary never supplies references or the constructed count to forward.
"""
import torch
from torch import nn
from torch.nn import functional as F


def targets(power, kind):
    """power: [B,C,F,T], nonnegative active-reference powers, inactive=0.

    Returns unit-row targets [B,N,C], normalized weights [B,N]. The weight
    measure averages each active source's energy-normalized distribution.
    Equal source contributions to this measure do not imply equal gradients.
    """
    if kind not in ('hard','soft'):raise ValueError(kind)
    p=power.detach().flatten(2).transpose(1,2).double()
    if not bool(torch.isfinite(p).all()) or bool((p<0).any()):raise ValueError('Invalid power')
    energy=p.sum(1,keepdim=True)
    active=energy>0
    n=active.sum(-1,keepdim=True).clamp_min(1)
    weight=(p/torch.where(active,energy,1.)).sum(-1)/n[...,0]
    total=p.sum(-1,keepdim=True);valid=total>0
    fraction=p/torch.where(valid,total,1.)
    if kind=='soft':y=fraction.sqrt()
    else:y=F.one_hot(p.argmax(-1),num_classes=p.shape[-1]).double()*valid
    # Identically zero bins have zero weight and zero target.
    return y,weight


def loss(embedding, y, weight):
    """Exact weighted affinity Frobenius loss without allocating N x N.

    E: [B,N,D] unit rows. Computation uses FP64 small Gram matrices and
    FP64 reductions to control cancellation; autograd returns FP32 to head.
    L=||E'WE||_F^2 -2||E'WY||_F^2 +||Y'WY||_F^2.
    """
    if embedding.shape[:2]!=y.shape[:2] or weight.shape!=y.shape[:2]:raise ValueError('Shape')
    e=embedding.double();y=y.double();w=weight.double()[...,None]
    ee=e.transpose(1,2)@(w*e)
    ey=e.transpose(1,2)@(w*y)
    yy=y.transpose(1,2)@(w*y)
    value=ee.square().sum((1,2))-2*ey.square().sum((1,2))+yy.square().sum((1,2))
    if not bool(torch.isfinite(value).all()) or bool((value < -1e-10).any()):raise FloatingPointError('Affinity loss')
    return value.mean()


class Capture:
    """Capture decoder feature only during an explicitly scoped forward."""
    def __init__(self,net):
        if hasattr(net,'source_affinity'):raise ValueError('Already augmented')
        if net.output.in_channels!=64:raise ValueError('Expected original full U-Net')
        # Initialization does not consume the caller's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0);net.source_affinity=nn.Conv2d(64,16,1)
        self.net=net;self.embedding=None;self.enabled=False
        self.handle=net.output.register_forward_pre_hook(self._hook)

    def _hook(self,module,args):
        if self.enabled:
            if self.embedding is not None:raise RuntimeError('Unconsumed captured features')
            raw=self.net.source_affinity(F.avg_pool2d(args[0],2))
            self.embedding=F.normalize(torch.tanh(raw).flatten(2).transpose(1,2),dim=-1,eps=1e-8)

    def take(self):
        value=self.embedding;self.embedding=None
        if value is None:raise RuntimeError('No captured embedding')
        return value

    def close(self):
        self.handle.remove();self.embedding=None;self.enabled=False
