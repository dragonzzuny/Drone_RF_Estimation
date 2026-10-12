"""Prepared source-tracking embedding; full waveform U-Net remains unchanged.

No class, reference, source count, or output-slot identity enters this encoder.
Time encoding repeats across three slots, so within-block slot permutations
commute with the network (up to ordinary floating-point roundoff).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TrackingEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape=nn.Linear(512,128)
        self.power=nn.Linear(1,128,bias=False)
        self.time_gain=nn.Parameter(torch.tensor(.1))
        layer=nn.TransformerEncoderLayer(128,4,dim_feedforward=512,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128);self.output=nn.Linear(128,128,bias=False)

    def forward(self,observed_features):
        if observed_features.ndim!=4 or observed_features.shape[2:]!=(3,513):
            raise ValueError('Observed-only [batch,time,3,513] features required')
        b,t,_,_=observed_features.shape
        h=self.shape(observed_features[...,:512])+self.power(observed_features[...,512:])
        positions=torch.arange(t,device=h.device,dtype=h.dtype)[:,None]
        rates=torch.exp(torch.arange(0,128,2,device=h.device,dtype=h.dtype)*(-math.log(10000.)/128))
        p=torch.zeros(t,128,device=h.device,dtype=h.dtype)
        p[:,0::2]=torch.sin(positions*rates);p[:,1::2]=torch.cos(positions*rates)
        h=h+self.time_gain*p[None,:,None,:]
        h=self.encoder(h.reshape(b,t*3,128)).reshape(b,t,3,128)
        return F.normalize(self.output(self.norm(h)),p=2,dim=-1,eps=1e-8)


def affinity_loss(embedding,labels,weights):
    """Permutation-invariant, energy-balanced within-observation affinity loss.

    Inactive output tokens have zero weight. The label indices are arbitrary
    within each mixture and never an absolute device/product identity.
    """
    b=embedding.shape[0];v=embedding.reshape(b,-1,128)
    y=labels.reshape(b,-1);w=weights.reshape(b,-1)
    assert y.shape==w.shape==v.shape[:2]
    assert bool(torch.all((y>=0) | (w==0))) and bool(torch.all(w>=0))
    torch.testing.assert_close(w.sum(-1),torch.ones(b,device=w.device),rtol=1e-6,atol=1e-6)
    affinity=v@v.transpose(-1,-2)
    target=(y[:,:,None]==y[:,None,:]).to(v.dtype)
    loss=((affinity-target).square()*w[:,:,None]*w[:,None,:]).sum((-1,-2))
    assert torch.isfinite(loss).all()
    return loss.mean()
