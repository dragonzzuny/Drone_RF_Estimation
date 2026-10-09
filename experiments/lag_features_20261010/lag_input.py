"""Exact-sample lagged complex STFT relationships at the first U-Net layer.

These are mixture features, not separated sources. Delaying the reconstructed
mixture by integer samples avoids rounding measured RF lags to STFT hops.
The power-only control uses the same delayed observations. The additional
complex products preserve relative phase; finite-mixture cross terms remain.
"""
import torch
from torch import nn
from torch.nn import functional as F
from drone_rf.waveform import analyze, synthesize

LAGS=(3334,6667,14270)


def lag_features(z):
    if z.ndim!=3 or not z.is_complex() or z.shape[1]!=512:
        raise ValueError('Expected complex STFT512 [B,F,T]')
    length=(z.shape[-1]-1)*128
    if length<=512:raise ValueError('Insufficient waveform length')
    waveform=synthesize(z,length)
    centers=torch.arange(z.shape[-1],device=z.device)*128
    current=z.abs().square();power=[];phase=[]
    for lag in LAGS:
        past=F.pad(waveform[:,:-lag],(lag,0)) if lag<length else torch.zeros_like(waveform)
        delayed=analyze(past)
        previous=delayed.abs().square()
        denominator=current+previous+1e-6
        valid=((centers>=lag+256)&(centers<=length-256))[None,None]
        cross=2*z*delayed.conj()/denominator
        power.extend((current/denominator*valid,previous/denominator*valid))
        phase.extend((cross.real*valid,cross.imag*valid))
    return torch.stack(power,1),torch.stack(phase,1)


class LagInputConv(nn.Module):
    def __init__(self,base,use_phase):
        super().__init__()
        if base.in_channels!=4 or base.out_channels!=64:
            raise ValueError('Expected original 4-to-64 full U-Net input convolution')
        self.base=base
        self.power=nn.Conv2d(6,64,3,padding=1,bias=False)
        self.phase=nn.Conv2d(6,64,3,padding=1,bias=False) if use_phase else None
        nn.init.zeros_(self.power.weight)
        if self.phase is not None:nn.init.zeros_(self.phase.weight)

    def forward(self,x):
        z=torch.complex(x[:,0],x[:,1])
        power,phase=lag_features(z)
        output=self.base(x)+self.power(power)
        if self.phase is not None:output=output+self.phase(phase)
        return output


def augment(net,arm):
    if arm not in ('lag_power','lag_power_phase'):raise ValueError(arm)
    net.down[0][0]=LagInputConv(net.down[0][0],use_phase=arm=='lag_power_phase')
    return net
