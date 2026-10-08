"""Matched short/long complex-context U-Nets, both trained from scratch.

RF Challenge motivates a long first kernel and time-domain RF processing:
https://arxiv.org/html/2409.08839v3
This is a new RFUAV multi-output adaptation, not their published architecture
or a reproduction. The established 64/128/256/512/1024 widths are preserved.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

LEGACY = Path(__file__).resolve().parents[1] / 'rfuav_dense_gated_20261008'
sys.path.insert(0, str(LEGACY / 'vendor'))
from drone_rf.context_model import TemporalEncoder, align_context

ARMS = ('short_context', 'long_context')
PACK = 16
LONG_SAMPLES = 1048576
TARGET_SAMPLES = 63872


def pack_iq(z):
    """Lossless sample-to-channel rearrangement, NOT filtered decimation."""
    if z.ndim!=2 or not z.is_complex() or z.shape[-1]%PACK:
        raise ValueError('Complex waveform length must be divisible by PACK')
    values=z.reshape(z.shape[0],-1,PACK).transpose(1,2)
    return torch.cat((values.real,values.imag),1)


def unpack_iq(x):
    if x.ndim!=3 or x.shape[1]!=2*PACK:
        raise ValueError('Expected real packed I/Q')
    return torch.complex(x[:,:PACK],x[:,PACK:]).transpose(1,2).reshape(x.shape[0],-1)


class WaveBlock(nn.Sequential):
    def __init__(self, incoming, outgoing, dilation=1, first_kernel=9):
        super().__init__(nn.Conv1d(incoming, outgoing, first_kernel,
                padding=dilation*(first_kernel//2), dilation=dilation),
            nn.GroupNorm(8, outgoing), nn.SiLU(),
            nn.Conv1d(outgoing, outgoing, 9, padding=4*dilation, dilation=dilation),
            nn.GroupNorm(8, outgoing), nn.SiLU())


class WaveformUNet(nn.Module):
    """Direct complex waveform output, 3 unordered slots and one background.

    Both arms use identical parameters and compute shapes. Short control masks
    samples outside the fixed scoring window BEFORE entering the network.
    Native I/Q samples are preserved in 16 polyphase lanes, not averaged out.
    The long arm sees 10.48576 ms; both score the same 0.63872 ms target region.
    GroupNorm/RMS effects are part of this input-masking contrast, not isolated.
    """
    def __init__(self, long_context):
        super().__init__()
        self.long_context=long_context
        widths=(64,128,256,512,1024)
        dilations=(1,1,2,4,8)
        self.down=nn.ModuleList(WaveBlock(2*PACK if i==0 else widths[i-1],width,
                                        dilations[i],101 if i==0 else 9)
                                for i,width in enumerate(widths))
        self.up=nn.ModuleList(WaveBlock(widths[i+1]+widths[i],widths[i],dilations[i])
                              for i in (3,2,1,0))
        self.output=nn.Conv1d(64,8*PACK,1)
        nn.init.normal_(self.output.weight,std=1e-3)
        nn.init.zeros_(self.output.bias)
        self.context_encoder=TemporalEncoder('tcn')
        self.context_projection=nn.Conv1d(128,1024,1,bias=False)
        self.count_head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))

    def forward(self, mixture, long_mixture, context_features, crop_start, long_start):
        if (mixture.ndim!=2 or not mixture.is_complex() or mixture.shape[-1]!=TARGET_SAMPLES
                or long_mixture.shape!=(mixture.shape[0],LONG_SAMPLES) or not long_mixture.is_complex()):
            raise ValueError('Complex short/long mixture geometry changed')
        if (context_features.shape!=(mixture.shape[0],65,256)
                or crop_start.shape!=(mixture.shape[0],)
                or torch.any(crop_start<0)
                or long_start.shape!=crop_start.shape or torch.any(long_start<0)
                or torch.any(long_start+LONG_SAMPLES>2097152)
                or torch.any(crop_start<long_start)
                or torch.any(crop_start+TARGET_SAMPLES>long_start+LONG_SAMPLES)
                or not torch.isfinite(context_features).all()
                or not torch.isfinite(long_mixture).all()):
            raise ValueError('Invalid mixture-only input geometry or values')
        scale=mixture.abs().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        positions=torch.arange(LONG_SAMPLES,device=mixture.device)[None]
        offset=(crop_start-long_start)[:,None]
        if self.long_context:
            observed=long_mixture
        else:
            observed=long_mixture*((positions>=offset)&(positions<offset+TARGET_SAMPLES))
        x=pack_iq(observed/scale)
        features=context_features.mean(-1,keepdim=True).expand_as(context_features)
        encoded=self.context_encoder(features)
        skips=[]
        for i,layer in enumerate(self.down):
            if i:
                x=F.avg_pool1d(x,4)
            x=layer(x); skips.append(x)
        # A packed frame centers at sample 7.5, four 4x pools at 2047.5.
        aligned=align_context(encoded,long_start+(PACK-1)/2,LONG_SAMPLES//PACK,
                              fine_hop=PACK,pooling_stride=256)
        x=x+.1*self.context_projection(aligned)
        for layer,skip in zip(self.up,reversed(skips[:-1])):
            x=layer(torch.cat((F.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        raw=self.output(x).float()
        packed=raw.reshape(raw.shape[0]*4,2*PACK,raw.shape[-1])
        full=unpack_iq(packed).reshape(raw.shape[0],4,LONG_SAMPLES)*scale[:,None]
        # Project only the scored interval; no extra long-reference supervision.
        indices=(offset+torch.arange(TARGET_SAMPLES,device=mixture.device)[None]).long()
        estimates=full.gather(2,indices[:,None].expand(-1,4,-1))
        estimates=estimates+(mixture-estimates.sum(1))[:,None]/4
        return estimates,self.count_head(encoded.mean(-1))


def build(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(0)
    return WaveformUNet(long_context=arm=='long_context')


def predict(model, item):
    # Explicit whitelist: no reference waveform, aircraft identity or true count.
    mixture=item['mixture']
    return model(mixture,item['long_mixture'],item['context_features'],
                 item['crop_start'],item['long_start'])
