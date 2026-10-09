"""Full U-Net plus frequency/time recurrent bottleneck, TF-GridNet-inspired.

This is NOT a TF-GridNet reproduction: the complete retained U-Net remains,
and recurrence operates on its coarsened bottleneck grid. It has no cross-frame
attention and does not invent inter-receiver spatial cues for a single RF input.
"""
import types
import torch
from torch import nn
from torch.nn import functional as F


class AxisBlock(nn.Module):
    def __init__(self,width=256,hidden=256):
        super().__init__()
        self.frequency_norm=nn.LayerNorm(width)
        self.frequency=nn.LSTM(width,hidden,batch_first=True,bidirectional=True)
        self.frequency_output=nn.Linear(2*hidden,width)
        self.time_norm=nn.LayerNorm(width)
        self.time=nn.LSTM(width,hidden,batch_first=True,bidirectional=True)
        self.time_output=nn.Linear(2*hidden,width)

    def forward(self,value):
        b,c,f,t=value.shape
        x=value.permute(0,3,2,1).reshape(b*t,f,c)
        h,_=self.frequency(self.frequency_norm(x));x=x+self.frequency_output(h)
        x=x.reshape(b,t,f,c).permute(0,2,1,3).reshape(b*f,t,c)
        h,_=self.time(self.time_norm(x));x=x+self.time_output(h)
        return x.reshape(b,f,t,c).permute(0,3,1,2)


class DualAxisAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv2d(1024,256,1)
        self.blocks=nn.ModuleList(AxisBlock() for _ in range(2))
        self.output=nn.Conv2d(256,1024,1)
        # Exact parent predictions initially. First step trains this readout;
        # internal recurrence receives gradients after readout becomes nonzero.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        h=self.input(x)
        for block in self.blocks:h=block(h)
        return x+self.output(h)


def forward(self,z,bottleneck_context=None):
    if not z.is_complex() or z.ndim!=3 or min(z.shape[-2:])<16:
        raise ValueError('Expected complex [B,F,T], F,T >=16')
    scale=z.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
    norm=z/scale
    x=torch.stack((norm.real,norm.imag,torch.full_like(norm.real,.5),torch.log1p(norm.abs())),1)
    skips=[]
    for i,layer in enumerate(self.down):
        if i:x=F.avg_pool2d(x,2)
        x=layer(x);skips.append(x)
    if bottleneck_context is not None:
        if bottleneck_context.shape!=(x.shape[0],x.shape[1],x.shape[3]):raise ValueError('Context shape')
        x=x+bottleneck_context[:,:,None,:]
    x=self.tf_axes(x)
    for layer,skip in zip(self.up,reversed(skips[:-1])):
        x=layer(torch.cat((F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
    raw=self.output(x).float()
    residual=torch.complex(raw[:,0::2],raw[:,1::2])*scale[:,None]
    return residual+(z-residual.sum(1))[:,None]/(self.max_sources+1)


def augment(net):
    if hasattr(net,'tf_axes'):raise ValueError('Already augmented')
    if net.max_sources!=3:raise ValueError('Three sources plus background required')
    net.tf_axes=DualAxisAdapter()
    net._forward=types.MethodType(forward,net)
    return net
