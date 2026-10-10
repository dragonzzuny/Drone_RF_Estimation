"""SepTDA-inspired full separator branch for the retained complex RF U-Net.

Implements full 128-wide / 256 BLSTM / two attractor / eight triple-path
configuration, with complex STFT framing and a residual RF decoder. This is
an adaptation, not the original audio waveform/count/loss implementation.
"""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def segment(x, size=96):
    """[B,T,D] -> [B,S,K,D], with symmetric half-chunk boundary padding."""
    if x.ndim != 3 or size%2 or x.shape[1]<1:raise ValueError('Invalid segmentation')
    hop=size//2;length=x.shape[1];right=hop+(-length)%hop
    padded=F.pad(x.transpose(1,2),(hop,right))
    chunks=padded.unfold(-1,size,hop).permute(0,2,3,1).contiguous()
    return chunks,(length,hop,right)


def overlap_add(chunks,geometry):
    """Exact averaging of overlapped features; differentiable through fold."""
    length,hop,right=geometry;b,s,k,d=chunks.shape
    if k!=2*hop:raise ValueError('Chunk geometry mismatch')
    columns=chunks.permute(0,3,2,1).reshape(b,d*k,s)
    total=length+hop+right
    output=F.fold(columns,(1,total),(1,k),stride=(1,hop))[:, :, 0]
    counts=F.fold(torch.ones(1,k,s,device=chunks.device,dtype=chunks.dtype),
                  (1,total),(1,k),stride=(1,hop))[:, :, 0]
    return (output/counts)[:,:,hop:hop+length].transpose(1,2).contiguous()


class RelativeAttention(nn.Module):
    def __init__(self,width=128,heads=4,buckets=32,max_distance=128):
        super().__init__();self.heads=heads;self.buckets=buckets;self.max_distance=max_distance
        self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width)
        self.relative_bias=nn.Embedding(buckets,heads)
        nn.init.zeros_(self.relative_bias.weight)

    def bucket(self,positions):
        half=self.buckets//2;distance=positions.abs();exact=half//2
        large=exact+(torch.log(distance.float().clamp_min(exact)/exact)/math.log(self.max_distance/exact)*(half-exact)).long()
        return (positions>0).long()*half+torch.where(distance<exact,distance,large.clamp_max(half-1))

    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        pos=torch.arange(t,device=x.device)
        bias=self.relative_bias(self.bucket(pos[None,:]-pos[:,None])).permute(2,0,1)[None]
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
        return self.output(h.transpose(1,2).reshape(b,t,d))


class LSTMAttention(nn.Module):
    def __init__(self,width=128,hidden=256):
        super().__init__()
        self.recurrent_norm=nn.LayerNorm(width)
        self.recurrent=nn.LSTM(width,hidden,batch_first=True,bidirectional=True)
        self.recurrent_output=nn.Linear(2*hidden,width)
        self.attention_norm=nn.LayerNorm(width);self.attention=RelativeAttention(width)
        self.feedforward_norm=nn.LayerNorm(width)
        self.feedforward=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def forward(self,x):
        h,_=self.recurrent(self.recurrent_norm(x));x=x+self.recurrent_output(h)
        x=x+self.attention(self.attention_norm(x))
        return x+self.feedforward(self.feedforward_norm(x))


class DualPath(nn.Module):
    def __init__(self):
        super().__init__();self.intra=LSTMAttention();self.inter=LSTMAttention();self.norm=nn.LayerNorm(128)

    def forward(self,x):
        b,s,k,d=x.shape;original=x
        x=self.intra(x.reshape(b*s,k,d)).reshape(b,s,k,d)
        x=self.inter(x.permute(0,2,1,3).reshape(b*k,s,d)).reshape(b,k,s,d).permute(0,2,1,3)
        return self.norm(x+original)


class AttractorLayer(nn.Module):
    def __init__(self,first=False):
        super().__init__();self.first=first
        if not first:
            self.self_attention=nn.MultiheadAttention(128,4,batch_first=True)
            self.self_norm=nn.LayerNorm(128)
        self.cross_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.cross_norm=nn.LayerNorm(128)
        self.ff=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128));self.ff_norm=nn.LayerNorm(128)

    def forward(self,q,context):
        if not self.first:
            mask=torch.ones(q.shape[1],q.shape[1],device=q.device,dtype=torch.bool).triu(1)
            v=self.self_norm(q);q=q+self.self_attention(v,v,v,attn_mask=mask,need_weights=False)[0]
        q=q+self.cross_attention(self.cross_norm(q),context,context,need_weights=False)[0]
        return q+self.ff(self.ff_norm(q))


class TriplePath(nn.Module):
    def __init__(self):
        super().__init__();self.temporal=DualPath()
        self.source=nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(128)

    def forward(self,x):
        b,c,s,k,d=x.shape;original=x
        x=self.temporal(x.reshape(b*c,s,k,d)).reshape(b,c,s,k,d)
        h=x.permute(0,2,3,1,4).reshape(b*s*k,c,d)
        h=self.source(h).reshape(b,s,k,c,d).permute(0,3,1,2,4)
        return self.norm(h+original)


class RFAttractorSeparator(nn.Module):
    def __init__(self,frequency_bins=512,checkpoint_blocks=True):
        super().__init__();self.frequency_bins=frequency_bins;self.checkpoint_blocks=checkpoint_blocks
        self.encoder=nn.Conv1d(2*frequency_bins,256,3,padding=1)
        self.project=nn.Linear(256,128);self.dual=DualPath()
        self.queries=nn.Parameter(torch.randn(3,128)*.02)
        self.attractors=nn.ModuleList([AttractorLayer(first=True),AttractorLayer()])
        self.film_scale=nn.Linear(128,128);self.film_shift=nn.Linear(128,128)
        self.triple=nn.ModuleList(TriplePath() for _ in range(8))
        self.decode_features=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256),nn.GELU())
        self.source_readout=nn.Linear(256,2*frequency_bins)
        self.background_readout=nn.Linear(256,2*frequency_bins)
        for layer in (self.source_readout,self.background_readout):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def block(self,fn,*args):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(fn,*args,use_reentrant=False)
        return fn(*args)

    def latent(self,z):
        if not z.is_complex() or z.ndim!=3 or z.shape[1]!=self.frequency_bins:
            raise ValueError('Expected both-sided complex STFT [B,512,T]')
        b,f,t=z.shape
        encoded=F.gelu(self.encoder(torch.cat((z.real,z.imag),1))).transpose(1,2)
        chunks,geometry=segment(self.project(encoded))
        chunks=self.block(self.dual,chunks)
        context=overlap_add(chunks,geometry)
        q=self.queries[None].expand(b,-1,-1)
        for layer in self.attractors:q=self.block(layer,q,context)
        h=self.film_scale(q)[:,:,None,None,:]*chunks[:,None]+self.film_shift(q)[:,:,None,None,:]
        for layer in self.triple:h=self.block(layer,h)
        _,c,s,k,d=h.shape
        h=overlap_add(h.reshape(b*c,s,k,d),geometry).reshape(b,c,t,d)
        return self.decode_features(h),encoded,q

    def forward(self,z):
        source_features,mixture_features,_=self.latent(z)
        raw=self.source_readout(source_features)
        background=self.background_readout(mixture_features)[:,None]
        raw=torch.cat((raw,background),1)
        real,imag=raw.chunk(2,-1)
        return torch.complex(real,imag).permute(0,1,3,2).contiguous()


def forward(self,z,bottleneck_context=None):
    parent=self._septda_parent_forward(z,bottleneck_context)
    scale=z.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
    correction=self.septda(z/scale)*scale[:,None]
    # Every signal has an explicit learned estimate. Project all four streams,
    # including background; never force the third signal to be the residual.
    correction=correction-correction.mean(1,keepdim=True)
    return parent+correction


def augment(net):
    if hasattr(net,'septda') or net.max_sources!=3:raise ValueError('Expected unaugmented three-source parent')
    net.septda=RFAttractorSeparator()
    net._septda_parent_forward=net._forward
    net._forward=types.MethodType(forward,net)
    return net
