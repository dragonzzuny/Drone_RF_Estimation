"""Conditional time connections across chunks, preserving each fine U-Net path.

Unlike increasing the STFT input length, each window keeps its own normalization,
STFT boundaries and skip connections. Only bottleneck features cross windows.
Zero adapter projection reproduces the parent on every chunk initially.
This is inspired by chunk-based source separation, not a SkiM reproduction.
"""
import torch
from torch.nn import functional as F
from long_temporal_candidate import LongTemporalUNet
from drone_rf.context_model import align_context
from drone_rf.waveform import analyze, synthesize


class ChunkTemporalUNet(LongTemporalUNet):
    def encode(self, z, context):
        scale=z.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
        norm=z/scale
        x=torch.stack((norm.real,norm.imag,torch.full_like(norm.real,.5),torch.log1p(norm.abs())),dim=1)
        skips=[]
        for i,layer in enumerate(self.down):
            if i:x=F.avg_pool2d(x,2)
            x=layer(x);skips.append(x)
        assert context.shape==(x.shape[0],x.shape[1],x.shape[3])
        return x+context[:,:,None,:],skips,scale

    def decode(self, x, skips, scale, z):
        for layer,skip in zip(self.up,reversed(skips[:-1])):
            x=layer(torch.cat((F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),dim=1))
        raw=self.output(x).float()
        residual=torch.complex(raw[:,0::2],raw[:,1::2])*scale[:,None]
        return residual+(z-residual.sum(1))[:,None]/4

    def forward(self, spectra, context_features, crop_starts, primary_index=None):
        if not isinstance(spectra,(list,tuple)) or len(spectra)!=3 or len(crop_starts)!=3:
            raise ValueError('Exactly three ordered STFT windows and crop origins required')
        if context_features.shape[1:]!=(65,255):
            raise ValueError('Original mixed65x255 features required')
        features=context_features.mean(-1,keepdim=True).expand_as(context_features)
        encoded=self.context_encoder(features)
        states=[]
        for z,start in zip(spectra,crop_starts):
            if not z.is_complex() or z.ndim!=3 or min(z.shape[-2:])<16:
                raise ValueError('Complex STFT windows required')
            end=start+(z.shape[-1]-1)*self.fine_hop
            assert bool(torch.all(start>=0)) and bool(torch.all(end<=255*self.context_step))
            aligned=align_context(encoded,start,z.shape[-1],self.fine_hop,self.first_center,self.context_step)
            states.append(self.encode(z,.1*self.context_projection(aligned)))
        lengths=[x.shape[-1] for x,_,_ in states]
        mixed=self.gates(torch.cat([x for x,_,_ in states],dim=-1))
        parts=mixed.split(lengths,dim=-1)
        if primary_index is not None:
            if self.training or len(context_features)!=1 or primary_index not in (0,1,2):
                raise ValueError('Single-window evaluation only for selective decoding')
            _,skips,scale=states[primary_index]
            return dict(estimates_primary=self.decode(parts[primary_index],skips,scale,spectra[primary_index]),
                        count_logits=self.count_head(encoded.mean(-1)))
        estimates=[self.decode(x,skips,scale,z) for x,(_,skips,scale),z in zip(parts,states,spectra)]
        return dict(estimates=estimates,count_logits=self.count_head(encoded.mean(-1)))


def predict_bundle(net, items):
    if len(items)!=3:raise ValueError('Three contiguous observed windows required')
    width=items[0]['mixture'].shape[-1]
    for left,right in zip(items,items[1:]):
        assert torch.equal(left['crop_start']+width,right['crop_start'])
        assert torch.equal(left['context_features'],right['context_features'])
    # No references, construction counts or categories enter the network.
    spectra=[analyze(item['mixture']) for item in items]
    out=net(spectra,items[0]['context_features'],[item['crop_start'] for item in items])
    return [synthesize(z,width) for z in out['estimates']],out['count_logits']


def predict_primary(net, item):
    width=item['mixture'].shape[-1]
    if item['long_mixture'].shape[-1]!=3*width:
        raise ValueError('Three fine windows of mixture-only context required')
    delta=item['crop_start']-item['long_start'];indices=delta//width
    assert bool(torch.all(delta%width==0)) and bool(torch.all((indices>=0)&(indices<3)))
    if len(indices)!=1:raise ValueError('C8 inference expects one target window at a time')
    spectra=[analyze(item['long_mixture'][:,j*width:(j+1)*width]) for j in range(3)]
    out=net(spectra,item['context_features'],[item['long_start']+j*width for j in range(3)],
            primary_index=int(indices[0]))
    return synthesize(out['estimates_primary'],width),out['count_logits']
