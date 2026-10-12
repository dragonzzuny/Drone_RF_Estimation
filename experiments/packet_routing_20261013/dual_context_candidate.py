"""Conditional short complex context plus ordered long-power context.

Reuse the earlier temporal-variation idea on the current B model; this is not a
first implementation of ordered context. Both additions are initially zero.
The entire original U-Net and its mean-context/count path remain present.
"""
import copy
import torch
from chunk_temporal_candidate import ChunkTemporalUNet
from drone_rf.context_model import align_context


ADDED_PREFIXES=('gates.','ordered_encoder.','ordered_projection.')


class DualContextUNet(ChunkTemporalUNet):
    def __init__(self):
        super().__init__()
        self.ordered_encoder=copy.deepcopy(self.context_encoder)
        self.ordered_projection=copy.deepcopy(self.context_projection)
        torch.nn.init.zeros_(self.ordered_projection.weight)

    def initialize_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and missing and all(k.startswith(ADDED_PREFIXES) for k in missing)
        self.ordered_encoder.load_state_dict(self.context_encoder.state_dict())
        for key,value in state.items():assert torch.equal(self.state_dict()[key],value),key
        assert not bool(self.ordered_projection.weight.count_nonzero())
        assert not bool(self.gates.expand.weight.count_nonzero())
        return self

    def variation(self,features):
        # The same new encoder is applied twice. A constant context has exactly
        # zero variation, including padding effects, for every branch weight.
        anchor=features[...,:1]
        constant=(anchor+(features-anchor).mean(-1,keepdim=True)).expand_as(features)
        delta=self.ordered_encoder(features)-self.ordered_encoder(constant)
        return .1*self.ordered_projection(delta)

    def forward(self,spectra,context_features,crop_starts,primary_index=None):
        if not isinstance(spectra,(list,tuple)) or len(spectra)!=3 or len(crop_starts)!=3:
            raise ValueError('Exactly three observed ordered fine windows required')
        assert context_features.shape[1:]==(65,255) and torch.isfinite(context_features).all()
        mean=context_features.mean(-1,keepdim=True).expand_as(context_features)
        encoded=self.context_encoder(mean);extra=self.variation(context_features);states=[]
        for z,start in zip(spectra,crop_starts):
            assert z.is_complex() and z.ndim==3 and min(z.shape[-2:])>=16
            end=start+(z.shape[-1]-1)*self.fine_hop
            assert bool(torch.all(start>=0)) and bool(torch.all(end<=255*self.context_step))
            def aligned(value):
                return align_context(value,start,z.shape[-1],self.fine_hop,self.first_center,self.context_step)
            bias=.1*self.context_projection(aligned(encoded))+aligned(extra)
            states.append(self.encode(z,bias))
        lengths=[x.shape[-1] for x,_,_ in states]
        joined=self.gates(torch.cat([x for x,_,_ in states],dim=-1)).split(lengths,dim=-1)
        if primary_index is not None:
            if self.training or len(context_features)!=1 or primary_index not in (0,1,2):
                raise ValueError('Single-window evaluation only for selective decoding')
            _,skips,scale=states[primary_index]
            return dict(estimates_primary=self.decode(joined[primary_index],skips,scale,spectra[primary_index]),
                        count_logits=self.count_head(encoded.mean(-1)))
        predictions=[self.decode(x,skips,scale,z) for x,(_,skips,scale),z in zip(joined,states,spectra)]
        return dict(estimates=predictions,count_logits=self.count_head(encoded.mean(-1)))
