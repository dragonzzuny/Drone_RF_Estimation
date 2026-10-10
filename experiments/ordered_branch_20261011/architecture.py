"""Preserve the parent U-Net and learn an aligned temporal-variation branch.

The added branch is zero for a temporally constant context, for any weights.
This is an RF adaptation hypothesis, not a novel temporal-learning theorem.
"""
import copy
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.context_model import align_context
w=worker.watch


class TemporalVariationBranch(nn.Module):
    def __init__(self,parent):
        super().__init__()
        if parent.context_mode!='mean':
            raise ValueError('Preserved mean-context parent required')
        self.parent=parent.requires_grad_(False)
        self.ordered_encoder=copy.deepcopy(parent.context_encoder).requires_grad_(True)
        self.ordered_projection=copy.deepcopy(parent.context_projection).requires_grad_(True)
        self.gate=nn.Parameter(torch.zeros(parent.context_projection.out_channels))

    def context(self,features,crop_start,fine_frames):
        if features.ndim!=3 or features.shape[1:]!=(65,255):
            raise ValueError('Original native65x255 context required')
        mean=features.mean(-1,keepdim=True).expand_as(features)
        base=self.parent.context_encoder(mean)
        def align(encoded):
            return align_context(encoded,crop_start,fine_frames,self.parent.fine_hop,
                self.parent.first_center,self.parent.context_step)
        base_bias=.1*self.parent.context_projection(align(base))
        # Shared new encoder on the ordered and constant inputs isolates
        # temporal variation from the already-preserved mean-context path.
        # This algebraically equivalent mean is exactly the anchor for a
        # constant FP32 context; repeated summation need not preserve constants.
        # The parent's original mean above remains untouched.
        anchor=features[...,:1]
        branch_mean=(anchor+(features-anchor).mean(-1,keepdim=True)).expand_as(features)
        variation=self.ordered_encoder(features)-self.ordered_encoder(branch_mean)
        correction=.1*self.ordered_projection(align(variation))
        bias=base_bias+self.gate.tanh()[None,:,None]*correction
        return bias,self.parent.count_head(base.mean(-1)),base_bias,correction

    def forward(self,z,context_features,crop_start):
        if z.ndim!=3 or not z.is_complex() or z.shape[1:]!=(512,500):
            raise ValueError('Original full512x500 complex STFT required')
        if crop_start.shape!=(z.shape[0],) or torch.any(crop_start<0):
            raise ValueError('Nonnegative crop origins required')
        if torch.any(crop_start+(z.shape[-1]-1)*self.parent.fine_hop>255*self.parent.context_step):
            raise ValueError('Crop outside original long context')
        bias,logits,_,_=self.context(context_features,crop_start,z.shape[-1])
        return dict(estimates=self.parent._forward(z,bias),count_logits=logits)


def build(checkpoint):
    torch.manual_seed(0)
    parent=worker.make_model('retained_unet',checkpoint)
    assert sum(p.numel() for p in parent.parameters())==32142859
    return TemporalVariationBranch(parent)


def optimizer(net):
    branch=list(net.ordered_encoder.parameters())+list(net.ordered_projection.parameters())
    return torch.optim.AdamW([dict(params=branch,lr=1e-4,weight_decay=1e-4,name='temporal_branch'),
        dict(params=[net.gate],lr=1e-2,weight_decay=0.,name='gate')],foreach=False)


def verify_parent(net,checkpoint):
    reference=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    for name,tensor in net.parent.state_dict().items():
        assert torch.equal(tensor.detach().cpu(),reference[name]),name
    return len(reference)
