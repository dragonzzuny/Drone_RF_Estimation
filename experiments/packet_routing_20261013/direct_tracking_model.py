"""Global-PIT expected waveform error over discrete block permutations.

Inference only permutes existing I/Q blocks. The soft probabilities used in
training describe categorical choices, not an amplitude-weighted waveform mix.
"""
import numpy as np
import torch
from torch import nn
from learned_tracking_model import TrackingEmbedding
from spectral_routing import PERMS


class DirectTracking(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=TrackingEmbedding()
        self.identity=nn.Linear(128,3,bias=False)

    def forward(self,observed_features):
        logits=self.identity(self.encoder(observed_features))
        return torch.stack([sum(logits[:,:,p[j],j] for j in range(3)) for p in PERMS],dim=-1)


def waveform_cost(cache,count):
    energy=cache['energy_r'].sum(0);mixture_energy=float(cache['energy_m'].sum())
    assert count in (1,2,3) and np.all(energy[:count]>0) and mixture_energy>0
    weight=np.zeros(3,np.float64)
    weight[:count]=1/(count*energy[:count])
    if count<3:weight[count:]=.1/((3-count)*mixture_energy)
    cost=np.empty((len(cache['bounds']),6,6),np.float64)
    for q,p in enumerate(PERMS):
        for g,reference_order in enumerate(PERMS):
            cost[:,q,g]=sum(weight[reference_order[j]]*cache['sse'][:,p[j],reference_order[j]] for j in range(3))
    assert np.isfinite(cost).all() and np.all(cost>=0)
    return cost.astype(np.float32)


def expected_waveform_loss(logits,cost):
    assert logits.shape==cost.shape[:-1] and logits.shape[-1]==6 and cost.shape[-1]==6
    expected=(logits.softmax(-1)[...,None]*cost).sum((1,2))
    loss=expected.min(-1).values.mean()
    assert torch.isfinite(loss)
    return loss


def decode(logits):
    """One injective permutation per block; no count or reference argument."""
    probabilities=logits.detach().softmax(-1)
    choice=probabilities.argmax(-1).detach().cpu().numpy()
    orders=np.asarray(PERMS,dtype=np.int64)[choice]
    receipt=dict(mean_max_probability=float(probabilities.max(-1).values.mean()),
        mean_entropy=float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()))
    return orders,receipt
