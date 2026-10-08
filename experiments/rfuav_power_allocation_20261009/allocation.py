"""Supervised relative STFT power allocation; inference is unchanged.

Use the waveform loss's ONE whole-window source permutation, with the background
fixed. Target powers are NOT assumed to add to mixture power. Energy weighting
keeps nearly empty frequency bins from dominating the auxiliary objective.
"""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0,str(LEGACY))
sys.path.insert(0,str(LEGACY/'vendor'))
from drone_rf.waveform import analyze


def allocation_kl(estimates,references,assignment):
    b,k,t=references.shape
    if estimates.shape!=(b,k+1,t) or assignment.shape!=(b,k):
        raise ValueError('Incompatible source geometry')
    aligned=estimates[:,:k].gather(1,assignment[...,None].expand(-1,-1,t))
    aligned=torch.cat((aligned,estimates[:,-1:]),1)
    # The primary experiment has no extra additive background beyond references.
    # Numerical mixture-sum roundoff is not given a new spectral target.
    targets=torch.cat((references,torch.zeros_like(references[:,:1])),1)
    p=analyze(aligned.reshape(-1,t)).abs().square().reshape(b,k+1,512,-1)
    q=analyze(targets.reshape(-1,t)).abs().square().reshape_as(p)
    energy=q.sum(1,keepdim=True)
    scale=energy.mean((2,3),keepdim=True).clamp_min(1e-20)
    # Relative smoothing avoids log(0), also for an empty predicted slot.
    epsilon=1e-8*scale
    pred=(p+epsilon)/(p.sum(1,keepdim=True)+(k+1)*epsilon)
    truth=q/energy.clamp_min(1e-20)
    divergence=(truth*(truth.clamp_min(1e-20).log()-pred.log())).sum(1)
    weight=energy[:,0]
    return ((divergence*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-20)).mean()
