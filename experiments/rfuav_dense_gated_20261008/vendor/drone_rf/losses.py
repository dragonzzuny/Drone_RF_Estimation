"""Exact utterance-level permutation assignment for <=4 complex sources.

Target order may change; assignment is ONE permutation for the entire waveform,
not an oracle that switches source identity every STFT bin or short window.
Absent slots use mixture-normalized output energy, never division by zero NMSE.
"""
import itertools
import torch


def pit_waveform_loss(estimates, references, active, mixture, background_reference=None, assignment_mode='pit'):
    """[B,K+1,T] estimates; [B,K,T] refs; bool [B,K] active; [B,T] mixture.

    K final aircraft slots are unordered, while the last background slot is fixed.
    The scientific active-source label definition must be fixed by each dataset.
    This utility does not infer physical aircraft count from an RF recording.
    """
    if estimates.ndim!=3 or references.ndim!=3 or not estimates.is_complex() or not references.is_complex():
        raise ValueError('Complex [B,K,T] tensors required')
    if assignment_mode not in {'pit','fixed'}:
        raise ValueError('Use pit or fixed for the matched supervision ablation')
    b,k,t=references.shape
    if k not in (2,3,4) or estimates.shape!=(b,k+1,t) or active.shape!=(b,k) or mixture.shape!=(b,t):
        raise ValueError('Incompatible source geometry')
    if active.dtype!=torch.bool:
        raise ValueError('active must be an explicit boolean target')
    if torch.any(references.masked_select(~active[...,None].expand_as(references)).abs()>0):
        raise ValueError('Inactive reference slots must be zero')
    if background_reference is None:
        background_reference=mixture-references.sum(1)
    if background_reference.shape!=(b,t):
        raise ValueError('Background shape mismatch')
    mix_power=mixture.abs().square().mean(-1).clamp_min(1e-8)
    ref_power=references.abs().square().mean(-1)
    # Floor controls numerical explosions in locally inactive segments.
    denominator=torch.maximum(ref_power,1e-6*mix_power[:,None])
    error=(estimates[:,:k,None,:]-references[:,None,:,:]).abs().square().mean(-1)
    normalized=error/torch.where(active,denominator,mix_power[:,None])[:,None,:]
    ec=estimates[:,:k]-estimates[:,:k].mean(-1,keepdim=True)
    rc=references-references.mean(-1,keepdim=True)
    cross=(ec[:,:,None,:]*rc[:,None,:,:].conj()).mean(-1).abs().square()
    var=ec.abs().square().mean(-1)[:,:,None]*rc.abs().square().mean(-1)[:,None,:]
    coherence=1-(cross/var.clamp_min(1e-16)).clamp(0,1)
    meaningful=active & (ref_power>1e-6*mix_power[:,None])
    pair_cost=normalized+coherence*meaningful[:,None,:]
    perms=torch.tensor(list(itertools.permutations(range(k))),device=estimates.device)
    targets=torch.arange(k,device=estimates.device)
    with torch.no_grad():
        costs=pair_cost[:,perms,targets].mean(-1)
        best=costs.argmin(-1) if assignment_mode=='pit' else torch.zeros(b,device=estimates.device,dtype=torch.long)
    chosen=perms[best]
    batch=torch.arange(b,device=estimates.device)[:,None]
    source_loss=pair_cost[batch,chosen,targets].mean()
    background_nmse=(estimates[:,-1]-background_reference).abs().square().mean(-1)/mix_power
    return dict(loss=source_loss+background_nmse.mean(),source_loss=source_loss,
        background_loss=background_nmse.mean(),assignment=chosen)


def source_count_loss(logits, counts, eligible):
    """1/2/3-source CE on explicitly approved long-context count targets only.

    A nonzero recorded waveform can be receiver noise. Neither nonzero energy
    nor the number of files authorizes a physical-aircraft-count label. Callers
    must provide an audited eligibility mask; no implicit default is supplied.
    """
    if logits.ndim != 2 or logits.shape[1] != 3 or counts.shape != logits.shape[:1]:
        raise ValueError('Expected [B,3] count logits and [B] counts')
    if eligible.shape != counts.shape or eligible.dtype != torch.bool:
        raise ValueError('Explicit boolean count-target eligibility required')
    if counts.dtype != torch.long or torch.any((counts < 1) | (counts > 3)):
        raise ValueError('Count targets must be long integers 1, 2 or 3')
    values = torch.nn.functional.cross_entropy(logits, counts - 1, reduction='none')
    return (values * eligible).sum() / eligible.sum().clamp_min(1)
