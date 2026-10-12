"""Mixture-only neighbouring context; unchanged primary targets and C8 matching.

The new model receives three adjacent fine windows. For a fixed original target,
use a central target when possible, otherwise move it to the available edge.
All samples remain inside the same admitted 20.8896 ms mixture context. Sources
are read only to construct that observation; references never enter inference.
"""
import numpy as np
import torch
from sequence_data import Mixtures, Sequences, WINDOW, LENGTH, ROLES, component_gains
from sequence_eval import score, summarize, merge, predict_eight
from sequence_objective import assignments
from study import batch
from chunk_temporal_candidate import predict_primary


def neighbour_start(start):
    if not 0 <= start <= LENGTH-WINDOW:
        raise ValueError('Target window outside its admitted context')
    first = start-WINDOW if WINDOW <= start <= LENGTH-2*WINDOW else (
        start if start < WINDOW else start-2*WINDOW)
    assert 0 <= first <= LENGTH-3*WINDOW
    assert (start-first)//WINDOW in (0,1,2) and (start-first)%WINDOW == 0
    return first


def observed_segment(data, index, start, length):
    """Match the existing per-source complex64 cast and summation exactly."""
    assert 0 <= start and start+length <= LENGTH
    row=data.rows[index];k=int(row['count']);ids=row['indices'][:k]
    gains=component_gains([data.library.clips[int(i)]['mean_power'] for i in ids],
                          row['levels'][:k],row['phases'][:k])
    contributions=np.zeros((3,length),np.complex64)
    for j,(i,gain) in enumerate(zip(ids,gains)):
        assert data.library.clips[int(i)]['diagnostic_role']==data.role
        contributions[j]=data.library._array(int(i))[start:start+length]*gain
    return contributions.sum(0)


def add_neighbours(data, index, raw, device):
    start=int(raw['crop_start']);first=neighbour_start(start)
    observation=observed_segment(data,index,first,3*WINDOW)
    offset=start-first
    assert np.array_equal(observation[offset:offset+WINDOW],raw['mixture'])
    result=batch(raw,device)
    result['long_mixture']=torch.from_numpy(observation)[None].to(device)
    result['long_start']=torch.tensor([first],dtype=torch.long,device=device)
    return result


@torch.no_grad()
def short_eval(net,device,progress=None):
    net.eval();dev=Mixtures(ROLES[1]);rows=[]
    for i in range(len(dev)):
        item=add_neighbours(dev,i,dev[i],device)
        _,y,logits,_=predict_eight(net,predict_primary,item)
        k=int(item['construction_count'][0]);row=score(y,item['references'],item['mixture'],k,logits)
        raw=dev.rows[i]
        row.update(index=i,categories=[dev.library.clips[int(j)]['category'] for j in raw['indices'][:k]],
                   nominal_levels_db=raw['levels'][:k].tolist())
        rows.append(row)
        if progress is not None and i%25==0:progress('SHORT_DEV',case=i+1,total=len(dev))
    return dict(rows=rows,rows_sha256=dev.rows_hash,**summarize(rows),
        context='3 adjacent observed windows; primary targets unchanged; mixture-only neighbours')


@torch.no_grad()
def long_eval(net,device,progress=None):
    net.eval();dev=Sequences(ROLES[1]);merged=[];unmerged=[];diagnostics=[]
    for i in range(len(dev)):
        chunks=dev[i];reference=np.concatenate([v['references'] for v in chunks],-1)
        mixture=reference.sum(0);k=int(chunks[0]['construction_count'])
        start=int(chunks[0]['crop_start']);ys=[];items=[];base_preds=[]
        for j in range(5):
            offset=j*(WINDOW//2)
            raw=dict(mixture=mixture[offset:offset+WINDOW].copy(),
                references=reference[:,offset:offset+WINDOW].copy(),
                active=np.any(reference[:,offset:offset+WINDOW]!=0,axis=-1),
                context_features=chunks[0]['context_features'],crop_start=start+offset,construction_count=k)
            item=add_neighbours(dev,i,raw,device)
            _,y,logits,_=predict_eight(net,predict_primary,item)
            ys.append(y[0].cpu().numpy())
            if j%2==0:items.append(item);base_preds.append(y)
        result,receipt=merge(ys,[j*(WINDOW//2) for j in range(5)],3*WINDOW)
        ref=torch.from_numpy(reference)[None].to(device);mix=torch.from_numpy(mixture)[None].to(device)
        row=score(torch.from_numpy(result)[None].to(device),ref,mix,k,logits)
        raw=dev.rows[i];cats=[dev.library.clips[int(j)]['category'] for j in raw['indices'][:k]]
        row.update(index=i,categories=cats,nominal_levels_db=raw['levels'][:k].tolist());merged.append(row)
        r2=score(torch.cat(base_preds,-1),ref,mix,k,logits);r2.update(index=i,categories=cats);unmerged.append(r2)
        _,cost=assignments(base_preds,items,'shared')
        diagnostics.append(dict(index=i,categories=cats,assignment_cost=cost,overlap_matching=receipt))
        if progress is not None:progress('LONG_DEV',case=i+1,total=len(dev))
    return dict(merged=dict(rows=merged,**summarize(merged)),
        raw_concatenation=dict(rows=unmerged,**summarize(unmerged)),assignment_diagnostics=diagnostics,
        disclaimer='Same24 B DEV targets; extra observed neighbouring context; not independent recording evaluation')
