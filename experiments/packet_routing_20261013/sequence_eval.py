"""Frozen short-DEV and long-sequence diagnostics; targets only enter scoring."""
import itertools
import numpy as np
import torch
from sequence_data import Mixtures, Sequences, WINDOW, ROLES
from models import predict
from study import batch
from drone_rf.waveform import waveform_metrics
from sequence_objective import assignments
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/phase_eight_20261011'))
from core import predict_eight
PERMS=list(itertools.permutations(range(3)))


def summarize(rows):
    groups=[]
    for count in (1,2,3):
        r=[r for r in rows if r['count']==count]
        assert r
        vals=[v for row in r for v in row['si_sdr']]
        groups.append(dict(count=count,cases=len(r),mean_nmse=float(np.mean([v for row in r for v in row['nmse']])),
            mean_si_sdr=float(np.mean(vals)) if all(v is not None for v in vals) else None,
            weakest_nmse=float(np.mean([row['nmse'][row['weakest_index']] for row in r])),
            count_accuracy=float(np.mean([row['predicted_count']==count for row in r]))))
    return dict(by_count=groups,selection_nmse=float(np.mean([g['mean_nmse'] for g in groups[1:]])))


def score(y,reference,mixture,count,logits):
    active=torch.tensor([[j<count for j in range(3)]],device=y.device)
    m=waveform_metrics(y,reference,active,mixture)
    si=m['si_sdr'][0,:count].tolist()
    assert torch.isfinite(m['nmse']).all() and float(m['sum_relative_error'][0])<1e-9
    return dict(count=count,nmse=m['nmse'][0,:count].tolist(),
        si_sdr=[v if np.isfinite(v) else None for v in si],
        reference_power=m['reference_power'][0,:count].tolist(),
        weakest_index=int(m['reference_power'][0,:count].argmin()),
        assignment=m['assignment'][0].tolist(),sum_relative_error=float(m['sum_relative_error'][0]),
        predicted_count=int(logits.argmax(-1)[0])+1)


def merge(ys,starts,length):
    """Same reference-free overlap matching and Hann floor as deployed event code."""
    total=np.zeros((4,length),np.complex128);weights=np.zeros(length)
    win=np.maximum(np.hanning(WINDOW),.001);receipts=[]
    for start,y in zip(starts,ys):
        overlap=weights[start:start+WINDOW]>0
        p=(0,1,2);costs=[]
        if overlap.any():
            anchor=total[:3,start:start+WINDOW][:,overlap]/weights[start:start+WINDOW][overlap]
            costs=[float(np.mean(abs(y[list(q)][:,overlap]-anchor)**2)) for q in PERMS]
            p=PERMS[int(np.argmin(costs))]
        total[:,start:start+WINDOW]+=y[[*p,3]]*win
        weights[start:start+WINDOW]+=win
        receipts.append(dict(start=start,permutation=list(p),costs=costs))
    assert np.all(weights>0)
    return (total/weights).astype(np.complex64),receipts


@torch.no_grad()
def short_eval(net,device,progress=None):
    net.eval();dev=Mixtures(ROLES[1]);rows=[]
    for i in range(len(dev)):
        item=batch(dev[i],device)
        _,y,logits,_=predict_eight(net,predict,item)
        k=int(item['construction_count'][0]);row=score(y,item['references'],item['mixture'],k,logits)
        raw=dev.rows[i]
        row.update(index=i,categories=[dev.library.clips[int(j)]['category'] for j in raw['indices'][:k]],
            nominal_levels_db=raw['levels'][:k].tolist())
        rows.append(row)
        if progress is not None and i%50==0:progress('SHORT_DEV',case=i+1,total=len(dev))
    return dict(rows=rows,rows_sha256=dev.rows_hash,**summarize(rows))


@torch.no_grad()
def long_eval(net,device,progress=None):
    net.eval();dev=Sequences(ROLES[1]);merged=[];unmerged=[];diagnostics=[]
    for i in range(len(dev)):
        chunks=dev[i];reference=np.concatenate([v['references'] for v in chunks],-1)
        mixture=reference.sum(0);k=int(chunks[0]['construction_count'])
        start=int(chunks[0]['crop_start']);ys=[];items=[];base_preds=[];orders=[]
        for j in range(5):
            offset=j*(WINDOW//2)
            item=batch(dict(mixture=mixture[offset:offset+WINDOW].copy(),
                references=reference[:,offset:offset+WINDOW].copy(),
                active=np.any(reference[:,offset:offset+WINDOW]!=0,axis=-1),
                context_features=chunks[0]['context_features'],crop_start=start+offset,construction_count=k),device)
            _,y,logits,diag=predict_eight(net,predict,item)
            ys.append(y[0].cpu().numpy());orders.append(diag)
            if j%2==0:items.append(item);base_preds.append(y)
        result,receipt=merge(ys,[j*(WINDOW//2) for j in range(5)],3*WINDOW)
        ref=torch.from_numpy(reference)[None].to(device);mix=torch.from_numpy(mixture)[None].to(device)
        row=score(torch.from_numpy(result)[None].to(device),ref,mix,k,logits)
        raw=dev.rows[i];cats=[dev.library.clips[int(j)]['category'] for j in raw['indices'][:k]]
        row.update(index=i,categories=cats,nominal_levels_db=raw['levels'][:k].tolist())
        merged.append(row)
        no_match=torch.cat(base_preds,-1)
        r2=score(no_match,ref,mix,k,logits);r2.update(index=i,categories=cats)
        unmerged.append(r2)
        _,cost=assignments(base_preds,items,'shared')
        diagnostics.append(dict(index=i,categories=cats,assignment_cost=cost,overlap_matching=receipt))
        if progress is not None:progress('LONG_DEV',case=i+1,total=len(dev))
    return dict(merged=dict(rows=merged,**summarize(merged)),
        raw_concatenation=dict(rows=unmerged,**summarize(unmerged)),
        assignment_diagnostics=diagnostics,
        disclaimer='24 preselected same-record B development sequences; not independent recordings; no references in inference/merge')
