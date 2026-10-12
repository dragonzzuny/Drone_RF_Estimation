"""Exact sufficient statistics for rearranging already estimated I/Q blocks.

References are confined to this scoring cache and training labels. They must
NEVER enter the feature encoder or deployment-time clustering. No waveform
can be synthesized from this small cache; it supports permutation scores only.
"""
import numpy as np
from spectral_routing import PERMS,WINDOW


def collect(output,reference,mixture):
    n=mixture.shape[-1];assert output.shape==(4,n) and reference.shape==(3,n)
    bounds=[(a,min(a+WINDOW,n)) for a in range(0,n,WINDOW)]
    sy=[];sr=[];ey=[];er=[];cross=[];sse=[];closure=[];em=[]
    for a,b in bounds:
        y=output[:3,a:b].astype(np.complex128);r=reference[:,a:b].astype(np.complex128)
        m=mixture[a:b].astype(np.complex128)
        sy.append(y.sum(-1));sr.append(r.sum(-1));ey.append(np.sum(abs(y)**2,-1));er.append(np.sum(abs(r)**2,-1))
        cross.append(y@r.conj().T)
        sse.append([[np.sum(abs(y[j]-r[q])**2) for q in range(3)] for j in range(3)])
        closure.append([np.sum(abs(output[[*p,3],a:b].sum(0)-m)**2) for p in PERMS])
        em.append(np.sum(abs(m)**2))
    return dict(bounds=np.asarray(bounds),sum_y=np.asarray(sy),sum_r=np.asarray(sr),
        energy_y=np.asarray(ey),energy_r=np.asarray(er),cross=np.asarray(cross),sse=np.asarray(sse),
        closure=np.asarray(closure),energy_m=np.asarray(em),length=np.asarray(n))


def score(cache,orders,count,predicted_count):
    orders=np.asarray(orders,dtype=np.int64);blocks=len(cache['bounds']);n=int(cache['length'])
    assert orders.shape==(blocks,3) and np.all(np.sort(orders,axis=-1)==np.arange(3))
    b=np.arange(blocks)[:,None]
    sy=cache['sum_y'][b,orders].sum(0);sr=cache['sum_r'].sum(0)
    py=cache['energy_y'][b,orders].sum(0)/n;pr=cache['energy_r'].sum(0)/n
    error=cache['sse'][b,orders].sum(0)/n
    cov=cache['cross'][b,orders].sum(0)/n-sy[:,None]*sr.conj()[None,:]/n**2
    vy=np.maximum(py-abs(sy/n)**2,0);vr=np.maximum(pr-abs(sr/n)**2,0)
    pm=max(float(cache['energy_m'].sum())/n,1e-8);active=np.arange(3)<count
    coherence=np.clip(abs(cov)**2/np.maximum(vy[:,None]*vr[None,:],1e-16),0,1)
    denominator=np.where(active,np.maximum(pr,1e-6*pm),pm)
    pair=error/denominator[None,:]+(1-coherence)*(active & (pr>1e-6*pm))[None,:]
    costs=np.array([pair[p,np.arange(3)].mean() for p in PERMS]);mapping=PERMS[int(costs.argmin())]
    nmse=[];si=[]
    for q in range(count):
        j=mapping[q];nmse.append(float(error[j,q]/pr[q]))
        projected=abs(cov[j,q])**2/vr[q] if vr[q]>0 else 0
        residual=vy[j]-projected
        value=10*np.log10(projected/residual) if projected>0 and residual>0 else np.nan
        si.append(float(value) if np.isfinite(value) else None)
    permutation_indices=np.array([PERMS.index(tuple(p)) for p in orders])
    closure=cache['closure'][np.arange(blocks),permutation_indices].sum()/n/pm
    return dict(count=count,nmse=nmse,si_sdr=si,reference_power=pr[:count].tolist(),
        weakest_index=int(pr[:count].argmin()),assignment=list(mapping),
        sum_relative_error=float(closure),predicted_count=int(predicted_count))


def training_labels(cache,count):
    """Local optimal source mapping; equal total weight per active source.

    Activity weights use each reference's energy fraction, including recorded
    receiver noise. They do not claim a clean-signal activity annotation.
    """
    labels=np.full((len(cache['bounds']),3),-1,np.int64)
    weights=np.zeros(labels.shape,np.float64)
    total=cache['energy_r'].sum(0)
    for i,(a,b) in enumerate(cache['bounds']):
        piece={k:(v[i:i+1] if k!='length' else np.asarray(b-a)) for k,v in cache.items()}
        piece['bounds']=np.asarray([[0,b-a]])
        result=score(piece,[[0,1,2]],count,count)
        for j in range(count):
            slot=result['assignment'][j];labels[i,slot]=j
            weights[i,slot]=cache['energy_r'][i,j]/total[j]/count
    np.testing.assert_allclose(weights.sum(),1.,rtol=1e-12,atol=1e-12)
    return labels,weights.astype(np.float32)
