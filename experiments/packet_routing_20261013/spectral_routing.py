"""Reference-free block routing, preserving every estimated complex sample.

This changes output identity only, not component values. Known-category FIT
templates and observation-only weighted spherical clustering are separate arms.
"""
import itertools
import numpy as np
from scipy import fft

PERMS=list(itertools.permutations(range(3)))
WINDOW=63872


def profiles(outputs):
    assert outputs.shape[0]==4 and np.isfinite(outputs).all()
    bounds=[(a,min(a+WINDOW,outputs.shape[-1])) for a in range(0,outputs.shape[-1],WINDOW)]
    window=np.sqrt(.5-.5*np.cos(2*np.pi*np.arange(512)/512)).astype(np.float32)
    embedding=[];power=[]
    for a,b in bounds:
        z=outputs[:3,a:b];length=z.shape[-1]//512*512
        spec=fft.fft(z[:,:length].reshape(3,-1,512)*window,axis=-1,workers=1)
        p=(spec.real.astype(np.float64)**2+spec.imag.astype(np.float64)**2).mean(1)
        norm=np.linalg.norm(p,axis=1,keepdims=True)
        embedding.append(p/np.where(norm>0,norm,1.))
        power.append(np.mean(abs(z.astype(np.complex128))**2,axis=1))
    return np.array(embedding),np.array(power),bounds


def spherical_centers(vectors,weights,count):
    assert count in (1,2,3)
    x=vectors.reshape(-1,512);w=weights.reshape(-1)
    assert np.isfinite(x).all() and np.isfinite(w).all() and w.sum()>0
    w=w/w.sum();chosen=[int(w.argmax())]
    while len(chosen)<count:
        distance=np.maximum(0,1-(x@x[chosen].T).max(1))
        score=w*distance**2;score[chosen]=-1
        chosen.append(int(score.argmax()))
    centers=x[chosen].copy();previous=None;iterations=0
    for _ in range(20):
        labels=(x@centers.T).argmax(1)
        if previous is not None and np.array_equal(labels,previous):break
        previous=labels.copy();iterations+=1
        for j in range(count):
            selected=labels==j
            if not selected.any() or w[selected].sum()==0:continue
            value=(x[selected]*w[selected,None]).sum(0);norm=np.linalg.norm(value)
            if norm>0:centers[j]=value/norm
    return centers,dict(initial_indices=chosen,iterations=iterations,
        cluster_weight=[float(w[previous==j].sum()) for j in range(count)])


def route(outputs,bank,method,predicted_count):
    vectors,powers,bounds=profiles(outputs)
    if method=='fit_dictionary':
        centers=np.asarray(bank['psd'],np.float64);centers/=np.linalg.norm(centers,axis=1,keepdims=True)
        metadata=dict(method=method,template_categories=bank['categories'],reference_data_used=False)
    elif method=='observation_clusters':
        centers,metadata=spherical_centers(vectors,powers,predicted_count)
        metadata.update(method=method,predicted_count=predicted_count,reference_data_used=False)
    else:raise ValueError(method)
    assert len(centers)<=3
    routed=np.empty_like(outputs);orders=[]
    for i,(a,b) in enumerate(bounds):
        similarity=vectors[i]@centers.T
        scores=[]
        for p in PERMS:
            scores.append(sum(powers[i,p[j]]*similarity[p[j],j] for j in range(len(centers))))
        order=PERMS[int(np.argmax(scores))]
        routed[:,a:b]=outputs[[*order,3],a:b]
        assert np.array_equal(routed[:,a:b],outputs[[*order,3],a:b])
        orders.append(dict(start=a,stop=b,permutation=list(order),scores=scores,
            output_power=powers[i].tolist()))
    assert np.array_equal(routed[3],outputs[3])
    return routed,dict(metadata,orders=orders,centers=centers.tolist(),
        note='Nonoverlapping0.63872ms hard permutations; final shorter block retained; no smoothing or phase/gain fitting')
