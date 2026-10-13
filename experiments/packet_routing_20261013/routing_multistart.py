"""Fixed observed-only restarts for the same constrained cosine objective."""
import numpy as np
from spectral_routing import PERMS
from spectral_routing_constrained import fit as original_fit


def seeded_fit(vectors,powers,count,first):
    x=vectors.reshape(-1,512);weight=powers.reshape(-1)
    chosen=[int(first)]
    for _ in range(1,count):
        similarity=x@x[chosen].T
        distance=np.maximum(1-similarity.max(1),0)
        value=distance*np.sqrt(weight/weight.max())
        value[chosen]=-np.inf
        chosen.append(int(value.argmax()))
    centers=x[chosen].copy()
    norms=np.linalg.norm(centers,axis=-1,keepdims=True)
    centers/=np.where(norms>0,norms,1.)
    w=powers/powers.sum();previous=None;trace=[];updates=0
    def assign():
        similarity=vectors@centers.T
        values=np.array([[sum(w[b,p[j]]*similarity[b,p[j],j] for j in range(count))
                          for p in PERMS] for b in range(len(vectors))])
        choices=values.argmax(1)
        return choices,float(values[np.arange(len(vectors)),choices].sum())
    for _ in range(20):
        choices,value=assign();trace.append(value)
        if previous is not None and np.array_equal(choices,previous):break
        previous=choices.copy()
        for j in range(count):
            slots=np.array([PERMS[q][j] for q in choices]);index=np.arange(len(vectors))
            c=(vectors[index,slots]*w[index,slots,None]).sum(0);norm=np.linalg.norm(c)
            if norm>0:centers[j]=c/norm
        updates+=1
    choices,value=assign();trace.append(value)
    assert np.min(np.diff(trace))>=-1e-12
    return centers,choices,dict(initial_flat_tokens=chosen,centroid_updates=updates,
        weighted_cosine_objective=trace,monotone_objective=True)


def fit(vectors,powers,count):
    assert vectors.shape==(*powers.shape,512) and count in (1,2,3)
    base=original_fit(vectors,powers,count)
    first=[int(np.argmax(powers[:,j])*3+j) for j in range(3)]
    trials=[base]+[seeded_fit(vectors,powers,count,i) for i in first]
    objectives=[t[2]['weighted_cosine_objective'][-1] for t in trials]
    best=int(np.argmax(objectives))
    assert objectives[best]>=objectives[0]-1e-12
    center,choice,receipt=trials[best]
    return center,choice,dict(receipt,selected_start=best,start_objectives=objectives,
        original_objective_preserved=True,reference_used=False)
