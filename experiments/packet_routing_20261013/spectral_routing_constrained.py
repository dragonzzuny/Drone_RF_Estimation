"""One objective for block-wise injective assignment and spherical centers.

No references or category dictionaries. Unlike free k-means followed by a
different assignment rule, both updates maximize the same weighted cosine sum.
"""
import numpy as np
from spectral_routing import PERMS,profiles


def fit(vectors,powers,count):
    assert count in (1,2,3) and vectors.shape==(*powers.shape,512)
    assert powers.shape[1]==3 and np.all(powers>=0) and powers.sum()>0
    w=powers/powers.sum();initial=np.argsort(-powers.sum(0),kind='stable')[:count]
    centers=np.stack([(vectors[:,j]*powers[:,j,None]).sum(0) for j in initial])
    norm=np.linalg.norm(centers,axis=1,keepdims=True);centers/=np.where(norm>0,norm,1.)
    previous=None;trace=[];updates=0
    def assign():
        similarity=vectors@centers.T
        values=np.array([[sum(w[b,p[j]]*similarity[b,p[j],j] for j in range(count))
                          for p in PERMS] for b in range(len(vectors))])
        choices=values.argmax(1);value=float(values[np.arange(len(vectors)),choices].sum())
        return choices,value
    for _ in range(20):
        choices,value=assign();trace.append(value)
        if previous is not None and np.array_equal(choices,previous):break
        previous=choices.copy()
        for j in range(count):
            slots=np.array([PERMS[q][j] for q in choices]);indices=np.arange(len(vectors))
            c=(vectors[indices,slots]*w[indices,slots,None]).sum(0);norm=np.linalg.norm(c)
            if norm>0:centers[j]=c/norm
        updates+=1
    choices,value=assign();trace.append(value)
    assert np.min(np.diff(trace))>=-1e-12,trace
    return centers,choices,dict(initial_output_slots=initial.tolist(),centroid_updates=updates,
        weighted_cosine_objective=trace,monotone_objective=True)


def route(outputs,predicted_count):
    vectors,powers,bounds=profiles(outputs);centers,choices,metadata=fit(vectors,powers,predicted_count)
    routed=np.empty_like(outputs);orders=[]
    for i,(a,b) in enumerate(bounds):
        p=PERMS[int(choices[i])];routed[:,a:b]=outputs[[*p,3],a:b]
        assert np.array_equal(routed[:,a:b],outputs[[*p,3],a:b])
        orders.append(dict(start=a,stop=b,permutation=list(p),output_power=powers[i].tolist()))
    assert np.array_equal(routed[3],outputs[3])
    return routed,dict(metadata,method='constrained_slot_clusters',predicted_count=predicted_count,
        centers=centers.tolist(),orders=orders,reference_data_used=False,category_dictionary_used=False,
        note='Same PSD/block/power/initial slot centers asv2; injective assignment used within every centroid update')
