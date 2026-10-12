"""Two fixed hypotheses after v1: preserve slot priors; remove absent templates."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from spectral_routing import profiles,PERMS


def centers_from_slots(vectors,powers,count):
    x=vectors.reshape(-1,512);weights=powers.reshape(-1);weights=weights/weights.sum()
    initial=np.argsort(-powers.sum(0),kind='stable')[:count]
    centers=np.stack([(vectors[:,j]*powers[:,j,None]).sum(0) for j in initial])
    norms=np.linalg.norm(centers,axis=1,keepdims=True);centers/=np.where(norms>0,norms,1.)
    previous=None;iterations=0
    for _ in range(20):
        labels=(x@centers.T).argmax(1)
        if previous is not None and np.array_equal(labels,previous):break
        previous=labels.copy();iterations+=1
        for j in range(count):
            chosen=labels==j
            if not chosen.any() or weights[chosen].sum()==0:continue
            c=(x[chosen]*weights[chosen,None]).sum(0);norm=np.linalg.norm(c)
            if norm>0:centers[j]=c/norm
    return centers,dict(initial_output_slots=initial.tolist(),iterations=iterations,
        cluster_weight=[float(weights[previous==j].sum()) for j in range(count)])


def route(outputs,bank,method,predicted_count,observed):
    assert predicted_count in (1,2,3) and observed.shape==(outputs.shape[-1],)
    vectors,powers,bounds=profiles(outputs)
    if method=='count_dictionary':
        root=Path(__file__).resolve().parents[2]
        sys.path.insert(0,str(root/'experiments/rfuav_native_frequency_20261009'))
        from spectral_baseline import profile
        dictionary=np.asarray(bank['psd'],np.float64).T
        p=profile(observed);p=p/p.sum();coefficients,residual=nnls(dictionary,p)
        count=min(predicted_count,dictionary.shape[1])
        selected=np.argsort(-coefficients,kind='stable')[:count]
        centers=dictionary[:,selected].T.copy();centers/=np.linalg.norm(centers,axis=1,keepdims=True)
        metadata=dict(method=method,selected_categories=[bank['categories'][j] for j in selected],
            fitted_coefficients=coefficients.tolist(),nnls_residual=float(residual),predicted_count=predicted_count,
            dictionary_size_caps_count=True)
    elif method=='slot_initialized_clusters':
        centers,metadata=centers_from_slots(vectors,powers,predicted_count)
        metadata.update(method=method,predicted_count=predicted_count)
    else:raise ValueError(method)
    routed=np.empty_like(outputs);orders=[]
    for i,(a,b) in enumerate(bounds):
        similarity=vectors[i]@centers.T
        scores=[sum(powers[i,p[j]]*similarity[p[j],j] for j in range(len(centers))) for p in PERMS]
        order=PERMS[int(np.argmax(scores))];routed[:,a:b]=outputs[[*order,3],a:b]
        assert np.array_equal(routed[:,a:b],outputs[[*order,3],a:b])
        orders.append(dict(start=a,stop=b,permutation=list(order),scores=scores,output_power=powers[i].tolist()))
    assert np.array_equal(routed[3],outputs[3])
    return routed,dict(metadata,orders=orders,centers=centers.tolist(),reference_data_used=False,
        note='Same block length, energy-weighted cosine assignment and20iteration cap asv1; no confidence threshold sweep')
