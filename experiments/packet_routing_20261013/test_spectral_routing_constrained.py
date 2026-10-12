"""Mathematical and amplitude-imbalance synthetic checks, no RF score tuning."""
from pathlib import Path
import numpy as np
from spectral_routing_constrained import fit,route
from spectral_routing import PERMS,WINDOW
from evaluate_parent_full_context import ROOT,write,sha


def run():
    rng=np.random.default_rng(0);random_checks=[]
    for count in (1,2,3):
        for trial in range(12):
            x=rng.uniform(size=(33,3,512));x/=np.linalg.norm(x,axis=-1,keepdims=True)
            p=10**rng.uniform(-4,2,size=(33,3));centers,choices,meta=fit(x,p,count)
            assert all(len(set(PERMS[q][:count]))==count for q in choices)
            assert meta['monotone_objective'];random_checks.append(dict(count=count,trial=trial,
                steps=meta['centroid_updates'],gain=meta['weighted_cosine_objective'][-1]-meta['weighted_cosine_objective'][0]))
    waveform=[]
    t=np.arange(3*WINDOW);base=np.stack([np.exp(2j*np.pi*f*t) for f in (.02,.13,-.22)]).astype(np.complex64)
    for count in (1,2,3):
        y=np.zeros((4,len(t)),np.complex64)
        for b in range(3):
            order=PERMS[b];part=np.zeros((3,WINDOW),np.complex64)
            for j in range(count):part[j]=base[j,b*WINDOW:(b+1)*WINDOW]*10**(-j/2)
            y[:3,b*WINDOW:(b+1)*WINDOW]=part[list(order)]
        out,meta=route(y,count)
        assert np.array_equal(out[3],y[3])
        np.testing.assert_allclose(out.sum(0),y.sum(0),atol=3e-7,rtol=3e-7)
        assert all(sorted(v['permutation'])==[0,1,2] for v in meta['orders'])
        waveform.append(dict(count=count,all_complex_samples_permuted_only=True,monotone_objective=meta['monotone_objective']))
    result=dict(status='PASS',random_objective_cases=random_checks,unequal_power_waveforms=waveform,
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('spectral_routing_constrained.py'))},
        rf_data_used=False,performance_claim=False)
    write(ROOT/'reports/2026-10-13/CONSTRAINED_ROUTING_CPU_CHECK.json',result);print('PASS',len(random_checks),waveform)


if __name__=='__main__':run()
