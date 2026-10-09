"""CPU contract checks, including an independent simplex-grid dual comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import cagrad
from check import run as check_accumulation


def run():
    accumulation=check_accumulation()
    cases=[np.eye(3),np.array([[1.,2.,-3.],[-2.,1.,1.],[1.,-2.,1.]]),
           np.ones((3,5)),np.zeros((3,4)),np.array([[1.,0.],[-1.,0.],[0.,0.]])]
    rng=np.random.default_rng(0);cases.extend(rng.normal(size=(3,13)) for _ in range(30))
    errors=[]
    for g in cases:
        a=g@g.T;solution=cagrad.solve(a);coef=np.array(solution['coefficients']);d=coef@g;g0=g.sum(0)
        assert np.linalg.norm(d-g0)<=.50001*np.linalg.norm(g0)+1e-9
        assert np.isfinite(d).all()
        assert np.allclose(np.array(cagrad.solve(a,0)['coefficients'])@g,g0)
        actual,receipt=cagrad.surgery(torch.tensor(g,dtype=torch.float64))
        assert np.allclose(actual.numpy(),d,atol=1e-10,rtol=1e-10)
        if solution['branch']=='dual':
            weights=np.array(solution['weights']);base=a@np.ones(3);rad=.5*np.linalg.norm(g0)
            def f(w):return w@base+rad*np.sqrt(max(w@a@w,0.))
            # Coarse enumeration is independent of SLSQP; optimum cannot be worse.
            grid=min(f(np.array([i/30,j/30,1-(i+j)/30])) for i in range(31) for j in range(31-i))
            gap=f(weights)-grid;assert gap<1e-7;errors.append(float(gap))
    here=Path(__file__).resolve().parent;root=here.parents[1]
    hashes=dict(accumulation['source_sha256'])
    for p in (here/'cagrad.py',Path(__file__)):hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(status='PASS',cases=len(cases),c_zero_recovers_original_mean=True,
        trust_ball_checked=True,dual_no_worse_than_independent_grid=True,
        maximum_grid_objective_excess=max(errors),accumulation_check=accumulation,
        source_sha256=hashes,gpu_use=False,recorded_iq_reads=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
