"""Two-halfspace Euclidean projection of a proposed parameter displacement.

GEM-inspired geometry, adapted here to the ACTUAL AdamW displacement. This is
only a mathematical diagnostic until a separate training protocol invokes it.
For protected gradients H, require H @ delta_new <= 0 (parameter displacement,
not a positive descent-gradient vector). Use the 2x2 Gram and H @ delta only.
"""
import itertools
import numpy as np


def solve(gram, dots):
    a=np.asarray(gram,dtype=np.float64);q=np.asarray(dots,dtype=np.float64)
    if a.shape!=(2,2) or q.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Expected finite two-task Gram and directional products')
    a=(a+a.T)/2
    # Normalize constraints by their own gradient norms without changing the
    # feasible halfspaces. Zero gradients imply zero true directional products.
    norms=np.sqrt(np.maximum(np.diag(a),0.))
    if np.any((norms==0)&(np.abs(q)>1e-12)):raise ValueError('Inconsistent zero-gradient dot')
    scales=np.where(norms>0,norms,1.)
    b=a/(scales[:,None]*scales[None,:]);v=q/scales
    if np.linalg.eigvalsh(b).min() < -1e-5:raise ValueError('Indefinite Gram')
    tolerance=1e-9*max(float(np.linalg.norm(v)),1e-12)
    feasible=[]
    for n in range(3):
        for active in itertools.combinations(range(2),n):
            coefficients=np.zeros(2)
            if active:
                idx=list(active);matrix=b[np.ix_(idx,idx)];rhs=v[idx]
                value=np.linalg.lstsq(matrix,rhs,rcond=1e-10)[0]
                if np.max(np.abs(matrix@value-rhs))>tolerance:continue
                coefficients[idx]=value
            residual=v-b@coefficients
            if coefficients.min() < -tolerance or residual.max()>tolerance:continue
            cost=max(float(coefficients@b@coefficients),0.)
            feasible.append((cost,coefficients,active))
    if not feasible:raise ValueError('No numerically feasible projection')
    cost,coef,active=min(feasible,key=lambda x:x[0])
    alpha=coef/scales
    return dict(coefficients=alpha.tolist(),active_constraints=list(active),
                correction_squared_norm=cost,protected_dot_after=(q-a@alpha).tolist(),
                protected_dot_before=q.tolist(),constraint_tolerance=tolerance)


def check():
    from scipy.optimize import minimize
    rng=np.random.default_rng(0);cases=0;maximum_error=0.
    pairs=[np.eye(2),np.array([[1.,0.],[-1.,0.]]),np.array([[1.,0.],[1.,0.]]),np.zeros((2,2))]
    pairs.extend(rng.normal(size=(2,7)) for _ in range(40))
    for h in pairs:
        delta=rng.normal(size=h.shape[1]);a=h@h.T;q=h@delta
        solution=solve(a,q);actual=delta-h.T@np.array(solution['coefficients'])
        assert (h@actual).max()<1e-8
        opt=minimize(lambda d:.5*np.sum((d-delta)**2),np.zeros_like(delta),jac=lambda d:d-delta,
            method='SLSQP',constraints=[dict(type='ineq',fun=lambda d:-h@d,jac=lambda d:-h)],
            options=dict(ftol=1e-12,maxiter=200))
        assert opt.success
        error=float(np.linalg.norm(actual-opt.x));assert error<1e-6;maximum_error=max(maximum_error,error)
        assert np.isclose(np.sum((actual-delta)**2),solution['correction_squared_norm'])
        cases+=1
    return dict(status='PASS',cases=cases,maximum_error_vs_independent_primal_solver=maximum_error,
                gpu_use=False,recorded_iq_reads=0,training_implemented=False)


if __name__=='__main__':
    import argparse,json,hashlib
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=check();value['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(value,indent=2)+'\n');print(value)
