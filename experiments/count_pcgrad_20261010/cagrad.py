"""CAGrad Eq.(3), Liu et al. NeurIPS 2021, on sample-weighted count tasks.

g0=sum(rows) preserves the original batch mean. Equivalently task gradients
in the original K-task formulation are K*rows. Their common scaling changes
neither optimal weights nor the normalized conflict-averse direction.
No post hoc 1/(1+c^2) rescaling: use the paper's unrescaled trust-ball update.
"""
import numpy as np
from scipy.optimize import minimize
import torch


def solve(gram, c=.5):
    a=np.asarray(gram,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all() or not 0<=c<1:
        raise ValueError('Invalid Gram matrix or c')
    a=(a+a.T)/2
    scale=max(float(np.max(np.diag(a))),1e-30)
    a=a/scale
    if np.linalg.eigvalsh(a).min() < -1e-5:
        raise ValueError('Gram matrix not positive semidefinite')
    k=len(a);one=np.ones(k);base=a@one;g0norm=np.sqrt(max(float(one@base),0.))
    uniform=one/k
    if c==0 or g0norm<1e-14:
        return dict(weights=uniform.tolist(),coefficients=one.tolist(),c=c,
                    solver_success=True,solver_iterations=0,kkt_gap=0.,branch='ordinary_or_zero')
    radius=c*g0norm
    # Smooth sqrt at a tiny, explicitly reported normalized Gram scale. This
    # prevents division by zero at opposing-gradient degeneracies; no fake
    # claim of exact nonsmooth optimality in that case.
    epsilon=1e-20
    def objective(w):
        return float(w@base+radius*np.sqrt(max(float(w@a@w),0.)+epsilon))
    def jac(w):
        return base+radius*(a@w)/np.sqrt(max(float(w@a@w),0.)+epsilon)
    fit=minimize(objective,uniform,jac=jac,method='SLSQP',bounds=[(0.,1.)]*k,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:one)],
        options=dict(ftol=1e-12,maxiter=200))
    weights=fit.x
    if not fit.success or weights.min() < -1e-8 or abs(weights.sum()-1)>1e-8:
        raise ValueError('CAGrad dual solve failed: '+fit.message)
    weights=np.maximum(weights,0.);weights/=weights.sum()
    gradient=jac(weights);gap=float(weights@gradient-gradient.min())
    if gap>1e-5:
        raise ValueError('Inaccurate simplex solution: '+str(gap))
    weighted_norm=np.sqrt(max(float(weights@a@weights),0.)+epsilon)
    coefficients=one+radius/weighted_norm*weights
    return dict(weights=weights.tolist(),coefficients=coefficients.tolist(),c=c,
        solver_success=True,solver_iterations=int(fit.nit),kkt_gap=gap,
        dual_objective_normalized=objective(weights),gram_scale=scale,
        normalized_sqrt_epsilon=epsilon,branch='dual')


def surgery(groups,rng=None):
    if groups.ndim!=2 or not torch.isfinite(groups).all():raise ValueError('Invalid gradients')
    gram=torch.stack([torch.stack([torch.dot(a,b) for b in groups]) for a in groups])
    receipt=solve(gram.cpu().double().numpy(),.5)
    coefficients=groups.new_tensor(receipt['coefficients'])
    # Avoid a full K-by-P multiplication temporary.
    result=torch.zeros_like(groups[0])
    for coefficient,row in zip(coefficients,groups):result.add_(coefficient*row)
    ordinary=groups.sum(0)
    stats=dict(gram=gram.tolist(),cagrad=receipt,ordinary_norm=float(ordinary.norm()),
        projected_norm=float(result.norm()),change_norm=float((result-ordinary).norm()),
        projected_direction_dot_original_tasks=[float(torch.dot(result,g)) for g in groups])
    if not torch.isfinite(result).all():raise ValueError('Nonfinite CAGrad direction')
    if stats['change_norm']>.5001*stats['ordinary_norm']+1e-5:
        raise ValueError('Trust-ball constraint violated')
    return result,stats
