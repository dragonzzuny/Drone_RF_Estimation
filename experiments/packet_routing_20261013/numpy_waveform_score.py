"""Bounded-memory double-precision sufficient statistics for complex metrics.

Whole-observation PIT retains NMSE plus squared-coherence and inactive energy.
The metric convention is the frozen waveform.py convention, not real-audio SDR.
Double-precision assignment can differ from FP32 at numerical ties; callers
must verify active assignments and numerical metrics against saved baselines.
"""
import itertools
import numpy as np

PERMS=list(itertools.permutations(range(3)))


def score(output,reference,mixture,count,predicted_count,chunk=65536):
    n=mixture.shape[-1]
    assert output.shape==(4,n) and reference.shape==(3,n) and count in (1,2,3)
    sy=np.zeros(3,np.complex128);sr=sy.copy()
    ey=np.zeros(3);er=ey.copy();cross=np.zeros((3,3),np.complex128)
    error=np.zeros((3,3));em=0.;sum_error=0.
    for start in range(0,n,chunk):
        end=min(start+chunk,n)
        y=output[:3,start:end].astype(np.complex128)
        r=reference[:,start:end].astype(np.complex128)
        m=mixture[start:end].astype(np.complex128)
        assert np.isfinite(y).all() and np.isfinite(r).all() and np.isfinite(m).all()
        sy+=y.sum(-1);sr+=r.sum(-1)
        ey+=np.sum(abs(y)**2,-1);er+=np.sum(abs(r)**2,-1)
        cross+=y@r.conj().T
        for j in range(3):
            for q in range(3):error[j,q]+=np.sum(abs(y[j]-r[q])**2)
        em+=float(np.sum(abs(m)**2))
        # Frozen waveform_metrics sums the complex64 outputs before subtracting
        # its complex128 mixture. Preserve that detail for closure verification.
        sum_error+=float(np.sum(abs(output[:,start:end].sum(0)-m)**2))
    py,pr,pm=ey/n,er/n,max(em/n,1e-8)
    cov=cross/n-sy[:,None]*sr.conj()[None,:]/n**2
    vy=np.maximum(py-abs(sy/n)**2,0)
    vr=np.maximum(pr-abs(sr/n)**2,0)
    coh=np.clip(abs(cov)**2/np.maximum(vy[:,None]*vr[None,:],1e-16),0,1)
    active=np.arange(3)<count
    denominator=np.where(active,np.maximum(pr,1e-6*pm),pm)
    cost=error/n/denominator[None,:]+(1-coh)*(active & (pr>1e-6*pm))[None,:]
    costs=np.array([np.mean(cost[p,np.arange(3)]) for p in PERMS])
    order=PERMS[int(costs.argmin())]
    nmse=[];si=[]
    for q in range(count):
        j=order[q];nmse.append(float(error[j,q]/er[q]))
        projected=abs(cov[j,q])**2/vr[q] if vr[q]>0 else 0
        residual=vy[j]-projected
        value=10*np.log10(projected/residual) if projected>0 and residual>0 else np.nan
        si.append(float(value) if np.isfinite(value) else None)
    return dict(count=count,nmse=nmse,si_sdr=si,reference_power=pr[:count].tolist(),
        weakest_index=int(pr[:count].argmin()),assignment=list(order),
        sum_relative_error=float(sum_error/n/pm),predicted_count=int(predicted_count))
