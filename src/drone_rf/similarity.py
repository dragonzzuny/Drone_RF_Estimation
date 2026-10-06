"""CPU waveform similarity diagnostic, not a proof of recording independence."""
import numpy as np


def shifted_coherence(x, y, max_lag=1024):
    """Largest squared complex correlation with overlap-wise mean removal.

    Positive lag compares x[lag:] to y[:-lag]. Bounds keep at least 75% overlap.
    A high value may reflect repeated protocol structure; it is not a copy verdict.
    """
    x=np.asarray(x,dtype=np.complex128);y=np.asarray(y,dtype=np.complex128)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<4:
        raise ValueError('Equal one-dimensional complex windows required')
    n=len(x)
    if not 0<=max_lag<=n//4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid lag bound or nonfinite waveform')
    size=1<<(2*n-2).bit_length()
    correlation=np.fft.ifft(np.fft.fft(x,size)*np.fft.fft(y,size).conj())
    lags=np.arange(-max_lag,max_lag+1)
    lo_x=np.maximum(lags,0);hi_x=n+np.minimum(lags,0)
    lo_y=np.maximum(-lags,0);hi_y=n-np.maximum(lags,0)
    lengths=hi_x-lo_x
    sx=np.concatenate(([0j],np.cumsum(x)));sy=np.concatenate(([0j],np.cumsum(y)))
    ex=np.concatenate(([0.],np.cumsum(np.abs(x)**2)))
    ey=np.concatenate(([0.],np.cumsum(np.abs(y)**2)))
    sumx=sx[hi_x]-sx[lo_x];sumy=sy[hi_y]-sy[lo_y]
    numerator=correlation[lags%size]-sumx*sumy.conj()/lengths
    vx=np.maximum(ex[hi_x]-ex[lo_x]-np.abs(sumx)**2/lengths,0)
    vy=np.maximum(ey[hi_y]-ey[lo_y]-np.abs(sumy)**2/lengths,0)
    denominator=vx*vy
    valid=denominator>np.finfo(float).eps*np.maximum((ex[-1]*ey[-1]),np.finfo(float).tiny)
    score=np.zeros(len(lags))
    np.divide(np.abs(numerator)**2,denominator,out=score,where=valid)
    score=np.clip(score,0,1)
    best=int(np.argmax(score))
    return dict(rho_squared=float(score[best]),lag_samples=int(lags[best]),
                overlap_samples=int(lengths[best]),variance_defined=bool(valid[best]))
