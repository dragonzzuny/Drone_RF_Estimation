"""Exploratory TRAIN-only lag-product periodicity versus PSD-preserving surrogates.

Three surrogates are descriptive controls, not a significance test. No protocol
labels, lower bounds, source-separability or model-performance claims follow.
"""
import argparse
from pathlib import Path
from collections import defaultdict
import os
import time
import traceback
import numpy as np
from scipy import fft
import complex_lag_diagnosis as base

LAGS=(0,3334,6667,14270)


def peaks(x,fs):
    rows=[]
    for lag in LAGS:
        product=x*np.conj(x) if lag==0 else x[lag:]*np.conj(x[:-lag])
        product=product-product.mean();n=len(product)
        product*=np.hanning(n)
        energy=float(np.vdot(product,product).real)
        if energy<=0:raise ValueError('Constant lag product')
        spectrum=fft.fft(product,workers=2)
        frequencies=fft.fftfreq(n,1/fs)
        keep=np.flatnonzero((np.abs(frequencies)>=1000)&(np.abs(frequencies)<=100000))
        power=np.abs(spectrum[keep])**2/(n*energy)
        j=int(np.argmax(power));frequency=float(frequencies[keep[j]])
        rows.append(dict(lag_samples=lag,lag_us=lag/fs*1e6,peak_frequency_hz=frequency,
            peak_period_us=1e6/abs(frequency),peak_fraction_total_lag_product_energy=float(power[j]),
            bins_searched=len(keep),frequency_resolution_hz=fs/n))
    return rows


def checks():
    n=65536;fs=65536.;t=np.arange(n)/fs
    # A known periodic envelope has a component at 4096 Hz in |x|^2.
    x=(1+.5*np.cos(2*np.pi*4096*t))*np.exp(2j*np.pi*137*t)
    p=peaks(x,fs)[0]
    assert abs(abs(p['peak_frequency_hz'])-4096)<1e-9
    rng=np.random.default_rng(2);x=rng.normal(size=n)+1j*rng.normal(size=n)
    X=fft.fft(x,workers=2);surrogate=fft.ifft(np.abs(X)*np.exp(1j*rng.uniform(-np.pi,np.pi,n)),workers=2)
    error=float(np.max(np.abs(np.abs(fft.fft(surrogate,workers=2))-np.abs(X)))/np.max(np.abs(X)))
    assert error<1e-12
    return dict(status='PASS',known_envelope_frequency_hz=4096,relative_max_fft_magnitude_error=error)


def run(study,root,public):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate cyclic diagnosis')
    test=checks();p=base.json.loads((study/'PROTOCOL.json').read_text())
    manifest_path=Path(p['preparation'])/'NATIVE_MANIFEST.json';m=base.json.loads(manifest_path.read_text())
    groups=defaultdict(list)
    for c in m['clips']:
        if c['role']=='train_pack':groups[c['pack_id']].append(c)
    selected=[]
    for key,clips in sorted(groups.items()):
        clips.sort(key=lambda c:(c['source_path'],c['offset_samples']));selected.append(clips[len(clips)//2])
    assert len(selected)==8
    protocol=dict(status='REGISTERED_CPU_TRAIN_LAG_PRODUCT',source_sha256={str(q.relative_to(base.ROOT)):base.digest(q) for q in (Path(__file__),Path(base.__file__))},
        manifest_sha256=base.digest(manifest_path),study_protocol_sha256=base.digest(study/'PROTOCOL.json'),
        clips=[{k:c[k] for k in ('clip_id','pack_id','category','cache_sha256','samples','fs_hz')} for c in selected],
        selection='Metadata-ordered middle clip per existing TRAIN recording group; fixed before waveform reads',
        lags_samples=list(LAGS),frequency_search_hz=[1000,100000],surrogates_per_clip=3,seed=0,
        method='Whole-clip demean; x[t]*conj(x[t-lag]); remove product mean; Hann; two-sided FFT. Peak bin fraction of total windowed product energy.',
        surrogate='Randomize every complex FFT phase independently, preserving whole-record magnitude spectrum; finite linear-lag boundaries are not preserved exactly',
        limitations='Descriptive three-surrogate comparison; no p-value, multiple-comparison inference, protocol label, independence claim or separation result',
        model_updates=0,heldout_read=False,validation_read=False,gpu_use=False,checks=test,registered_at=time.time())
    base.write(root/'PROTOCOL.json',protocol);rng=np.random.default_rng(0);rows=[]
    for c in selected:
        path=Path(c['cache_path'])
        if base.digest(path)!=c['cache_sha256']:raise ValueError('Changed TRAIN cache')
        x=np.array(np.load(path,mmap_mode='r',allow_pickle=False),dtype=np.complex128,copy=True)
        assert x.shape==(c['samples'],) and c['fs_hz']==100000000
        x-=x.mean();X=fft.fft(x,workers=2);magnitude=np.abs(X)
        actual=peaks(x,c['fs_hz']);controls=[];errors=[]
        for _ in range(3):
            y=fft.ifft(magnitude*np.exp(1j*rng.uniform(-np.pi,np.pi,len(x))),workers=2)
            error=float(np.max(np.abs(np.abs(fft.fft(y,workers=2))-magnitude))/np.max(magnitude))
            if error>1e-12:raise ValueError('Surrogate magnitude mismatch')
            errors.append(error);controls.append(peaks(y,c['fs_hz']));del y
        rows.append(dict(clip_id=c['clip_id'],pack_id=c['pack_id'],category=c['category'],actual=actual,surrogates=controls,
            relative_max_fft_magnitude_errors=errors))
        base.write(root/'STATE.json',dict(status='CPU_TRAIN_LAG_PRODUCT',clips=len(rows),total=8,pid=os.getpid(),time=time.time()))
        del x,X,magnitude
    for rel,sha in protocol['source_sha256'].items():
        if base.digest(base.ROOT/rel)!=sha:raise ValueError('Frozen source changed')
    if base.digest(manifest_path)!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    result=dict(status='COMPLETE',protocol_sha256=base.digest(root/'PROTOCOL.json'),rows=rows,checks=test,
        model_updates=0,heldout_read=False,validation_read=False,interpretation=protocol['limitations'])
    base.write(root/'COMPLETE.json',result);base.write(public,result)
    base.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',training_groups=8,surrogates_per_group=3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('study','run','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:run(a.study.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        base.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
