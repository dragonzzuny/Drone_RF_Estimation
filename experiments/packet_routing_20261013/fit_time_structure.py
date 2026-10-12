"""Descriptive time scales from fixed TRAIN clips, no new model or DEV selection."""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'local/bandwidth_evaluation_20261011_v1/TIME_BLOCK_MANIFEST.json'
OUT=ROOT/'local/time_structure_fit_20261013_v1'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def correlation(x,max_lag):
    """Global centering, lag-overlap cosine normalization; no circular wrap."""
    x=np.asarray(x,dtype=np.complex128);x=x-x.mean();n=len(x)
    assert 0<max_lag<n
    size=1 << (2*n-2).bit_length()
    f=np.fft.fft(x,n=size)
    cross=np.fft.ifft(f*f.conj())[:max_lag+1]
    energy=np.r_[0.,np.cumsum(abs(x)**2)]
    lags=np.arange(max_lag+1)
    denominator=np.sqrt((energy[n]-energy[lags])*energy[n-lags])
    if energy[-1]<=1e-30:raise ValueError('Degenerate centered waveform; do not invent a correlation')
    result=cross/np.maximum(denominator,1e-30)
    assert abs(result[0]-1)<1e-10 and np.max(abs(result))<=1+1e-9
    return result


def peaks(curve,minimum,distance,step_ms):
    indices,_=find_peaks(curve,distance=distance)
    indices=indices[indices>=minimum]
    chosen=sorted(indices,key=lambda i:float(curve[i]),reverse=True)[:3]
    return [dict(lag_ms=float(i*step_ms),correlation=float(curve[i])) for i in chosen]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path(__file__);manifest=read(MANIFEST)
    categories=sorted({c['category'] for c in manifest['clips'] if c['diagnostic_role']=='fit_block'})
    assert len(categories)==5
    selected=[]
    for category in categories:
        clips=sorted([c for c in manifest['clips'] if c['category']==category and c['diagnostic_role']=='fit_block'],
                     key=lambda c:(c['raw_support_start_samples'],c['clip_id']))
        for rank in (0,len(clips)//2,len(clips)-1):selected.append(clips[rank])
    plan=dict(source_sha256=sha(source),manifest_sha256=sha(MANIFEST),selection='First,middle,last admitted fit context per category, ordered by raw start',
        selected=selected,complex_lag_max_ms=2.,complex_peak_min_ms=.05,complex_peak_spacing_ms=.05,
        envelope_block_samples=1024,envelope_lag_max_ms=8.,envelope_peak_min_ms=.1,envelope_peak_spacing_ms=.1,
        peak_count=3,peak_statistical_significance=False,heldout_access=False,dev_access=False)
    path=OUT/'PLAN.json'
    if path.exists():assert read(path)==plan
    else:write(path,plan)
    # Verify normalization against a direct complex inner product before reading RF.
    rng=np.random.default_rng(0);synthetic=rng.normal(size=137)+1j*rng.normal(size=137);synthetic-=synthetic.mean()
    check=correlation(synthetic,45)
    for lag in (0,1,17,45):
        a,b=synthetic[lag:],synthetic[:len(synthetic)-lag]
        expected=np.sum(a*b.conj())/np.sqrt(np.sum(abs(a)**2)*np.sum(abs(b)**2))
        np.testing.assert_allclose(check[lag],expected,rtol=1e-12,atol=1e-12)
    rows=[];began=time.time()
    for i,c in enumerate(selected):
        assert c['diagnostic_role']=='fit_block' and c['fs_hz']==100000000
        file=Path(c['cache_path']);assert sha(file)==c['cache_sha256'];x=np.load(file,allow_pickle=False)
        assert x.shape==(2088960,) and np.isfinite(x).all()
        acf=correlation(x,200000);magnitude=abs(acf)
        block=1024;n=(len(x)//block)*block
        power=(abs(x[:n].astype(np.complex128))**2).reshape(-1,block).mean(-1)
        envelope=correlation(power,int(8./.01024)).real
        p=OUT/f'{i:02d}_{c["clip_id"]}.npz'
        np.savez_compressed(p,complex_acf=acf.astype(np.complex64),power_envelope_acf=envelope.astype(np.float32))
        row=dict(category=c['category'],clip_id=c['clip_id'],role='fit_block',context_ms=len(x)/100000.,
            source_start_seconds=c['raw_support_start_samples']/c['fs_hz'],
            waveform_peaks=peaks(magnitude,5000,5000,.00001),
            envelope_peaks=peaks(envelope,10,10,.01024),
            envelope_coefficient_of_variation=float(power.std()/power.mean()),
            complex_corr_at_window=float(magnitude[63872]),complex_corr_at_three_windows=float(magnitude[191616]),
            cache_sha256=c['cache_sha256'],curves_sha256=sha(p))
        rows.append(row)
        write(OUT/'STATE.json',dict(status='FIT_CORRELATION',completed=i+1,total=len(selected),seconds=time.time()-began))
        del x,acf,magnitude,power,envelope;gc.collect()
    assert sha(source)==plan['source_sha256'] and sha(MANIFEST)==plan['manifest_sha256']
    result=dict(status='VERIFIED',rows=rows,plan_sha256=sha(path),seconds=time.time()-began,
        caveat='15 chronological TRAIN contexts from five admitted recording groups. Peaks are descriptive, not significance tests or proof of source-discriminating periodicity; 20.89ms cannot establish long repetition periods.',
        synthetic_direct_correlation_check='PASS',new_model_training=False,dev_access=False,heldout_access=False)
    write(OUT/'RESULT.json',result);write(ROOT/'reports/2026-10-13/FIT_TIME_STRUCTURE.json',result)
    write(OUT/'AUDIT.json',dict(status='PASS',result_sha256=sha(OUT/'RESULT.json'),plan_sha256=sha(path),source_sha256=sha(source)))
    write(OUT/'STATE.json',dict(status='COMPLETE_AUDITED',cases=len(rows),seconds=time.time()-began))
    print(json.dumps([dict(category=r['category'],peaks=r['waveform_peaks'][:1],envelope=r['envelope_peaks'][:1]) for r in rows]),flush=True)


if __name__=='__main__':run()
