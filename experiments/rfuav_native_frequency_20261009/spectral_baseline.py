"""TRAIN-only PSD templates, mixture-only NNLS, and soft spectral filtering.

An explicitly limited known-category baseline. References and true count enter
only scoring. It is not an oracle mask or a blind unknown-modulation separator.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from scipy import fft
from scipy.optimize import nnls
import torch
from native import FS,BANDS,band_for,LENGTH
from native_data import ROOT,LEGACY,HERE,NativeLibrary,NativeMixtures,sha256,write_json,component_gains
from drone_rf.waveform import analyze,synthesize,waveform_metrics
from study import finite_values,value_status

FFT=512


def profile(x):
    """Mean two-sided sqrt-Hann power periodogram, no reference normalization."""
    blocks=x[:len(x)//FFT*FFT].reshape(-1,FFT)
    win=np.sqrt(.5-.5*np.cos(2*np.pi*np.arange(FFT)/FFT)).astype(np.float32)
    spec=fft.fft(blocks*win,axis=-1,workers=1)
    return np.mean(spec.real.astype(np.float64)**2+spec.imag.astype(np.float64)**2,axis=0)


def freeze(root):
    names=[HERE/'spectral_baseline.py',HERE/'native.py',HERE/'native_data.py',
           *LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in names}
    protocol=dict(status='REGISTERED_TRAIN_SPECTRAL_BASELINE',source_sha256=sources,
        prep_protocol_sha256=sha256(root/'PREP_PROTOCOL.json'),fft=FFT,seed=0,
        templates='normalize each TRAIN-context mean PSD to unit sum; average contexts within source file, files within pack, packs within category',
        mixture_fit='NNLS of normalized long-mixture mean PSD against all training category templates in receiver band',
        inference='constant-in-time per-frequency power masks from fitted template powers; normalized to sum1; mixture phase retained',
        coefficient_floor=1e-8,count_threshold_fraction=.01,
        count_meaning='construction-count diagnostic; not physical count or activity annotation',
        background_output='zero; soft masks partition all observed mixture power',
        phase_gain_reference_fitting=False,reference_identity_at_inference=False,true_count_at_inference=False,
        validation_cases=630,heldout_read=False,hyperparameter_search=False)
    path=root/'SPECTRAL_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise RuntimeError('Spectral protocol changed')
    else:
        for name,digest in sources.items():
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha256(target)!=digest:raise RuntimeError('Snapshot conflict')
            if not target.exists():shutil.copyfile(ROOT/name,target)
        write_json(path,protocol)
    return sha256(path)


def templates(root,protocol):
    path=root/'spectral/TEMPLATES.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value['protocol_sha256']!=protocol:raise RuntimeError('Templates changed')
        return value
    lib=NativeLibrary(root/'preparation/NATIVE_MANIFEST.json','train_pack')
    by_file={};counts=defaultdict(int);metadata={};clips_used=[]
    for i,c in enumerate(lib.clips):
        if c['role']!='train_pack':continue
        p=profile(lib._array(i));p=p/p.sum();key=c['source_path']
        by_file[key]=by_file.get(key,np.zeros(FFT))+p;counts[key]+=1
        metadata[key]=(c['pack_id'],c['category'],band_for(c['center_hz']))
        clips_used.append(c['clip_id'])
        if len(clips_used)%100==0:
            write_json(root/'SPECTRAL_PROGRESS.json',dict(stage='TRAIN_TEMPLATES',completed=len(clips_used),total=3902,time=time.time(),pid=os.getpid()))
    by_pack=defaultdict(list);pack_meta={}
    for name,p in by_file.items():
        pack,cat,band=metadata[name];by_pack[pack].append(p/counts[name]);pack_meta[pack]=(cat,band)
    by_cat=defaultdict(list);cat_band={}
    for pack,values in by_pack.items():
        cat,band=pack_meta[pack];by_cat[cat].append(np.mean(values,axis=0));cat_band[cat]=band
    banks=[]
    for band in BANDS:
        cats=sorted(c for c in by_cat if cat_band[c]==band)
        banks.append(dict(band=band,categories=cats,psd=[np.mean(by_cat[c],axis=0).tolist() for c in cats]))
    if len(clips_used)!=3902 or len(by_file)!=83 or len(by_pack)!=8:
        raise RuntimeError('TRAIN template scope differs')
    value=dict(status='TRAIN_ONLY_TEMPLATES_READY',protocol_sha256=protocol,banks=banks,
        train_contexts=3902,train_files=83,train_packs=8,clip_ids=clips_used,
        native_manifest_sha256=sha256(root/'preparation/NATIVE_MANIFEST.json'),validation_used=False)
    write_json(path,value);return value


def infer(mixture,long_profile,bank):
    matrix=np.asarray(bank['psd'],dtype=np.float64).T
    y=long_profile/long_profile.sum()
    coefficients,residual=nnls(matrix,y)
    if coefficients.sum()<=0:raise RuntimeError('Empty positive mixture fit')
    fractions=coefficients/coefficients.sum()
    predicted_count=int(np.clip(np.count_nonzero(fractions>=.01),1,3))
    positive=coefficients+coefficients.sum()*1e-8
    power=matrix*positive[None]
    masks=(power/np.maximum(power.sum(1,keepdims=True),1e-30)).T
    x=torch.as_tensor(mixture)[None];z=analyze(x)
    spectra=torch.zeros((1,4,*z.shape[-2:]),dtype=z.dtype)
    spectra[0,:len(positive)]=torch.tensor(masks,dtype=torch.float32)[...,None]*z[0]
    output=synthesize(spectra,len(mixture))
    return output,predicted_count,fractions.tolist(),float(residual)


def evaluate(root,bank_value,protocol):
    data=NativeMixtures(root/'preparation','validation_pack',1)
    banks={b['band']:b for b in bank_value['banks']};rows=[]
    for i,row in enumerate(data.rows):
        count=int(row['count']);indices=row['indices'][:count]
        clips=[data.library.clips[int(j)] for j in indices]
        band=band_for(clips[0]['center_hz'])
        gains=component_gains([c['mean_power'] for c in clips],row['levels'][:count],row['phases'][:count])
        mix=np.zeros(LENGTH,np.complex64)
        for j,gain in zip(indices,gains):mix+=(data.library._array(int(j))*gain).astype(np.complex64)
        item=data[i];p=profile(mix)
        if not np.array_equal(mix[int(row['crop_start']):int(row['crop_start'])+data.length],item['mixture']):
            raise RuntimeError('Baseline input mismatch')
        # infer receives mixture and receiver-band template bank ONLY.
        output,pred_count,coefficients,residual=infer(item['mixture'],p,banks[band])
        active=torch.as_tensor(item['active'])[None];ref=torch.as_tensor(item['references'])[None]
        score=waveform_metrics(output,ref,active,torch.as_tensor(item['mixture'])[None])
        power=score['reference_power'][0][active[0]];si=score['si_sdr'][0][active[0]]
        gain=si-score['input_si_sdr'][0][active[0]]
        rows.append(dict(index=i,count=count,categories=[c['category'] for c in clips],pack_ids=[c['pack_id'] for c in clips],
            nominal_levels_db=row['levels'][:count].tolist(),reference_power=power.tolist(),
            nmse=finite_values(score['nmse'][0][active[0]]),si_sdr=finite_values(si),si_status=value_status(si),
            input_si_sdr=finite_values(score['input_si_sdr'][0][active[0]]),si_sdr_gain=finite_values(gain),
            weakest_index=int(power.argmin()),predicted_count=pred_count,
            assignment=score['assignment'][0].tolist(),bank_categories=banks[band]['categories'],
            fitted_power_fractions=coefficients,nnls_residual=residual,
            inactive_leak=float(score['inactive_leak'][0].sum()),background_nmse=float(score['background_nmse'][0]),
            sum_relative_error=float(score['sum_relative_error'][0])))
        if rows[-1]['sum_relative_error']>1e-9:raise RuntimeError('Baseline sum inconsistency')
        if (i+1)%25==0:
            write_json(root/'SPECTRAL_PROGRESS.json',dict(stage='VALIDATION',completed=i+1,total=len(data),time=time.time(),pid=os.getpid()))
    groups=[]
    for count in (1,2,3):
        group=[r for r in rows if r['count']==count]
        def avg(key):
            values=[v for r in group for v in r[key]]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        groups.append(dict(count=count,cases=len(group),mean_nmse=avg('nmse'),mean_si_sdr=avg('si_sdr'),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            construction_count_accuracy=float(np.mean([r['count']==r['predicted_count'] for r in group])),
            nonfinite_si_sdr=sum(v is None for r in group for v in r['si_sdr'])))
    write_json(root/'spectral/VALIDATION.json',dict(status='COMPLETE',protocol_sha256=protocol,
        by_count=groups,rows=rows,independent_test=False,heldout_read=False))
    print(json.dumps(dict(stage='SPECTRAL_BASELINE_COMPLETE',by_count=groups)),flush=True)


def run(root):
    protocol=freeze(root);(root/'spectral').mkdir(exist_ok=True)
    if (root/'SPECTRAL_COMPLETE.json').exists():raise RuntimeError('Already completed')
    while not (root/'preparation/features/validation_pack_001.json').exists():
        if (root/'PREP_FAILURE.json').exists():raise RuntimeError('CPU prep failed')
        time.sleep(10)
    torch.set_num_threads(2)
    value=templates(root,protocol);evaluate(root,value,protocol)
    if freeze(root)!=protocol:raise RuntimeError('Source changed')
    write_json(root/'SPECTRAL_COMPLETE.json',dict(status='COMPLETE',protocol_sha256=protocol,time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();os.sched_setaffinity(0,{12,13});os.nice(15)
    try:run(args.run.resolve())
    except Exception:
        write_json(args.run/'SPECTRAL_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
