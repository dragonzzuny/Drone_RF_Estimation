"""Known-category TRAIN-only PSD/NNLS baseline for the current B split.

The older spectral baseline used all original TRAIN recordings, which include
the later B DEV time blocks. Reusing those templates would contaminate B.
Only the frozen spectral algorithm is reused; templates are fitted anew once.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/bandwidth_timeblock_20261011'))
from data import Library,Mixtures,MANIFEST,ROLES,read
from drone_rf.context_data import component_gains
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from spectral_baseline import profile,infer
import torch
from drone_rf.waveform import waveform_metrics

OUT=ROOT/'local/b_timeblock_spectral_20261013_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists()
    sources=[Path(__file__),ROOT/'experiments/rfuav_native_frequency_20261009/spectral_baseline.py',
        ROOT/'experiments/bandwidth_timeblock_20261011/data.py',MANIFEST]
    hashes={str(p):sha(p) for p in sources}
    write(OUT/'PLAN.json',dict(registered_time=time.time(),hashes=hashes,
        scope='Only fit_block templates; same fixed within_record_dev630; no heldout',
        algorithm='Unchanged native20261009 PSD templates + nonnegative least squares + fixed spectral power masks',
        fitting='Unit-sum context PSD; average contexts within file; equal file means within category',
        hyperparameter_search=False,templates_reused_from_old_C=False,
        count_threshold=.01,coefficient_floor=1e-8,mixture_phase=True,
        receiver_band_is_observed_metadata=True,reference_or_count_in_inference=False))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    clips=read(MANIFEST)['clips'];lib=Library(clips,ROLES[0]);files=defaultdict(list);metadata={};ids=[]
    for i,c in enumerate(clips):
        if c['diagnostic_role']!=ROLES[0]:continue
        psd=profile(lib._array(i));psd=psd/psd.sum();files[c['source_path']].append(psd)
        metadata[c['source_path']]=(c['category'],c['common_center_hz']);ids.append(c['clip_id'])
        if len(ids)%64==0:write(OUT/'STATE.json',dict(status='FIT_TEMPLATES',contexts=len(ids),total=1296,pid=os.getpid()))
    assert len(ids)==1296
    bycat=defaultdict(list);centers={}
    for name,psds in files.items():
        cat,center=metadata[name];centers[cat]=center;bycat[cat].append(np.mean(psds,axis=0))
    banks={}
    for center in sorted(set(centers.values())):
        cats=sorted(c for c in bycat if centers[c]==center)
        banks[center]=dict(categories=cats,psd=[np.mean(bycat[c],axis=0).tolist() for c in cats])
    write(OUT/'TEMPLATES.json',dict(fit_contexts=len(ids),fit_files=len(files),fit_groups=len(bycat),banks=banks,
        clip_ids=ids,dev_contexts_used=False,plan_sha256=sha(OUT/'PLAN.json')))
    data=Mixtures(ROLES[1]);rows=[]
    for i,row in enumerate(data.rows):
        k=int(row['count']);indices=row['indices'][:k];cs=[data.library.clips[int(j)] for j in indices]
        center=cs[0]['common_center_hz'];assert all(c['common_center_hz']==center for c in cs)
        gains=component_gains([c['mean_power'] for c in cs],row['levels'][:k],row['phases'][:k])
        mix=np.zeros(cs[0]['samples'],np.complex64)
        for j,g in zip(indices,gains):mix+=(data.library._array(int(j))*g).astype(np.complex64)
        item=data[i];start=int(row['crop_start'])
        assert np.array_equal(mix[start:start+len(item['mixture'])],item['mixture'])
        # Observed mixture waveform/profile and receiver-band dictionary only.
        y,pred_count,fractions,residual=infer(item['mixture'],profile(mix),banks[center])
        ref=torch.from_numpy(item['references'])[None];m=torch.from_numpy(item['mixture'])[None]
        metrics=waveform_metrics(y,ref,torch.from_numpy(item['active'])[None],m)
        assert float(metrics['sum_relative_error'][0])<1e-9
        si=metrics['si_sdr'][0,:k].tolist()
        rows.append(dict(index=i,count=k,categories=[c['category'] for c in cs],
            nmse=metrics['nmse'][0,:k].tolist(),si_sdr=[v if np.isfinite(v) else None for v in si],
            weakest_index=int(metrics['reference_power'][0,:k].argmin()),reference_power=metrics['reference_power'][0,:k].tolist(),
            predicted_count=pred_count,assignment=metrics['assignment'][0].tolist(),
            fitted_fractions=fractions,nnls_residual=residual,sum_relative_error=float(metrics['sum_relative_error'][0])))
        if i%25==0:write(OUT/'STATE.json',dict(status='EVALUATING',cases=i+1,total=len(data),pid=os.getpid()))
    from sequence_eval import summarize
    result=dict(status='COMPLETE',rows=rows,**summarize(rows),fit_contexts=1296,fit_groups=5,
        fixed_dev_schedule_sha256=data.rows_hash,known_category_dictionary=True,independent_recording_validation=False)
    write(OUT/'RESULT.json',result)
    assert all(sha(p)==h for p,h in hashes.items())
    assert not set(ids)&{c['clip_id'] for c in clips if c['diagnostic_role']==ROLES[1]}
    write(OUT/'AUDIT.json',dict(status='PASS',hashes_unchanged=True,fit_dev_ids_disjoint=True,
        result_sha256=sha(OUT/'RESULT.json'),templates_sha256=sha(OUT/'TEMPLATES.json'),time=time.time()))
    write(OUT/'STATE.json',dict(status='COMPLETE_AUDITED',pid=os.getpid()))
    print('B_SPECTRAL_CONTROL_COMPLETE',json.dumps(result['by_count']),flush=True)


if __name__=='__main__':run()
