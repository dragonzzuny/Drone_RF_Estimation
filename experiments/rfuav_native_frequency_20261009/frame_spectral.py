"""Prespecified frame-wise NNLS comparison using the SAME TRAIN PSD bank.

Tests local power adaptation, not recurrence/periodicity or a new deep network.
The constant-mask baseline's count prediction is carried over for scoring;
no true count or identity enters waveform reconstruction.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from native_data import ROOT,LEGACY,HERE,NativeMixtures,sha256,write_json
from native import band_for
from drone_rf.waveform import analyze,synthesize,waveform_metrics
from study import finite_values,value_status


def frame_nnls(matrix,power):
    """Exact active-set enumeration for 2/3 templates, independently per frame."""
    matrix=np.asarray(matrix,dtype=np.float64);power=np.asarray(power,dtype=np.float64)
    if matrix.ndim!=2 or power.ndim!=2 or matrix.shape[0]!=power.shape[0] or matrix.shape[1] not in (2,3):
        raise ValueError('Expected 2/3 spectral templates and matching frame powers')
    if not np.isfinite(matrix).all() or not np.isfinite(power).all() or np.any(matrix<0) or np.any(power<0):
        raise ValueError('Finite nonnegative powers required')
    gram=matrix.T@matrix;right=matrix.T@power
    k,frames=right.shape;best=np.zeros((k,frames));cost=np.zeros(frames)
    for size in range(1,k+1):
        for active in itertools.combinations(range(k),size):
            idx=np.array(active);coef=np.linalg.pinv(gram[np.ix_(idx,idx)],rcond=1e-12)@right[idx]
            feasible=(coef>=-1e-12).all(axis=0);coef=np.maximum(coef,0)
            candidate=np.zeros((k,frames));candidate[idx]=coef
            objective=np.sum(candidate*(gram@candidate-2*right),axis=0)
            replace=feasible & (objective<cost)
            best[:,replace]=candidate[:,replace];cost[replace]=objective[replace]
    return best


@torch.no_grad()
def infer(mixture,bank):
    x=torch.as_tensor(mixture)[None];z=analyze(x)
    powers=z[0].abs().square().numpy().astype(np.float64)
    normalized=powers/np.maximum(powers.sum(axis=0,keepdims=True),1e-30)
    matrix=np.asarray(bank['psd'],dtype=np.float64).T
    coef=frame_nnls(matrix,normalized)
    smoothed=coef+np.maximum(coef.sum(axis=0,keepdims=True),1e-30)*1e-8
    allocated=matrix.T[:,:,None]*smoothed[:,None,:]
    masks=allocated/np.maximum(allocated.sum(axis=0,keepdims=True),1e-300)
    # All-zero spectral bins have zero mixture contribution; ensure valid partition.
    empty=masks.sum(axis=0)==0
    masks[:,empty]=1/len(masks)
    spectra=torch.zeros((1,4,*z.shape[-2:]),dtype=z.dtype)
    spectra[0,:len(masks)]=torch.as_tensor(masks,dtype=torch.float32)*z[0]
    return synthesize(spectra,len(mixture))


def freeze(root):
    files=[HERE/'frame_spectral.py',HERE/'native.py',HERE/'native_data.py',
           *LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in files}
    protocol=dict(status='REGISTERED_LOCAL_POWER_COMPARISON',source_sha256=sources,
        preparation_protocol_sha256=sha256(root/'PREP_PROTOCOL.json'),
        spectral_protocol_sha256=sha256(root/'SPECTRAL_PROTOCOL.json'),
        templates='unchanged TRAIN-only PSD bank from registered constant-mask baseline',
        intervention='NNLS coefficients independently re-estimated for every observed mixture STFT frame',
        fft=512,hop=128,window='sqrt-Hann',coefficient_floor=1e-8,
        prediction_inputs='mixture IQ and receiver-band template bank only',
        count_evaluation='carry over constant-mask baseline mixture-only count prediction; no new count optimization',
        questions=['does local power fitting reduce unscaled waveform NMSE and improve complex SI-SDR?',
                   'do weakest contributions improve?'],
        acceptance='both metrics better for each count2/3 against constant masks; report all630cases and weakest',
        neural_training_updates=0,threshold_search=False,heldout_read=False,independent_test=False,
        temporal_claim='local adaptation only; no learned temporal recurrence or hopping model')
    path=root/'FRAME_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise RuntimeError('Frame protocol changed')
    else:
        for name,digest in sources.items():
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha256(target)!=digest:raise RuntimeError('Snapshot conflict')
            if not target.exists():shutil.copyfile(ROOT/name,target)
        write_json(path,protocol)
    return sha256(path)


def run(root):
    protocol=freeze(root)
    if (root/'FRAME_COMPLETE.json').exists():raise RuntimeError('Already complete')
    while not (root/'SPECTRAL_COMPLETE.json').exists():
        if (root/'SPECTRAL_FAILURE.json').exists():raise RuntimeError('Constant-mask baseline failed')
        time.sleep(10)
    torch.set_num_threads(2)
    bank_value=json.loads((root/'spectral/TEMPLATES.json').read_text())
    banks={b['band']:b for b in bank_value['banks']}
    prior=json.loads((root/'spectral/VALIDATION.json').read_text())['rows']
    data=NativeMixtures(root/'preparation','validation_pack',1);rows=[]
    for i,row in enumerate(data.rows):
        item=data[i];count=int(row['count'])
        # Known receiver band is input metadata. No subset of true classes is used.
        band=band_for(data.library.clips[int(row['indices'][0])]['center_hz'])
        estimates=infer(item['mixture'],banks[band])
        active=torch.as_tensor(item['active'])[None]
        m=waveform_metrics(estimates,torch.as_tensor(item['references'])[None],active,
                           torch.as_tensor(item['mixture'])[None])
        power=m['reference_power'][0][active[0]];si=m['si_sdr'][0][active[0]]
        np.testing.assert_allclose(power.tolist(),prior[i]['reference_power'],rtol=1e-10,atol=1e-15)
        result={k:prior[i][k] for k in ('index','count','categories','pack_ids','nominal_levels_db','predicted_count')}
        gain=si-m['input_si_sdr'][0][active[0]]
        result.update(reference_power=power.tolist(),nmse=finite_values(m['nmse'][0][active[0]]),
            si_sdr=finite_values(si),si_status=value_status(si),si_sdr_gain=finite_values(gain),
            input_si_sdr=finite_values(m['input_si_sdr'][0][active[0]]),weakest_index=int(power.argmin()),
            assignment=m['assignment'][0].tolist(),inactive_leak=float(m['inactive_leak'][0].sum()),
            background_nmse=float(m['background_nmse'][0]),sum_relative_error=float(m['sum_relative_error'][0]))
        if result['sum_relative_error']>1e-9:raise RuntimeError('Frame-mask sum failed')
        rows.append(result)
        if (i+1)%25==0:write_json(root/'FRAME_PROGRESS.json',dict(stage='VALIDATION',completed=i+1,total=630,time=time.time(),pid=os.getpid()))
    groups=[]
    for count in (1,2,3):
        group=[r for r in rows if r['count']==count]
        def avg(key):
            values=[v for r in group for v in r[key]]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        groups.append(dict(count=count,cases=len(group),mean_nmse=avg('nmse'),mean_si_sdr=avg('si_sdr'),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in group])),
            construction_count_accuracy=float(np.mean([r['count']==r['predicted_count'] for r in group]))))
    if freeze(root)!=protocol:raise RuntimeError('Frame source changed')
    write_json(root/'spectral/FRAME_VALIDATION.json',dict(status='COMPLETE',protocol_sha256=protocol,
        templates_sha256=sha256(root/'spectral/TEMPLATES.json'),by_count=groups,rows=rows,
        heldout_read=False,independent_test=False))
    write_json(root/'FRAME_COMPLETE.json',dict(status='COMPLETE',protocol_sha256=protocol,time=time.time(),by_count=groups))
    print(json.dumps(dict(stage='FRAME_COMPLETE',by_count=groups)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{12,13});os.nice(15)
    try:run(args.run.resolve())
    except Exception:
        write_json(args.run/'FRAME_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
