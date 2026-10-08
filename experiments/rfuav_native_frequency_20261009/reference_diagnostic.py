"""Reference-assisted native power masks: diagnosis only, NEVER inference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from native_data import ROOT,HERE,LEGACY,NativeMixtures,sha256,write_json
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_reference_tf_power import reference_masks
from drone_rf.waveform import analyze,synthesize,complex_si_sdr
from study import finite_values,value_status


def run(root):
    output=root/'REFERENCE_NATIVE_POWER.json'
    if output.exists() or (root/'REFERENCE_PROTOCOL.json').exists():raise RuntimeError('Refuse duplicate diagnostic')
    files=[HERE/'reference_diagnostic.py',HERE/'native.py',HERE/'native_data.py',
           ROOT/'scripts/diagnose_reference_tf_power.py',*LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in files}
    protocol=dict(status='REGISTERED_REFERENCE_ASSISTED_DIAGNOSTIC',source_sha256=sources,
        native_preparation_sha256=sha256(root/'preparation/PREPARATION.json'),
        scope='all420already-used development mixtures with count2/3',
        masks=['reference_frequency_mean_power','reference_time_frequency_power'],
        fft=512,hop=128,window='sqrt-Hann',requires_reference_iq=True,requires_true_count=True,
        deployable=False,performance_bound=False,model_updates=0,heldout_read=False,
        interpretation='diagnostic with unavailable source information; does not prove inferability or optimality')
    for rel,digest in sources.items():
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=digest:raise RuntimeError('Snapshot conflict')
        if not target.exists():shutil.copyfile(ROOT/rel,target)
    write_json(root/'REFERENCE_PROTOCOL.json',protocol)
    while not (root/'PREP_COMPLETE.json').exists():
        if (root/'PREP_FAILURE.json').exists():raise RuntimeError('Preparation failed')
        time.sleep(10)
    torch.set_num_threads(2);data=NativeMixtures(root/'preparation','validation_pack',1)
    parent=json.loads((root/'gpu/VALIDATION_000.json').read_text())['rows']
    rows=[];start=time.time()
    with torch.no_grad():
        for i,row in enumerate(data.rows):
            count=int(row['count'])
            if count==1:continue
            item=data[i];refs=torch.as_tensor(item['references'])[None];mix=torch.as_tensor(item['mixture'])[None]
            length=mix.shape[-1];spectra=analyze(refs.reshape(-1,length)).reshape(1,3,512,-1);z=analyze(mix)
            power=refs[:,:count].to(torch.complex128).abs().square().mean(-1)
            np.testing.assert_allclose(power[0].tolist(),parent[i]['reference_power'],rtol=1e-10,atol=1e-15)
            roundtrip=((synthesize(spectra,length)[:,:count]-refs[:,:count]).to(torch.complex128).abs().square().mean(-1)/power)[0]
            if float(roundtrip.max())>1e-9:raise RuntimeError('Reference STFT round trip failed')
            modes={}
            for name,mask in reference_masks(spectra,count).items():
                estimate=synthesize(mask*z[:,None],length)
                error=estimate[:,:count].to(torch.complex128)-refs[:,:count].to(torch.complex128)
                nmse=(error.abs().square().mean(-1)/power)[0]
                si=complex_si_sdr(estimate[:,:count].to(torch.complex128),refs[:,:count].to(torch.complex128))[0]
                sum_error=float((estimate.sum(1)-mix).abs().square().sum()/mix.abs().square().sum())
                if not bool(torch.isfinite(nmse).all()) or sum_error>1e-9:raise RuntimeError('Reference diagnostic failed')
                modes[name]=dict(nmse=nmse.tolist(),si_sdr=finite_values(si),si_status=value_status(si),sum_relative_error=sum_error)
            rows.append(dict(index=i,count=count,categories=parent[i]['categories'],pack_ids=parent[i]['pack_ids'],
                reference_power=power[0].tolist(),weakest_index=int(power[0].argmin()),roundtrip_nmse=roundtrip.tolist(),modes=modes))
            if len(rows)%50==0:write_json(root/'REFERENCE_PROGRESS.json',dict(completed=len(rows),total=420,seconds=time.time()-start,time=time.time(),pid=os.getpid()))
    groups=[]
    for count in (2,3):
        subset=[r for r in rows if r['count']==count]
        if len(subset)!=210:raise RuntimeError('Missing diagnostic cases')
        modes={}
        for name in protocol['masks']:
            si=[v for r in subset for v in r['modes'][name]['si_sdr']]
            modes[name]=dict(mean_nmse=float(np.mean([v for r in subset for v in r['modes'][name]['nmse']])),
                mean_si_sdr=float(np.mean(si)) if all(v is not None for v in si) else None,
                weakest_nmse=float(np.mean([r['modes'][name]['nmse'][r['weakest_index']] for r in subset])))
        groups.append(dict(count=count,cases=210,modes=modes))
    if any(sha256(ROOT/p)!=digest for p,digest in sources.items()):raise RuntimeError('Diagnostic code changed')
    result=dict(protocol,status='COMPLETE',seconds=time.time()-start,summary=groups,rows=rows,
                protocol_sha256=sha256(root/'REFERENCE_PROTOCOL.json'))
    write_json(output,result);write_json(root/'REFERENCE_COMPLETE.json',dict(status='COMPLETE',time=time.time()))
    print(json.dumps(dict(reference_assisted_only=True,summary=groups,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{12,13});os.nice(15)
    try:run(args.run.resolve())
    except Exception:
        write_json(args.run/'REFERENCE_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
