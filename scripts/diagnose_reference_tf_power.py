"""Reference-assisted power allocation diagnostic, NEVER a deployable separator.

Fixed n_fft512/hop128 matches the incumbent. Both masks require the source
references and count. They are not bounds on nonlinear/complex separation.
Reads only already-used development mixtures and no checkpoints or heldout IQ.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0,str(LEGACY))
from study import admitted_dataset,batch,finite_values,value_status
from drone_rf.waveform import analyze,synthesize,complex_si_sdr
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256

PREPARATION=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')
SAVED=PREPARATION.parent/'dense_gpu_run/unet_mean/VALIDATION_001.json'


def reference_masks(spectra,count):
    power=spectra.abs().square()
    result={}
    for name,p in (('reference_frequency_mean_power',power.mean(-1,keepdim=True).expand_as(power)),
                   ('reference_time_frequency_power',power)):
        total=p.sum(1,keepdim=True)
        weights=p/torch.where(total>0,total,torch.ones_like(total))
        fallback=torch.zeros_like(weights);fallback[:,:count]=1/count
        result[name]=torch.where(total>0,weights,fallback)
    return result


@torch.no_grad()
def run(destination):
    if destination.exists():
        raise RuntimeError('Refuse overwrite')
    torch.set_num_threads(2)
    data=admitted_dataset(PREPARATION,'validation_pack',1)
    previous=json.loads(SAVED.read_text())
    hashes={str(p):sha256(p) for p in [Path(__file__),SAVED,PREPARATION/'PREPARATION.json',
        PREPARATION/'VALIDATION.npy',*LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]}
    protocol=dict(status='FROZEN_BEFORE_DIAGNOSTIC',source_data_sha256=hashes,
        scope='existing development count2/3, all420 cases; reference-assisted diagnostic only',
        masks=['reference_frequency_mean_power','reference_time_frequency_power'],
        fft512_hop128_sqrt_hann=True,heldout_iq_read=False,deployable=False,
        performance_bound=False,model_updates=0,checkpoint_selection_changed=False,
        question='How much error remains when true reference spectral powers determine mixture allocation?',
        limits='Observed power ratios use clean contributions, including their recorded noise; neither a physical oracle nor MMSE optimality is claimed.')
    write_json(destination.with_suffix('.protocol.json'),protocol)
    rows=[];start=time.time()
    for original in previous['rows']:
        index=original['index'];count=original['count']
        if count==1:
            continue
        item=batch(data[index],'cpu')
        refs=item['references'];mix=item['mixture'];length=mix.shape[-1]
        reference_spectrum=analyze(refs.reshape(-1,length)).reshape(1,3,512,-1)
        mixture_spectrum=analyze(mix)
        power=refs[:,:count].to(torch.complex128).abs().square().mean(-1)
        np.testing.assert_allclose(power[0].tolist(),original['reference_power'],rtol=1e-7,atol=1e-12)
        roundtrip=synthesize(reference_spectrum,length)
        roundtrip_nmse=((roundtrip[:,:count]-refs[:,:count]).to(torch.complex128).abs().square().mean(-1)/power)[0].tolist()
        if max(roundtrip_nmse)>1e-9:
            raise RuntimeError('STFT source round trip failed')
        modes={}
        for name,mask in reference_masks(reference_spectrum,count).items():
            estimates=synthesize(mask*mixture_spectrum[:,None],length)
            error=estimates[:,:count].to(torch.complex128)-refs[:,:count].to(torch.complex128)
            nmse=(error.abs().square().mean(-1)/power)[0]
            si=complex_si_sdr(estimates[:,:count].to(torch.complex128),refs[:,:count].to(torch.complex128))[0]
            relative_sum=float((estimates.sum(1)-mix).abs().square().sum()/mix.abs().square().sum())
            if not bool(torch.isfinite(nmse).all()) or relative_sum>1e-9:
                raise RuntimeError('Invalid diagnostic or reconstruction sum')
            modes[name]=dict(nmse=nmse.tolist(),si_sdr=finite_values(si),si_status=value_status(si),sum_relative_error=relative_sum)
        rows.append(dict(index=index,count=count,categories=original['categories'],pack_ids=original['pack_ids'],
            reference_power=original['reference_power'],weakest_index=original['weakest_index'],
            selected_unet_nmse=original['nmse'],selected_unet_si_sdr=original['si_sdr'],
            roundtrip_nmse=roundtrip_nmse,modes=modes))
        if len(rows)%50==0:
            write_json(destination.with_suffix('.progress.json'),dict(cases=len(rows),total=420,seconds=time.time()-start))
    groups=[]
    for count in (2,3):
        subset=[r for r in rows if r['count']==count]
        if len(subset)!=210:
            raise RuntimeError('Missing development cases')
        values={}
        for name in protocol['masks']:
            nmse=[v for r in subset for v in r['modes'][name]['nmse']]
            si=[v for r in subset for v in r['modes'][name]['si_sdr']]
            values[name]=dict(mean_nmse=float(np.mean(nmse)),
                mean_si_sdr=float(np.mean(si)) if all(v is not None for v in si) else None,
                si_nonfinite=sum(v is None for v in si),
                weakest_nmse=float(np.mean([r['modes'][name]['nmse'][r['weakest_index']] for r in subset])),
                beats_unet_cases=int(sum(np.mean(r['modes'][name]['nmse'])<np.mean(r['selected_unet_nmse']) for r in subset)))
        groups.append(dict(count=count,cases=210,
            unet_mean_nmse=float(np.mean([v for r in subset for v in r['selected_unet_nmse']])),
            unet_mean_si_sdr=float(np.mean([v for r in subset for v in r['selected_unet_si_sdr']])),modes=values))
    if any(sha256(Path(p))!=h for p,h in hashes.items()):
        raise RuntimeError('Diagnostic sources/data changed')
    result=dict(protocol, status='COMPLETE',seconds=time.time()-start,summary=groups,rows=rows)
    write_json(destination,result);print(json.dumps(dict(seconds=result['seconds'],summary=groups)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2] or cpus))
    os.nice(10);run(args.output)
