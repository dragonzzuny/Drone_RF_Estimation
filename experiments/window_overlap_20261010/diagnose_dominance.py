"""TRAIN48 source-dominant-bin coverage before proposing clustering labels."""
import argparse
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import diagnose as core
from drone_rf.waveform import analyze

ROOT=core.ROOT;w=core.w


def run(root,public):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    source=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    prior=w.read(source);fixed=[r for r in prior['rows'] if r['model']=='parent/e0']
    p=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    hashes=dict(p['source_sha256'])
    for name in ('diagnose.py','diagnose_dominance.py'):
        path=Path(__file__).with_name(name);hashes[str(path.relative_to(ROOT))]=w.digest(path)
    protocol=dict(status='REGISTERED_CPU_TRAIN_DOMINANCE',source_sha256=hashes,indices=[r['index'] for r in fixed],
        prior_sha256=w.digest(source),preparation_sha256=p['preparation_sha256'],
        thresholds=[.5,.8,.95],updates=0,heldout_read=False,gpu_use=False,
        scope='Same fixed TRAIN48 used earlier, all cases and source contributions; target-side diagnostics only',
        statistic='STFT reference power dominance; no clean-noise separation, model fitting or DEV selection',registered_at=time.time())
    assert len(fixed)==48
    for rel,sha in hashes.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol)
    torch.set_num_threads(2);started=time.time()
    data=core.base.worker.NativeMixtures(p['preparation'],'train_pack',1)
    rows=[]
    with torch.inference_mode():
        for position,index in enumerate(protocol['indices']):
            raw=data[index];count=int(raw['construction_count'])
            ref=torch.as_tensor(raw['references'][:count])
            spectrum=analyze(ref);power=spectrum.to(torch.complex128).abs().square()
            total=power.sum(0);fraction=power/total.clamp_min(1e-300)
            winner=power.argmax(0)
            wave_power=ref.to(torch.complex128).abs().square().mean(-1)
            weak=int(wave_power.argmin())
            for slot in range(count):
                category=data.library.clips[int(data.rows[index]['indices'][slot])]['category']
                energy=power[slot].sum();mask=winner==slot
                measurements=[]
                for threshold in protocol['thresholds']:
                    selected=fraction[slot]>=threshold
                    measurements.append(dict(threshold=threshold,bin_fraction=float(selected.double().mean()),
                        source_energy_fraction=float(power[slot][selected].sum()/energy)))
                row=dict(index=index,count=count,slot=slot,category=category,weakest=slot==weak,
                    waveform_power=float(wave_power[slot]),winning_bin_fraction=float(mask.double().mean()),
                    source_energy_in_winning_bins=float(power[slot][mask].sum()/energy),
                    energy_weighted_own_fraction=float((fraction[slot]*power[slot]).sum()/energy),
                    dominance=measurements)
                assert all(0<=row[k]<=1+1e-12 for k in ('winning_bin_fraction','source_energy_in_winning_bins','energy_weighted_own_fraction'))
                rows.append(row)
            w.write(root/'STATE.json',dict(status='CPU_TARGET_DOMINANCE',cases=position+1,total=48,pid=os.getpid(),time=time.time()))
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    summary=[]
    for count in (1,2,3):
        for group in ('all','weakest'):
            selected=[q for q in rows if q['count']==count and (group=='all' or q['weakest'])]
            summary.append(dict(count=count,group=group,contributions=len(selected),
                mean_winning_bin_fraction=float(np.mean([q['winning_bin_fraction'] for q in selected])),
                mean_source_energy_in_winning_bins=float(np.mean([q['source_energy_in_winning_bins'] for q in selected])),
                median_source_energy_in_winning_bins=float(np.median([q['source_energy_in_winning_bins'] for q in selected])),
                median_energy_in_80percent_bins=float(np.median([q['dominance'][1]['source_energy_fraction'] for q in selected]))))
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,summary=summary,
        seconds=time.time()-started,updates=0,gpu_use=False,heldout_read=False,
        limitation='Target-side TRAIN48 coverage, not a learned separator or an oracle waveform bound. Independent recording noise is included; sum of component powers is not asserted equal to mixture STFT power.')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=48,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',summary=summary,seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','public'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    try:run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
