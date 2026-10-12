"""Fixed power-level diagnostic of the final parent, on CPU beside GPU training.

One previously exposed event, four predeclared power ratios. No new examples
are selected based on CRC, no receiver tuning and no model training here.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/packet_power_probe_20261013_v1'
PARENT=ROOT/'local/timeblock_extend100_20261012_v1'
OLD=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012'
LEVELS=(0,-10,10,-20)


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def state(status,**kw):write(OUT/'STATE.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))


def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'PLAN.json').exists()
    hashes={str(ROOT/p):h for p,h in read(PARENT/'PROTOCOL.json')['source_sha256'].items()}
    for p in [Path(__file__),Path(__file__).with_name('sequence_eval.py'),Path(__file__).with_name('sequence_data.py'),
              Path(__file__).with_name('sequence_objective.py'),Path(__file__).with_name('check_packets.py')]:hashes[str(p)]=sha(p)
    plan=dict(registered=time.time(),source_sha256=hashes,levels_FPV_over_Mavic_db=list(LEVELS),
        selection='Existing macro NMSE best after audited parent100',cpu_only=True,optimizer_updates=0,
        event='Same exposed FPV/Mavic3Pro2ms event, native frequency offsets, original phase pair0/.7',
        normalization='Whole20.8896ms references; no per-window power adjustment',
        receiver='Existing immutable fixed15MHz pilot receiver; all modes unchanged; no CRC-guided tuning',
        inference='Original short-window C8/Hann, prediction-only overlap correspondence',
        primary='Record mixture and all outputs CRC status at every fixed power condition',
        waveform_metrics=True,reference_in_receiver=False,no_payload_fields_stored=True,
        outside_nominal_training_ratio=[-20],independent_recording_validation=False,
        zero_db_cpu_vs_saved_gpu_relative_squared_error_limit=1e-9,
        caveat='One known packet reused across power levels; not an identification rate or commercial-detector test')
    write(OUT/'PLAN.json',plan)
    while True:
        s=read(PARENT/'STATE.json')
        if s['status']=='COMPLETE_AUDITED' and s['epoch']==100:break
        if s['status'] in ('FAILED','STOPPED_WITH_RECOVERY'):raise RuntimeError('Parent needs attention')
        state('WAITING_PARENT_100',parent_epoch=s.get('epoch'));time.sleep(30)
    e=read(PARENT/'EPOCH_100.json');a=read(PARENT/'AUDIT_100.json')
    assert a['status']=='PASS' and a['event_sha256']==sha(PARENT/'EPOCH_100.json')
    selected=e['best']['epoch'];d=PARENT if selected>50 else ROOT/'local/bandwidth_timeblock_20261011_v1'
    cp=d/f'ACTUAL_{selected:03d}.pt';assert sha(cp)==read(d/f'EPOCH_{selected:03d}.json')['checkpoint_sha256']
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'INITIALIZATION.json',dict(epoch=selected,checkpoint=str(cp),checkpoint_sha256=sha(cp)))
    import numpy as np
    import torch
    from sequence_eval import merge,score
    from sequence_data import WINDOW
    from models import build,predict
    from core import predict_eight
    from study import batch
    from drone_rf.context_data import component_gains,mixture_context_features
    from check_packets import receiver
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rp=read(OLD/'receiver_pilot_v1/PLAN.json')
    for group in ('hashes','original_sources'):
        for p,h in rp[group].items():assert sha(p)==h,p
    p=read(OLD/'raw_diagnosis/SEPARATION_PLAN.json')
    first_path=OLD/'raw_diagnosis/FIRST_EVENT.npz';second_path=Path(p['second_clip']['cache_path'])
    assert sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256']
    first=np.load(first_path)['filtered'];second=np.load(second_path)
    powers=[float(np.mean(abs(x.astype(np.complex128))**2)) for x in (first,second)]
    original=np.load(OLD/'retest_best_e079/SEPARATED_EVENT.npz');lo,hi=p['roi']
    net=build('unet_mean').eval().requires_grad_(False)
    saved=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);net.load_state_dict(saved['model']);del saved
    results=[]
    with torch.inference_mode():
        for level in LEVELS:
            gains=component_gains(powers,[level,0],[0,.7]);refs=np.stack([(x*g).astype(np.complex64) for x,g in zip((first,second),gains)])
            mix=refs.sum(0);features=mixture_context_features(mix)['context_features']
            if level==0:
                assert np.array_equal(mix[lo:hi],original['mixture'])
                assert np.array_equal(refs[:,lo:hi],original['references'][:2])
            ys=[];began=time.time()
            for j,start in enumerate(p['starts']):
                item=batch(dict(mixture=mix[start:start+WINDOW].copy(),context_features=features,crop_start=start,
                    references=np.zeros((3,WINDOW),np.complex64),active=np.zeros(3,bool),construction_count=2),'cpu')
                _,y,logits,_=predict_eight(net,predict,item);ys.append(y[0].numpy())
                state('CPU_C8_WINDOWS',epoch=selected,level_db=level,windows=j+1,total=len(p['starts']),seconds=time.time()-began)
            begin=p['starts'][0];length=p['starts'][-1]+WINDOW-begin
            output,receipt=merge(ys,[s-begin for s in p['starts']],length);output=output[:,lo-begin:hi-begin]
            target=np.zeros((3,hi-lo),np.complex64);target[:2]=refs[:,lo:hi]
            metrics=score(torch.from_numpy(output)[None],torch.from_numpy(target)[None],torch.from_numpy(mix[lo:hi])[None],2,logits)
            cpu_gpu_error=None
            if level==0:
                baseline=ROOT/'local/packet_final100_20261013_v1'
                while not (baseline/'VERIFICATION.json').exists():
                    if (baseline/'FAILURE.json').exists():raise RuntimeError('Final GPU event diagnosis failed')
                    state('WAITING_GPU_EVENT_RECEIPT');time.sleep(15)
                assert read(baseline/'INITIALIZATION.json')['checkpoint_sha256']==sha(cp)
                assert sha(baseline/'ASSEMBLED.npz')==read(baseline/'VERIFICATION.json')['artifacts']['ASSEMBLED.npz']
                previous=np.load(baseline/'ASSEMBLED.npz')['C8_hann']
                cpu_gpu_error=float(np.mean(abs(output-previous)**2)/np.mean(abs(mix[lo:hi])**2))
                assert cpu_gpu_error<=plan['zero_db_cpu_vs_saved_gpu_relative_squared_error_limit'],cpu_gpu_error
            tasks=[('reference_FPV',target[0]),('reference_Mavic3Pro',target[1]),('mixture',mix[lo:hi])]
            tasks += [(f'output_{j}',output[j]) for j in range(3)]+[('background',output[3])]
            decoded=[]
            for label,wave in tasks:
                v=receiver.receive(wave,15_000_000)
                attempts=[a for m in v.get('modes',[]) for a in m['attempts']]
                decoded.append(dict(label=label,status=v['status'],crc16=any(a['crc16_ok'] for a in attempts),
                    crc24=any(a['crc24_ok'] for a in attempts),
                    device_types=sorted({str(a['device_type']) for a in attempts if a['crc16_ok'] and a['device_type'] is not None})))
            assert decoded[0]['crc16'],'Reference positive control must decode'
            artifact=OUT/f'POWER_{level:+03d}.npz';np.savez_compressed(artifact,outputs=output,references=target,mixture=mix[lo:hi])
            results.append(dict(level_FPV_over_Mavic_db=level,waveform=metrics,receiver=decoded,
                cpu_gpu_zero_relative_squared_error=cpu_gpu_error,artifact_sha256=sha(artifact),
                overlap_matching=receipt,seconds=time.time()-began))
            write(OUT/'RESULT.json',dict(epoch=selected,conditions=results,reference_in_receiver=False,complete=len(results)==len(LEVELS)))
            print('POWER_CONDITION_COMPLETE',level,json.dumps(decoded),flush=True)
    assert all(sha(path)==h for path,h in hashes.items())
    assert sha(first_path)==p['first_event_hash'] and sha(second_path)==p['second_clip']['cache_sha256']
    write(OUT/'AUDIT.json',dict(status='PASS',conditions=len(results),result_sha256=sha(OUT/'RESULT.json'),
        plan_sha256=sha(OUT/'PLAN.json'),inputs_unchanged=True,source_hashes_unchanged=True))
    state('COMPLETE_AUDITED',epoch=selected,conditions=4)


if __name__=='__main__':
    try:run()
    except BaseException:
        OUT.mkdir(exist_ok=True);write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));state('FAILED');raise
