"""CPU cache replay and deterministic, matched two-source mixture schedules.

This prepares inputs; it does not start or approve a GPU training run.
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

LENGTH=63872
DTYPE=np.dtype([('epoch','<i2'),('first','<i4'),('second','<i4'),('crop_start','<i4'),
                ('sir_db','<f4'),('phase_first','<f8'),('phase_second','<f8')])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while raw:=f.read(8*1024**2):h.update(raw)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def schedule(clips,role,epochs=1,repeats=4):
    groups=collections.defaultdict(list)
    for i,c in enumerate(clips):
        if c['role']==role:groups[c['category']].append(i)
    if len(groups)<2 or epochs<1 or repeats<1:raise ValueError('Insufficient mixture design')
    pairs=list(itertools.combinations(sorted(groups),2));rows=[]
    for epoch in range(1,epochs+1):
        rng=np.random.default_rng(np.random.SeedSequence([0,epoch,143 if role=='train_pack' else 144]))
        current=[]
        for first,second in pairs:
            for sir in (-10.,0.,10.):
                for _ in range(repeats):
                    a=int(rng.choice(groups[first]));b=int(rng.choice(groups[second]))
                    length=min(clips[a]['samples'],clips[b]['samples'])
                    if length<LENGTH:raise ValueError('Short source cache')
                    start=int(rng.integers(0,length-LENGTH+1))
                    current.append((epoch,a,b,start,sir,*rng.uniform(-np.pi,np.pi,2)))
        rows.extend(current[i] for i in rng.permutation(len(current)))
    return np.array(rows,dtype=DTYPE)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'COMPLETE.json').exists():raise FileExistsError('Existing completed preparation')
    os.nice(10)
    if args.cpus:os.sched_setaffinity(0,{int(n) for n in args.cpus.split(',')})
    started=time.time();manifest=json.loads(args.manifest.read_text());clips=manifest['clips']
    if manifest['status']!='RAW_CACHE_COMPLETE_PENDING_SPLIT_REVIEW':raise ValueError('Incomplete input cache')
    if manifest['exact_cross_role_duplicate_groups']:raise ValueError('Exact duplicate crosses split')
    allowed={'DJI AVATA2','DJI FPV COMBO','DJI MAVIC3 PRO','DJI MINI3','DJI MINI4 PRO'}
    roles=collections.defaultdict(set);max_power_delta=0.;identifiers=[]
    for i,c in enumerate(clips):
        if c['category'] not in allowed or c['role'] not in {'train_pack','validation_pack'}:
            raise ValueError('Controller/heldout/unexpected category in development cache')
        roles[c['pack_id']].add(c['role'])
        p=Path(c['cache_path'])
        if sha(p)!=c['cache_sha256']:raise ValueError('Cache file hash mismatch')
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        if x.shape!=(c['samples'],) or x.dtype!=np.dtype('complex64'):raise ValueError('Invalid cache array')
        if not np.isfinite(x).all():raise ValueError('Nonfinite source cache')
        h=hashlib.sha256();energy=0.
        for start in range(0,len(x),131072):
            chunk=x[start:start+131072];h.update(chunk.tobytes())
            re=chunk.real.astype(np.float64);im=chunk.imag.astype(np.float64)
            energy+=float(np.dot(re,re)+np.dot(im,im))
        if h.hexdigest()!=c['raw_clip_sha256']:raise ValueError('Decoded raw bytes disagree')
        power=energy/len(x);delta=abs(power-c['mean_power'])/max(c['mean_power'],1e-30)
        if power<=0 or delta>1e-10:raise ValueError('Power replay mismatch')
        max_power_delta=max(max_power_delta,delta)
        identifiers.append({k:v for k,v in c.items() if k!='cache_path'})
        write(args.output/'PROGRESS.json',dict(stage='CACHE_REPLAY',pid=os.getpid(),time=time.time(),
            completed_clips=i+1,total_clips=len(clips),elapsed_seconds=time.time()-started,gpu_used=False))
    if any(len(v)!=1 for v in roles.values()):raise ValueError('Recording pack crosses roles')
    training=schedule(clips,'train_pack',50,80);validation=schedule(clips,'validation_pack',1,4)
    assert len(training)==120000 and len(validation)==120
    assert len(training)//50==2400 and 2400//32==75
    for rows,role in ((training,'train_pack'),(validation,'validation_pack')):
        assert all(clips[int(r[n])]['role']==role for r in rows for n in ('first','second'))
    np.save(args.output/'TRAIN_SCHEDULE.npy',training,allow_pickle=False)
    np.save(args.output/'VALIDATION_SCHEDULE.npy',validation,allow_pickle=False)
    write(args.output/'CLIP_INDEX.json',dict(status='DEVELOPMENT_CANDIDATE_NOT_FINAL_ADMISSION',clips=identifiers))
    result=dict(status='CPU_CACHE_REPLAY_AND_MIXTURE_SCHEDULE_COMPLETE',time=time.time(),
        source_manifest_sha256=sha(args.manifest),cache_clips_verified=len(clips),max_relative_power_replay_error=max_power_delta,
        training_examples=len(training),epochs=50,examples_per_epoch=2400,effective_batch=32,updates_per_arm=3750,
        validation_cases=len(validation),sir_conditions_db=[-10,0,10],window_samples=LENGTH,
        full_context_samples=2097152,seed=0,power_normalization='per long component; preserve local fluctuations',
        reference_order='canonical aircraft category; identical schedule for fixed-order and PIT arms',
        frequency_interpretation='synthetic frequency-rebased complex baseband; not actual joint RF capture',
        files={name:sha(args.output/name) for name in ['TRAIN_SCHEDULE.npy','VALIDATION_SCHEDULE.npy','CLIP_INDEX.json']},
        worker_source_sha256=sha(__file__),elapsed_seconds=time.time()-started,
        training_admitted=False,gpu_training_started=False,model_performance_evaluated=False,heldout_read=False)
    write(args.output/'COMPLETE.json',result);write(args.output/'PROGRESS.json',dict(stage='COMPLETE',**result))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cpus',default='')
    args=parser.parse_args()
    try:run(args)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
