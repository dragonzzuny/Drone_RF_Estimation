"""Stream-hash admitted RF files, cache long raw clips, then audit split similarity.

No torch/CUDA import, no controller or heldout waveform read. Run beside GPU
training at low CPU/I/O priority. All outputs are local research artifacts.
"""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

# Set before NumPy initialization, independent of the GPU trainer's environment.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_rf.similarity import shifted_coherence
from inventory_rfuav import AIRCRAFT

CLIP_SAMPLES=2_097_152
BLOCK_BYTES=8*1024**2
COARSE_SAMPLES=8192
LAG=1024
FLAG_RHO_SQUARED=.98


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while raw:=f.read(BLOCK_BYTES):h.update(raw)
    return h.hexdigest()


def admitted_files(inventory):
    rows=[]
    for row in inventory['files']:
        if not row['aircraft_category'] or row['role']=='heldout_model_not_for_selection':continue
        if row['category'] not in AIRCRAFT or row['role'] not in {'train_pack','validation_pack'}:
            raise ValueError('Controller, heldout, or unexpected source role')
        if row['data_type']!='Complex Float' or row['fs_hz']!=100_000_000 or row['bytes']%8:
            raise ValueError('Unsupported raw IQ declaration')
        if row['samples_cf32']<4*CLIP_SAMPLES:
            raise ValueError('Insufficient length for three disjoint long clips')
        rows.append(row)
    if not rows:raise ValueError('Empty aircraft development cohort')
    roles=collections.defaultdict(set)
    for r in rows:roles[r['pack_id']].add(r['role'])
    if any(len(v)!=1 for v in roles.values()):raise ValueError('Pack crosses roles')
    if len({r['path'] for r in rows})!=len(rows):raise ValueError('Repeated source path')
    return sorted(rows,key=lambda r:r['relative_path'])


def extract_stream(path, offsets, length, expected_bytes, callback=None):
    """Whole-file SHA and exact selected intervals in one sequential read pass."""
    h=hashlib.sha256();buffers=[bytearray() for _ in offsets];cursor=0
    with Path(path).open('rb',buffering=0) as f:
        if hasattr(os,'posix_fadvise'):
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_SEQUENTIAL)
        while raw:=f.read(BLOCK_BYTES):
            h.update(raw)
            for buf, offset in zip(buffers,offsets):
                start,end=offset*8,(offset+length)*8
                a,b=max(start,cursor),min(end,cursor+len(raw))
                if a<b:buf.extend(raw[a-cursor:b-cursor])
            cursor+=len(raw)
            if callback:callback(cursor)
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(f.fileno(),max(0,cursor-len(raw)),len(raw),os.POSIX_FADV_DONTNEED)
    if cursor!=expected_bytes or any(len(b)!=length*8 for b in buffers):
        raise ValueError('Short/changed source or incomplete clip')
    return h.hexdigest(),buffers


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0


def run(args):
    args.output.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    with (args.output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'CONFIG.json').exists():
            raise RuntimeError('Run already exists; recovery must preserve completed source records')
        if args.cpus:
            cpus={int(n) for n in args.cpus.split(',')}
            if not cpus or not cpus<=os.sched_getaffinity(0):raise ValueError('CPU affinity unavailable')
            os.sched_setaffinity(0,cpus)
        os.nice(10)
        started=time.time();last_write=0
        inventory_sha=sha(args.inventory);inventory=json.loads(args.inventory.read_text())
        files=admitted_files(inventory);total_bytes=sum(r['bytes'] for r in files)
        expected_cache_bytes=len(files)*3*CLIP_SAMPLES*8
        if shutil.disk_usage(args.cache).free<expected_cache_bytes+2*1024**3:
            raise RuntimeError('Insufficient cache space')
        source_files=[Path(__file__),Path(__file__).with_name('inventory_rfuav.py'),
                      Path(__file__).resolve().parents[1]/'src/drone_rf/similarity.py']
        config=dict(started=started,pid=os.getpid(),inventory_sha256=inventory_sha,
            source_sha256={str(p):sha(p) for p in source_files},file_count=len(files),
            total_bytes=total_bytes,clip_samples=CLIP_SAMPLES,clips_per_file=3,
            offsets='floor((N-L)*q/4), q=1,2,3',expected_cache_bytes=expected_cache_bytes,
            cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),
            cuda_imported=False,heldout_read=False,coarse_samples=COARSE_SAMPLES,max_lag=LAG,
            flag_rho_squared=FLAG_RHO_SQUARED,independent_recording_proof=False)
        write(args.output/'CONFIG.json',config)
        sources=[];clips=[];completed_bytes=0

        def progress(stage,force=False,**kw):
            nonlocal last_write
            now=time.time()
            if not force and now-last_write<3:return
            row=dict(stage=stage,pid=os.getpid(),time=now,elapsed_seconds=now-started,
                completed_files=len(sources),total_files=len(files),completed_bytes=completed_bytes,
                total_bytes=total_bytes,cached_clips=len(clips),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_used=False,**kw)
            write(args.output/'PROGRESS.json',row);last_write=now

        progress('HASH_AND_CACHE',force=True)
        for index,row in enumerate(files):
            path=Path(row['path']);before=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):
                raise RuntimeError(f'Source changed: {row["relative_path"]}')
            while memory_available()<2*1024**3:
                progress('WAIT_MEMORY',force=True);time.sleep(10)
            length=CLIP_SAMPLES;offsets=[(row['samples_cf32']-length)*q//4 for q in (1,2,3)]
            def callback(amount):
                progress('HASH_AND_CACHE',current_file_index=index,current_file=row['relative_path'],
                         current_file_bytes_read=amount,bytes_read=completed_bytes+amount)
            digest,buffers=extract_stream(path,offsets,length,row['bytes'],callback)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Raw file changed during reading')
            record=dict(path=row['relative_path'],pack_id=row['pack_id'],category=row['category'],
                role=row['role'],bytes=row['bytes'],sha256=digest,clips=[])
            for slot,(offset,raw) in enumerate(zip(offsets,buffers)):
                data=np.frombuffer(raw,dtype='<c8')
                if not np.isfinite(data).all():raise ValueError('Nonfinite cached IQ')
                clip_id=hashlib.sha256(f'{row["relative_path"]}:{offset}:{length}'.encode()).hexdigest()[:24]
                target=args.cache/(clip_id+'.npy')
                if target.exists():raise FileExistsError(target)
                tmp=target.with_suffix('.npy.tmp')
                with tmp.open('wb') as out:np.save(out,data,allow_pickle=False)
                os.replace(tmp,target)
                power=np.abs(data.astype(np.complex128))**2
                mean_power=float(power.mean());dc=float(abs(np.mean(data.astype(np.complex128)))**2)
                local=power.reshape(-1,16384).mean(1)
                c=dict(clip_id=clip_id,cache_path=str(target),source_path=row['relative_path'],
                    source_sha256=digest,pack_id=row['pack_id'],category=row['category'],role=row['role'],
                    fs_hz=row['fs_hz'],center_hz=row['center_hz'],offset_samples=offset,samples=length,
                    cache_sha256=sha(target),raw_clip_sha256=hashlib.sha256(raw).hexdigest(),
                    mean_power=mean_power,dc_power_fraction=dc/max(mean_power,1e-30),
                    local_power_quantiles=np.quantile(local,[.05,.5,.95]).tolist(),
                    normalization_applied=False,resampling_applied=False,finite=True)
                clips.append(c);record['clips'].append(clip_id)
                del data,power,local
            del buffers,raw
            sources.append(record);completed_bytes+=row['bytes']
            with (args.output/'SOURCE_RECORDS.jsonl').open('a') as out:
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
            write(args.output/'CACHE_MANIFEST.json',dict(status='BUILDING',clips=clips,inventory_sha256=inventory_sha))
            progress('HASH_AND_CACHE',force=True)
            print(json.dumps(dict(files=len(sources),total=len(files),bytes_read=completed_bytes,
                cached_clips=len(clips),elapsed_seconds=time.time()-started)),flush=True)

        groups=collections.defaultdict(list)
        for r in sources:groups[r['sha256']].append(r)
        duplicates=[[dict(path=r['path'],role=r['role'],pack_id=r['pack_id']) for r in g]
                     for g in groups.values() if len(g)>1]
        crossing=[g for g in duplicates if len({r['role'] for r in g})>1]
        write(args.output/'EXACT_DUPLICATES.json',dict(groups=duplicates,cross_role_groups=crossing))
        write(args.output/'CACHE_MANIFEST.json',dict(status='RAW_CACHE_COMPLETE_PENDING_SPLIT_REVIEW',
            clips=clips,inventory_sha256=inventory_sha,source_count=len(sources),raw_source_bytes_hashed=completed_bytes,
            exact_cross_role_duplicate_groups=len(crossing),training_admitted=False))

        progress('SHIFTED_SIMILARITY',force=True)
        train=[c for c in clips if c['role']=='train_pack']
        validation=[c for c in clips if c['role']=='validation_pack']
        waves={c['clip_id']:np.load(c['cache_path'],mmap_mode='r',allow_pickle=False)[:COARSE_SAMPLES].copy()
               for c in clips}
        comparisons=0;flags=[];maximum=0.;per_validation=[]
        with (args.output/'SHIFTED_PAIRS.csv').open('w',newline='') as out:
            columns=['train_clip','validation_clip','same_category','rho_squared','lag_samples','overlap_samples','variance_defined']
            writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
            for v in validation:
                best=None
                for t in train:
                    stat=shifted_coherence(waves[t['clip_id']],waves[v['clip_id']],LAG)
                    result=dict(train_clip=t['clip_id'],validation_clip=v['clip_id'],
                        same_category=t['category']==v['category'],**stat)
                    writer.writerow(result);comparisons+=1;maximum=max(maximum,stat['rho_squared'])
                    if best is None or stat['rho_squared']>best['rho_squared']:best=result
                    if stat['rho_squared']>=FLAG_RHO_SQUARED:
                        tx=np.load(t['cache_path'],mmap_mode='r',allow_pickle=False)[:63872]
                        vx=np.load(v['cache_path'],mmap_mode='r',allow_pickle=False)[:63872]
                        flags.append(dict(**result,full_window=shifted_coherence(tx,vx,LAG)))
                per_validation.append(best)
                progress('SHIFTED_SIMILARITY',comparisons=comparisons,total_comparisons=len(train)*len(validation))
        write(args.output/'SIMILARITY_FLAGS.json',dict(flags=flags,best_per_validation=per_validation,
            window_samples=COARSE_SAMPLES,max_lag=LAG,flag_threshold_squared=FLAG_RHO_SQUARED,
            high_correlation_is_not_proof_of_copy=True))
        if sha(args.inventory)!=inventory_sha:raise RuntimeError('Inventory changed during run')
        if any(sha(p)!=h for p,h in config['source_sha256'].items()):raise RuntimeError('Worker source changed during run')
        summary=dict(status='CPU_PREPARATION_COMPLETE_REVIEW_REQUIRED',time=time.time(),
            files_hashed=len(sources),bytes_hashed=completed_bytes,cache_clips=len(clips),
            cached_samples=len(clips)*CLIP_SAMPLES,role_counts=dict(collections.Counter(c['role'] for c in clips)),
            exact_duplicate_groups=len(duplicates),exact_cross_role_duplicate_groups=len(crossing),
            shifted_pairs=comparisons,max_coarse_rho_squared=maximum,flagged_pairs=len(flags),
            high_full_window_pairs=sum(f['full_window']['rho_squared']>=FLAG_RHO_SQUARED for f in flags),
            elapsed_seconds=time.time()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_used=False,heldout_payload_read=False,training_admitted=False,model_performance_evaluated=False)
        write(args.output/'COMPLETE.json',summary);progress('COMPLETE',force=True)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cpus',default='')
    args=parser.parse_args()
    try:run(args)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),traceback=traceback.format_exc()))
        raise
