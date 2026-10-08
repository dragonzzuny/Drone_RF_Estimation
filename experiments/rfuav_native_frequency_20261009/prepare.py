"""Bounded CPU preparation, then per-epoch readiness for concurrent GPU work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from native import BANDS,FS,GUARD,LENGTH,WINDOW,TAPS,BETA,transform,mean_power,band_for
from native_data import ROOT,LEGACY,HERE,NativeMixtures,write_json,sha256
from dense_data import DenseLibrary

ORIGINAL=Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')


def freeze(root):
    paths=[*(HERE/name for name in ('native.py','native_data.py','prepare.py','test_native.py')),
           *LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}
    originals={str(p):sha256(p) for p in [ORIGINAL/'PREPARATION.json',ORIGINAL/'CACHE_MANIFEST.json',
                                       ORIGINAL/'TRAIN.npy',ORIGINAL/'VALIDATION.npy']}
    protocol=dict(status='REGISTERED_NATIVE_REPLAY',source_sha256=sources,data_sha256=originals,
        bands=BANDS,fs_hz=FS,taps=TAPS,kaiser_beta=BETA,guard_samples_each_end=GUARD,
        context_samples=LENGTH,window_samples=WINDOW,context_tokens=255,
        transform='complex RF bandpass FIR before coherent native-center frequency translation; no resampling',
        power='normalize guarded filtered long contexts before crop; retain local variation',
        targets='band-limited recorded source contributions, including receiver noise',
        physical_aircraft_count=False,heldout_access=False,original_roles_unchanged=True,
        raw_receiver_passband_calibrated=False,
        cropped_edges='clamp only crops entering the excluded edge; record every shift',
        model='unchanged full-size mean-context complex STFT U-Net; three slots and background',
        training_plan='five epochs, same 12000 rows, seed0, batch32 micro2, 375updates, AdamW1e-5; existing incumbent warm start',
        inference_baseline='training-only mean-PSD templates with nonnegative fitting; no reference identities or count at inference',
        selection='min mean NMSE at counts2/3 including e0; report both NMSE and complex SI-SDR, weakest-source and count',
        interpretation='new observation geometry; not an architecture improvement over center-aligned scores')
    path=root/'PREP_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:
            raise RuntimeError('Preparation protocol changed')
    else:
        for name in sources:
            target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        write_json(path,protocol)
    return sha256(path)


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'PREP.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'PREP_COMPLETE.json').exists():
            raise RuntimeError('Preparation already complete')
        digest=freeze(root);started=time.time()
        output=root/'preparation';output.mkdir(exist_ok=True)
        cache=output/'native_contexts';cache.mkdir(exist_ok=True)
        libraries={role:DenseLibrary(ORIGINAL/'CACHE_MANIFEST.json',role) for role in ('train_pack','validation_pack')}
        originals=libraries['train_pack'].clips
        missing=sum(not (cache/f'{c["clip_id"]}.json').exists() for c in originals)
        if shutil.disk_usage(root).free < missing*(LENGTH*8+128)+22*1024**3:
            raise RuntimeError('Insufficient disk space with 22GiB reserve')
        clips=[]
        for i,c in enumerate(originals):
            receipt=cache/f'{c["clip_id"]}.json';path=cache/f'{c["clip_id"]}.npy'
            if receipt.exists():
                entry=json.loads(receipt.read_text())
                if (entry['prep_protocol_sha256']!=digest or entry['original_cache_sha256']!=c['cache_sha256']
                        or sha256(path)!=entry['cache_sha256']):
                    raise RuntimeError('Native cache resume hash mismatch')
            else:
                x=libraries[c['role']]._array(i)
                y=transform(x,int(c['center_hz']),int(c['offset_samples']))
                p=mean_power(y);ratio=p/mean_power(x[GUARD:-GUARD])
                if y.shape!=(LENGTH,) or not np.isfinite(y).all() or p<=0 or not 0<ratio<=1.001:
                    raise RuntimeError('Invalid filtered power or geometry')
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as stream:np.save(stream,y,allow_pickle=False)
                os.replace(tmp,path)
                entry=dict(c,cache_path=str(path.resolve()),cache_sha256=sha256(path),samples=LENGTH,
                    mean_power=p,retained_power_fraction=ratio,original_cache_sha256=c['cache_sha256'],
                    original_cache_path=c['cache_path'],original_offset_samples=c['offset_samples'],
                    offset_samples=int(c['offset_samples'])+GUARD,
                    common_center_hz=BANDS[band_for(c['center_hz'])]['center_hz'],
                    normalization_applied=False,resampling_applied=False,rf_filter_applied=True,
                    prep_protocol_sha256=digest)
                # These raw-context statistics would be stale for filtered IQ.
                for key in ('dc_power_fraction','local_power_quantiles','raw_clip_sha256'):entry.pop(key,None)
                write_json(receipt,entry)
            clips.append(entry)
            if (i+1)%25==0 or i+1==len(originals):
                state=dict(stage='NATIVE_CONTEXTS',completed=i+1,total=len(originals),time=time.time(),
                           seconds=time.time()-started,pid=os.getpid())
                write_json(root/'PREP_PROGRESS.json',state);print(json.dumps(state),flush=True)
        write_json(output/'NATIVE_MANIFEST.json',dict(clips=clips,original_manifest_sha256=sha256(ORIGINAL/'CACHE_MANIFEST.json'),
            prep_protocol_sha256=digest,heldout_read=False))
        write_json(output/'PREPARATION.json',dict(status='NATIVE_RF_COORDINATE_PREPARATION',
            manifest_sha256=sha256(output/'NATIVE_MANIFEST.json'),original_preparation=str(ORIGINAL),
            schedule_sha256={n:sha256(ORIGINAL/n) for n in ('TRAIN.npy','VALIDATION.npy')},
            prep_protocol_sha256=digest,native_center_offsets_preserved=True,bands=BANDS,
            context_samples=LENGTH,window_samples=WINDOW,fs_hz=FS,heldout_access=False))
        (output/'features').mkdir(exist_ok=True)
        for role,epoch in [('validation_pack',1)]+[('train_pack',e) for e in range(1,6)]:
            data=NativeMixtures(output,role,epoch,use_features=False)
            receipt=data.cache_stem.with_suffix('.json');path=data.cache_stem.with_suffix('.npy')
            if receipt.exists():
                NativeMixtures(output,role,epoch);continue
            tmp=path.with_suffix('.tmp')
            features=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(data),65,255))
            maximum_sum_error=0.
            for i in range(len(data)):
                item=data.full_example(i)
                if not np.array_equal(item['mixture'],item['references'].sum(0)):
                    raise RuntimeError('Full/cropped native synthesis mismatch')
                features[i]=item['context_features']
                if (i+1)%50==0 or i+1==len(data):
                    state=dict(stage='MIXTURE_FEATURES',role=role,epoch=epoch,completed=i+1,total=len(data),
                               seconds=time.time()-started,time=time.time(),pid=os.getpid())
                    write_json(root/'PREP_PROGRESS.json',state);print(json.dumps(state),flush=True)
            features.flush();del features;os.replace(tmp,path)
            edge_changed=np.flatnonzero(data.rows['crop_start']+GUARD!=data.original_starts).tolist()
            write_json(receipt,dict(status='FEATURES_READY',rows_sha256=data.rows_hash,
                original_rows_sha256=data.original_rows_hash,features_sha256=sha256(path),
                preparation_sha256=sha256(output/'PREPARATION.json'),cases=len(data),
                shifted_edge_rows=edge_changed,full_short_mixture_equality=True,time=time.time()))
        if freeze(root)!=digest:raise RuntimeError('Source changed while preparing')
        write_json(root/'PREP_COMPLETE.json',dict(status='COMPLETE',protocol_sha256=digest,
            contexts=len(clips),train_contexts=sum(c['role']=='train_pack' for c in clips),
            validation_contexts=sum(c['role']=='validation_pack' for c in clips),
            seconds=time.time()-started,time=time.time(),heldout_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();os.sched_setaffinity(0,{12,13});os.nice(10)
    try:run(args.run.resolve())
    except Exception:
        args.run.mkdir(parents=True,exist_ok=True)
        write_json(args.run/'PREP_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
