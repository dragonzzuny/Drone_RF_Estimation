"""Adjacent waveform windows from the unchanged admitted time-block B split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/bandwidth_timeblock_20261011'))
from data import Library, Mixtures, schedule, read, MANIFEST, LENGTH, WINDOW, ROLES
from drone_rf.context_data import component_gains, mixture_context_features
from drone_rf.data import sha256
from drone_rf.context_training_data import write_json

OUT=ROOT/'local/sequence_assignment_20261013_v1'
CHUNKS=3


def sha_rows(rows):
    return hashlib.sha256(rows.tobytes()).hexdigest()


class Sequences:
    def __init__(self, role, epoch=1, features=True):
        self.role,self.epoch=role,epoch
        config=read(OUT/'DATA_PLAN.json')
        assert sha256(MANIFEST)==config['manifest_sha256']
        self.library=Library(read(MANIFEST)['clips'],role)
        self.stem=OUT/'preparation'/f'{role}_{epoch:03d}'
        self.rows=np.load(self.stem.with_suffix('.npy'),allow_pickle=False)
        assert sha_rows(self.rows)==config['schedules'][self.stem.name]
        self.features=None
        if features:
            p=self.stem.with_suffix('.features.npy');receipt=read(self.stem.with_suffix('.features.json'))
            assert receipt['rows_sha256']==sha_rows(self.rows) and sha256(p)==receipt['features_sha256']
            self.features=np.load(p,mmap_mode='r',allow_pickle=False)
            assert self.features.shape==(len(self.rows),65,255)

    def __len__(self):
        return len(self.rows)

    def components(self,index,long=False):
        row=self.rows[index];k=int(row['count']);start=0 if long else int(row['crop_start'])
        length=LENGTH if long else WINDOW*CHUNKS
        ids=row['indices'][:k]
        gains=component_gains([self.library.clips[int(i)]['mean_power'] for i in ids],row['levels'][:k],row['phases'][:k])
        refs=np.zeros((3,length),np.complex64)
        for j,(i,g) in enumerate(zip(ids,gains)):
            clip=self.library.clips[int(i)]
            assert clip['diagnostic_role']==self.role
            refs[j]=self.library._array(int(i))[start:start+length]*g
        return refs

    def __getitem__(self,index):
        assert self.features is not None
        row=self.rows[index];start=int(row['crop_start']);refs=self.components(index)
        mixture=refs.sum(0);feature=np.array(self.features[index])
        return [dict(mixture=mixture[j*WINDOW:(j+1)*WINDOW].copy(),
            references=refs[:,j*WINDOW:(j+1)*WINDOW].copy(),
            active=np.any(refs[:,j*WINDOW:(j+1)*WINDOW]!=0,axis=1),
            context_features=feature.copy(),crop_start=start+j*WINDOW,
            construction_count=int(row['count'])) for j in range(CHUNKS)]


def register():
    OUT.mkdir(parents=True,exist_ok=True);prep=OUT/'preparation';prep.mkdir(exist_ok=True)
    assert not (OUT/'DATA_PLAN.json').exists(),'Do not overwrite registered schedules'
    clips=read(MANIFEST)['clips'];schedules={};metadata={}
    for epoch in (1,2,3):
        rows=schedule(clips,ROLES[0],100+epoch,256)
        rng=np.random.default_rng(np.random.SeedSequence([0,1013,epoch]))
        rows['crop_start']=rng.integers(0,LENGTH-CHUNKS*WINDOW+1,len(rows))
        name=f'{ROLES[0]}_{epoch:03d}';np.save(prep/f'{name}.npy',rows,allow_pickle=False)
        schedules[name]=sha_rows(rows)
    # One predetermined case in every available (count, category, level) stratum.
    dev=Mixtures(ROLES[1]);indices=[];seen=set()
    for i,row in enumerate(dev.rows):
        k=int(row['count']);key=(k,tuple(clips[int(j)]['category'] for j in row['indices'][:k]),tuple(row['levels'][:k]))
        if key not in seen:
            seen.add(key);indices.append(i)
    rows=dev.rows[indices].copy()
    rows['crop_start']=np.minimum(rows['crop_start'],LENGTH-CHUNKS*WINDOW)
    name=f'{ROLES[1]}_001';np.save(prep/f'{name}.npy',rows,allow_pickle=False);schedules[name]=sha_rows(rows)
    write_json(OUT/'DATA_PLAN.json',dict(created_time=time.time(),manifest_sha256=sha256(MANIFEST),
        source_sha256=sha256(__file__),schedules=schedules,train_epochs=3,bundles_per_epoch=768,
        waveform_windows_per_bundle=CHUNKS,waveform_samples_per_window=WINDOW,
        duration_ms=CHUNKS*WINDOW/100000.,effective_bundles_per_update=8,updates_per_epoch=96,
        dev_original_indices=indices,dev_sequences=len(indices),fixed_short_dev_rows_sha256=dev.rows_hash,
        data_scope='same recording time-block B, existing admitted groups only; not independent recording validation',
        full_context_power_normalization=True,source_center_offsets_preserved=True,seed=0))
    print('REGISTERED',len(indices),'long DEV sequences',flush=True)


def prepare(role,epoch):
    data=Sequences(role,epoch,False);path=data.stem.with_suffix('.features.npy')
    if path.exists() and data.stem.with_suffix('.features.json').exists():
        Sequences(role,epoch);return
    tmp=path.with_suffix('.partial.npy')
    features=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(data),65,255))
    began=time.time()
    for i in range(len(data)):
        refs=data.components(i,True);mixture=refs.sum(0)
        features[i]=mixture_context_features(mixture)['context_features']
        start=int(data.rows[i]['crop_start'])
        assert np.array_equal(refs[:,start:start+CHUNKS*WINDOW],data.components(i)), 'Long/short synthesis mismatch'
        if i%32==0:
            write_json(OUT/'PREPARATION_STATE.json',dict(role=role,epoch=epoch,bundles=i+1,total=len(data),pid=os.getpid()))
    features.flush();del features;os.replace(tmp,path)
    write_json(data.stem.with_suffix('.features.json'),dict(status='PASS',rows_sha256=sha_rows(data.rows),
        features_sha256=sha256(path),seconds=time.time()-began))
    print('FEATURES_READY',role,epoch,round(time.time()-began,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare'])
    a=p.parse_args()
    if a.action=='register':register()
    else:
        for role,epoch in [(ROLES[1],1)]+[(ROLES[0],i) for i in (1,2,3)]:prepare(role,epoch)
