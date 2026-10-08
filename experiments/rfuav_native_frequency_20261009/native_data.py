"""Independent adapter for the RF-coordinate study; old loaders stay frozen."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from native import BANDS, FS, LENGTH, WINDOW, GUARD, crop_start, band_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0, str(LEGACY));sys.path.insert(0, str(LEGACY/'vendor'))
from drone_rf.data import ScheduledMixtures, DEVELOPMENT_CATEGORIES, sha256
from drone_rf.context_data import component_gains, mixture_context_features
from drone_rf.context_training_data import DTYPE, write_json


class NativeLibrary(ScheduledMixtures):
    def __init__(self, manifest, role):
        if role not in ('train_pack','validation_pack'):
            raise ValueError('Heldout excluded')
        value=json.loads(Path(manifest).read_text())
        self.clips=value['clips'];self.role=role;self.max_open=6
        self._maps,self._verified=OrderedDict(),{}
        packs={};ids=set()
        for c in self.clips:
            if (c['samples']!=LENGTH or c['fs_hz']!=FS or c['category'] not in DEVELOPMENT_CATEGORIES
                    or c['role'] not in ('train_pack','validation_pack') or c['clip_id'] in ids
                    or not np.isfinite(c['mean_power']) or c['mean_power']<=0
                    or c['common_center_hz']!=BANDS[band_for(c['center_hz'])]['center_hz']):
                raise ValueError('Invalid native source metadata')
            if packs.setdefault(c['pack_id'],c['role'])!=c['role']:
                raise ValueError('Pack leakage')
            ids.add(c['clip_id'])

    def _array(self,index):
        if self.clips[index]['role']!=self.role:
            raise ValueError('Requested wrong source role')
        return super()._array(index)


class NativeMixtures:
    def __init__(self, preparation, role, epoch=1, use_features=True):
        self.preparation=Path(preparation)
        self.config=json.loads((self.preparation/'PREPARATION.json').read_text())
        if self.config['status']!='NATIVE_RF_COORDINATE_PREPARATION':
            raise ValueError('Wrong native preparation')
        manifest=self.preparation/'NATIVE_MANIFEST.json'
        if sha256(manifest)!=self.config['manifest_sha256']:
            raise ValueError('Native manifest changed')
        self.library=NativeLibrary(manifest,role)
        name='TRAIN.npy' if role=='train_pack' else 'VALIDATION.npy'
        path=Path(self.config['original_preparation'])/name
        if sha256(path)!=self.config['schedule_sha256'][name]:
            raise ValueError('Original schedule changed')
        all_rows=np.load(path,allow_pickle=False)
        if all_rows.dtype!=DTYPE:
            raise ValueError('Invalid native schedule')
        self.rows=all_rows[all_rows['epoch']==epoch].copy()
        if not len(self.rows):
            raise ValueError('Empty epoch')
        self.original_rows_hash=hashlib.sha256(self.rows.tobytes()).hexdigest()
        self.original_starts=self.rows['crop_start'].copy()
        for row in self.rows:
            count=int(row['count'])
            if count not in (1,2,3) or np.any(row['indices'][count:]!=-1):
                raise ValueError('Invalid count/padding')
            clips=[self.library.clips[int(i)] for i in row['indices'][:count]]
            if (any(c['role']!=role for c in clips) or len({c['category'] for c in clips})!=count
                    or len({c['common_center_hz'] for c in clips})!=1):
                raise ValueError('Role, category or RF band mismatch')
            row['crop_start']=crop_start(int(row['crop_start']))
        self.role,self.epoch,self.length=role,epoch,WINDOW
        self.rows_hash=hashlib.sha256(self.rows.tobytes()).hexdigest()
        self.features=None
        if use_features:
            receipt=json.loads(self.cache_stem.with_suffix('.json').read_text())
            if (receipt['status']!='FEATURES_READY' or receipt['rows_sha256']!=self.rows_hash
                    or receipt['preparation_sha256']!=sha256(self.preparation/'PREPARATION.json')):
                raise ValueError('Unsealed native features')
            path=self.cache_stem.with_suffix('.npy')
            if sha256(path)!=receipt['features_sha256']:
                raise ValueError('Native mixture features changed')
            self.features=np.load(path,mmap_mode='r',allow_pickle=False)
            if self.features.shape!=(len(self.rows),65,255) or self.features.dtype!=np.float32:
                raise ValueError('Invalid native feature shape')

    @property
    def cache_stem(self):
        return self.preparation/'features'/f'{self.role}_{self.epoch:03d}'

    def __len__(self):
        return len(self.rows)

    def full_example(self,index):
        from drone_rf.context_data import contextual_mixture
        row=self.rows[index];count=int(row['count']);indices=row['indices'][:count]
        return contextual_mixture([self.library._array(int(i)) for i in indices],
            [self.library.clips[int(i)]['mean_power'] for i in indices],row['levels'][:count],
            row['phases'][:count],int(row['crop_start']),self.length)

    def __getitem__(self,index):
        if self.features is None:
            raise ValueError('Mixture-only features not ready')
        row=self.rows[index];count=int(row['count']);indices=row['indices'][:count]
        gains=component_gains([self.library.clips[int(i)]['mean_power'] for i in indices],
                             row['levels'][:count],row['phases'][:count])
        refs=np.zeros((3,self.length),np.complex64);start=int(row['crop_start'])
        for j,(i,gain) in enumerate(zip(indices,gains)):
            refs[j]=(self.library._array(int(i))[start:start+self.length]*gain).astype(np.complex64)
        return dict(mixture=refs.sum(0),references=refs,active=np.any(refs!=0,axis=1),
            context_features=np.array(self.features[index]),crop_start=start,
            construction_count=count,physical_count_eligible=False,index=int(index))
