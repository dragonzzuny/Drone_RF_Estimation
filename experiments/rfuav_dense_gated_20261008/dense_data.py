"""Admitted dense TRAINING corpus with unchanged record-disjoint validation.

Reuse the pinned project's mixing, feature transform and hash-checked mmap
reader. The new manifest/schedule adapter does not change the recorded I/Q.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from drone_rf.context_training_data import ContextMixtures, DTYPE
from drone_rf.data import DEVELOPMENT_CATEGORIES, ScheduledMixtures, sha256
from drone_rf.mixture_constraints import validate_sources


class DenseLibrary(ScheduledMixtures):
    def __init__(self, manifest, role, max_open=6):
        if role not in ('train_pack', 'validation_pack'):
            raise ValueError('Held-out access excluded')
        value = json.loads(Path(manifest).read_text())
        if value.get('exact_cross_role_duplicate_groups'):
            raise ValueError('Cross-role duplicate')
        self.clips = value['clips']
        self.role, self.max_open = role, max_open
        self._maps, self._verified = OrderedDict(), {}
        packs, ids = {}, set()
        for clip in self.clips:
            if clip['category'] not in DEVELOPMENT_CATEGORIES or clip['role'] not in ('train_pack', 'validation_pack'):
                raise ValueError('Unapproved category or role')
            if clip['clip_id'] in ids:
                raise ValueError('Duplicate clip id')
            ids.add(clip['clip_id'])
            if packs.setdefault(clip['pack_id'], clip['role']) != clip['role']:
                raise ValueError('Pack crosses train/validation')
            if (clip['samples'] != 2097152 or clip['fs_hz'] != 100000000
                    or not np.isfinite(clip['mean_power']) or clip['mean_power'] <= 0):
                raise ValueError('Invalid native context')


class DenseContextMixtures(ContextMixtures):
    def __init__(self, preparation, role, epoch=1, use_features=True):
        self.preparation = Path(preparation)
        self.config = json.loads((self.preparation / 'PREPARATION.json').read_text())
        if self.config['status'] != 'DENSE_GATED_COMPARISON_ADMITTED':
            raise ValueError('Dense preparation not admitted')
        if role not in ('train_pack', 'validation_pack') or epoch < 1:
            raise ValueError('Invalid development role/epoch')
        name = 'TRAIN.npy' if role == 'train_pack' else 'VALIDATION.npy'
        path = self.preparation / name
        if sha256(path) != self.config['files'][name]:
            raise ValueError('Dense schedule changed')
        if sha256(self.config['manifest']) != self.config['manifest_sha256']:
            raise ValueError('Dense manifest changed')
        self.library = DenseLibrary(self.config['manifest'], role)
        all_rows = np.load(path, allow_pickle=False)
        if all_rows.dtype != DTYPE:
            raise ValueError('Unexpected schedule dtype')
        self.rows = all_rows[all_rows['epoch'] == epoch].copy()
        if not len(self.rows):
            raise ValueError('No rows for epoch')
        self.role, self.epoch = role, epoch
        self.length = self.config['window_samples']
        self.features = None
        self.schedule_hash = sha256(path)
        self.rows_hash = hashlib.sha256(self.rows.tobytes()).hexdigest()
        for row in self.rows:
            count = int(row['count'])
            if count not in (1, 2, 3) or np.any(row['indices'][count:] != -1):
                raise ValueError('Invalid source count')
            indices = row['indices'][:count]
            if np.any(indices < 0) or np.any(indices >= len(self.library.clips)):
                raise ValueError('Invalid source index')
            clips = [self.library.clips[int(i)] for i in indices]
            if any(c['role'] != role for c in clips) or len({c['category'] for c in clips}) != count:
                raise ValueError('Role leak or repeated category')
            validate_sources([dict(c, dataset='RFUAV') for c in clips])
            if not 0 <= int(row['crop_start']) <= self.config['context_samples'] - self.length:
                raise ValueError('Invalid crop')
            if not np.isfinite(np.r_[row['levels'], row['phases']]).all():
                raise ValueError('Nonfinite mixture parameters')
        if use_features:
            self.load_features()
