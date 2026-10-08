"""Pinned 1/2/3-component schedules and auditable context feature caches.

Synthetic component count is known from construction. It is NOT an annotation
of physical aircraft or active emitters. Physical count eligibility stays false.
"""
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from .context_data import component_gains, contextual_mixture
from .data import ScheduledMixtures, sha256


DTYPE = np.dtype([('epoch', '<i2'), ('count', 'u1'), ('indices', '<i4', (3,)),
                  ('levels', '<f4', (3,)), ('phases', '<f8', (3,)), ('crop_start', '<i4')])


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def strata(categories, count):
    levels = [[0.]] if count == 1 else ([[-10., 0.], [0., 0.], [10., 0.]] if count == 2 else
        [[0., 0., 0.]] + [[v if j == i else 0. for j in range(3)]
                          for v in (-10., 10.) for i in range(3)])
    return [(pair, power) for pair in itertools.combinations(sorted(categories), count) for power in levels]


def make_schedule(clips, role, epochs=50, examples_per_count=800, length=63872):
    if role not in ('train_pack', 'validation_pack') or epochs < 1 or examples_per_count < 1:
        raise ValueError('Invalid development schedule')
    groups = collections.defaultdict(list)
    for i, clip in enumerate(clips):
        if clip['role'] == role:
            groups[clip['category']].append(i)
    if len(groups) < 3:
        raise ValueError('Three source categories required')
    output = []
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(np.random.SeedSequence([0, epoch, 145 if role == 'train_pack' else 146]))
        rows = np.zeros(3 * examples_per_count, dtype=DTYPE)
        rows['indices'] = -1
        rows['epoch'] = epoch
        position = 0
        for count in (1, 2, 3):
            conditions = strata(groups, count)
            for number in range(examples_per_count):
                categories, levels = conditions[((epoch - 1) * examples_per_count + number) % len(conditions)]
                indices = [int(rng.choice(groups[cat])) for cat in categories]
                clips_here = [clips[i] for i in indices]
                samples = {c['samples'] for c in clips_here}
                if len(samples) != 1 or min(samples) < length or len({c['fs_hz'] for c in clips_here}) != 1:
                    raise ValueError('Mismatched or too short contexts')
                row = rows[position]
                row['count'] = count
                row['indices'][:count] = indices
                row['levels'][:count] = levels
                row['phases'][:count] = rng.uniform(-np.pi, np.pi, count)
                row['crop_start'] = rng.integers(0, min(samples) - length + 1)
                position += 1
        output.extend(rows[rng.permutation(len(rows))])
    return np.array(output, dtype=DTYPE)


class ContextMixtures:
    """Short I/Q + optionally precomputed long-mixture features for one epoch."""
    def __init__(self, preparation, role, epoch=1, use_features=True):
        self.preparation = Path(preparation)
        self.config = json.loads((self.preparation / 'PREPARATION.json').read_text())
        if self.config['status'] != 'PINNED_SYNTHETIC_COMPONENT_EXPERIMENT':
            raise ValueError('Unsealed context preparation')
        name = 'TRAIN.npy' if role == 'train_pack' else 'VALIDATION.npy'
        if role not in ('train_pack', 'validation_pack') or epoch < 1:
            raise ValueError('Invalid role/epoch')
        if sha256(self.preparation / name) != self.config['files'][name]:
            raise ValueError('Context schedule changed')
        self.library = ScheduledMixtures(self.config['manifest'], self.config['base_schedule'], role, max_open=6)
        if sha256(self.config['manifest']) != self.config['manifest_sha256']:
            raise ValueError('Context manifest changed')
        all_rows = np.load(self.preparation / name, allow_pickle=False)
        if all_rows.dtype != DTYPE:
            raise ValueError('Unexpected context schedule format')
        self.rows = all_rows[all_rows['epoch'] == epoch].copy()
        if not len(self.rows):
            raise ValueError('Epoch missing from schedule')
        self.role, self.epoch = role, epoch
        self.length = int(self.config['window_samples'])
        self.features = None
        self.schedule_hash = sha256(self.preparation / name)
        self.rows_hash = hashlib.sha256(self.rows.tobytes()).hexdigest()
        for row in self.rows:
            count = int(row['count'])
            if count not in (1, 2, 3) or np.any(row['indices'][count:] != -1):
                raise ValueError('Invalid component count or absent indices')
            indices = row['indices'][:count]
            if np.any(indices < 0) or np.any(indices >= len(self.library.clips)):
                raise ValueError('Component index outside manifest')
            clips = [self.library.clips[int(i)] for i in indices]
            if any(c['role'] != role for c in clips) or len({c['category'] for c in clips}) != count:
                raise ValueError('Role leakage or duplicate category in construction')
            if any(c['samples'] != self.config['context_samples'] or c['fs_hz'] != self.config['fs_hz'] for c in clips):
                raise ValueError('Context geometry differs from protocol')
            start = int(row['crop_start'])
            if start < 0 or start + self.length > self.config['context_samples']:
                raise ValueError('Invalid crop coordinate')
            if not np.isfinite(np.r_[row['levels'], row['phases']]).all():
                raise ValueError('Nonfinite construction parameters')
        if use_features:
            self.load_features()

    @property
    def cache_stem(self):
        return self.preparation / 'features' / f'{self.role}_{self.epoch:03d}'

    def load_features(self):
        receipt = json.loads(self.cache_stem.with_suffix('.json').read_text())
        if receipt['status'] != 'FEATURES_READY' or receipt['rows_sha256'] != self.rows_hash:
            raise ValueError('Feature cache does not match mixture schedule')
        if 'feature_sources' in self.config and receipt.get('feature_sources') != self.config['feature_sources']:
            raise ValueError('Feature cache used a different transformation')
        path = self.cache_stem.with_suffix('.npy')
        if sha256(path) != receipt['feature_sha256']:
            raise ValueError('Feature cache changed')
        self.features = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.features.shape != (len(self.rows), 65, 256) or self.features.dtype != np.float32:
            raise ValueError('Unexpected context feature geometry')

    def __len__(self):
        return len(self.rows)

    def full_example(self, index):
        row = self.rows[index]
        count = int(row['count'])
        indices = row['indices'][:count]
        clips = [self.library.clips[int(i)] for i in indices]
        return contextual_mixture([self.library._array(int(i)) for i in indices],
            [c['mean_power'] for c in clips], row['levels'][:count], row['phases'][:count],
            int(row['crop_start']), self.length)

    def __getitem__(self, index):
        if self.features is None:
            raise ValueError('Context cache must be prepared before training')
        row = self.rows[index]
        count = int(row['count'])
        clips = [self.library.clips[int(i)] for i in row['indices'][:count]]
        gains = component_gains([c['mean_power'] for c in clips], row['levels'][:count], row['phases'][:count])
        start = int(row['crop_start'])
        refs = np.zeros((3, self.length), np.complex64)
        for j, (index_here, gain) in enumerate(zip(row['indices'][:count], gains)):
            refs[j] = (self.library._array(int(index_here))[start:start + self.length] * gain).astype(np.complex64)
        return dict(mixture=refs.sum(0), references=refs, active=np.any(refs != 0, axis=1),
            context_features=np.array(self.features[index]), crop_start=start,
            construction_count=count, physical_count_eligible=False, index=int(index))
