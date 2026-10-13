"""Explicit within-record diagnostic; does not relax the old recording loader."""
from collections import OrderedDict, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(LEGACY / 'vendor'))
from drone_rf.data import ScheduledMixtures, DEVELOPMENT_CATEGORIES, sha256
from drone_rf.context_data import contextual_mixture, component_gains
from drone_rf.context_training_data import DTYPE, strata, write_json

OUT = ROOT / 'local/wavenet_timeblock100_20261013_v1'
PREP = OUT / 'preparation'
MANIFEST = ROOT / 'local/bandwidth_evaluation_20261011_v1/TIME_BLOCK_MANIFEST.json'
FS, LENGTH, WINDOW = 100_000_000, 2_088_960, 63_872
ROLES = ('fit_block', 'within_record_dev')


def read(path):
    return json.loads(Path(path).read_text())


def validate_clips(clips):
    groups = defaultdict(list)
    ids = set()
    hashes = defaultdict(set)
    for c in clips:
        assert c['clip_id'] not in ids
        ids.add(c['clip_id'])
        assert c['category'] in DEVELOPMENT_CATEGORIES
        assert c['role'] == c['original_role'] == 'train_pack'
        assert '/VTSBW=10/' in c['source_path']
        assert c['diagnostic_role'] in (*ROLES, 'gap_excluded')
        assert c['fs_hz'] == FS and c['samples'] == LENGTH
        assert not c['normalization_applied'] and not c['resampling_applied']
        assert c['rf_filter_applied'] and np.isfinite(c['mean_power']) and c['mean_power'] > 0
        center = 2_460_000_000 if c['center_hz'] < 3e9 else 5_780_000_000
        assert c['common_center_hz'] == center
        assert c['raw_support_end_samples'] - c['raw_support_start_samples'] == LENGTH + 8192
        groups[c['category']].append(c)
        if c['diagnostic_role'] in ROLES:
            hashes[c['cache_sha256']].add(c['diagnostic_role'])
    assert set(groups) == DEVELOPMENT_CATEGORIES
    assert all(len(roles) == 1 for roles in hashes.values()), 'Cross-role identical cache'
    for values in groups.values():
        assert len({c['pack_id'] for c in values}) == 1
        fit = [c for c in values if c['diagnostic_role'] == ROLES[0]]
        dev = [c for c in values if c['diagnostic_role'] == ROLES[1]]
        assert fit and dev
        assert min(c['raw_support_start_samples'] for c in dev) - max(c['raw_support_end_samples'] for c in fit) >= FS


class Library(ScheduledMixtures):
    def __init__(self, clips, role):
        assert role in ROLES
        validate_clips(clips)
        self.clips, self.role, self.max_open = clips, role, 3
        self._maps, self._verified = OrderedDict(), {}

    def _array(self, index):
        assert 0 <= index < len(self.clips)
        assert self.clips[index]['diagnostic_role'] == self.role, 'Diagnostic role leakage'
        return super()._array(index)


def schedule(clips, role, epoch, per_count):
    assert role in ROLES and epoch >= 1
    groups = defaultdict(list)
    centers = {}
    for i, c in enumerate(clips):
        if c['diagnostic_role'] == role:
            groups[c['category']].append(i)
            assert centers.setdefault(c['category'], c['common_center_hz']) == c['common_center_hz']
    rng = np.random.default_rng(np.random.SeedSequence([0, epoch, 1011, ROLES.index(role)]))
    rows = np.zeros(per_count * 3, dtype=DTYPE)
    rows['indices'] = -1
    rows['epoch'] = epoch
    position = 0
    for count in (1, 2, 3):
        conditions = [(cats, levels) for cats, levels in strata(groups, count)
                      if len({centers[c] for c in cats}) == 1]
        assert len({tuple(c) for c, _ in conditions}) == {1:5, 2:4, 3:1}[count]
        for n in range(per_count):
            cats, levels = conditions[((epoch - 1) * per_count + n) % len(conditions)]
            row = rows[position]
            row['count'] = count
            row['indices'][:count] = [rng.choice(groups[c]) for c in cats]
            row['levels'][:count] = levels
            row['phases'][:count] = rng.uniform(-np.pi, np.pi, count)
            row['crop_start'] = rng.integers(0, LENGTH - WINDOW + 1)
            position += 1
    return rows[rng.permutation(len(rows))]


class Mixtures:
    def __init__(self, role, epoch=1, features=True):
        assert role in ROLES and (role != ROLES[1] or epoch == 1)
        config = read(PREP / 'PREPARATION.json')
        assert config['status'] == 'TIME_BLOCK_SCHEDULES_READY'
        assert sha256(MANIFEST) == config['manifest_sha256']
        self.library = Library(read(MANIFEST)['clips'], role)
        self.role, self.epoch, self.length = role, epoch, WINDOW
        path = PREP / f'{role}_{epoch:03d}.npy'
        assert sha256(path) == config['schedules'][path.name]
        self.rows = np.load(path, allow_pickle=False)
        assert self.rows.dtype == DTYPE
        assert len(self.rows) == (2400 if role == ROLES[0] else 630)
        self.rows_hash = hashlib.sha256(self.rows.tobytes()).hexdigest()
        for row in self.rows:
            k = int(row['count'])
            assert k in (1, 2, 3) and np.all(row['indices'][k:] == -1)
            assert np.all(row['indices'][:k] >= 0)
            cs = [self.library.clips[int(i)] for i in row['indices'][:k]]
            assert all(c['diagnostic_role'] == role for c in cs)
            assert len({c['category'] for c in cs}) == k
            assert len({c['common_center_hz'] for c in cs}) == 1
            assert 0 <= row['crop_start'] <= LENGTH - WINDOW
            assert np.isfinite(np.r_[row['levels'], row['phases']]).all()
        self.features = None
        if features:
            receipt = read(self.stem.with_suffix('.json'))
            assert receipt['status'] == 'FEATURES_READY' and receipt['rows_sha256'] == self.rows_hash
            assert receipt['preparation_sha256'] == sha256(PREP / 'PREPARATION.json')
            assert sha256(self.stem.with_suffix('.npy')) == receipt['features_sha256']
            self.features = np.load(self.stem.with_suffix('.npy'), mmap_mode='r', allow_pickle=False)
            assert self.features.shape == (len(self.rows), 65, 255) and self.features.dtype == np.float32

    @property
    def stem(self):
        return PREP / 'features' / f'{self.role}_{self.epoch:03d}'

    def __len__(self):
        return len(self.rows)

    def full_example(self, index):
        r = self.rows[index]
        k = int(r['count'])
        ids = r['indices'][:k]
        return contextual_mixture([self.library._array(int(i)) for i in ids],
            [self.library.clips[int(i)]['mean_power'] for i in ids], r['levels'][:k],
            r['phases'][:k], int(r['crop_start']), WINDOW)

    def __getitem__(self, index):
        assert self.features is not None
        r = self.rows[index]
        k, start = int(r['count']), int(r['crop_start'])
        ids = r['indices'][:k]
        gains = component_gains([self.library.clips[int(i)]['mean_power'] for i in ids],
                                r['levels'][:k], r['phases'][:k])
        refs = np.zeros((3, WINDOW), np.complex64)
        for j, (i, g) in enumerate(zip(ids, gains)):
            refs[j] = self.library._array(int(i))[start:start + WINDOW] * g
        return dict(mixture=refs.sum(0), references=refs, active=np.any(refs != 0, axis=1),
            context_features=np.array(self.features[index]), crop_start=start,
            construction_count=k, physical_count_eligible=False)


def prepare_features(role, epoch):
    data = Mixtures(role, epoch, features=False)
    if data.stem.with_suffix('.json').exists():
        Mixtures(role, epoch)
        return
    data.stem.parent.mkdir(exist_ok=True)
    temporary = data.stem.with_suffix('.partial.npy')
    feature = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(data), 65, 255))
    start = time.time()
    for i in range(len(data)):
        item = data.full_example(i)
        value = item['context_features']
        assert value.shape == (65, 255) and np.isfinite(value).all()
        feature[i] = value
        if i % 100 == 0:
            write_json(OUT / 'PREPARATION_STATE.json', dict(stage='FEATURES', role=role,
                epoch=epoch, examples=i+1, total=len(data), seconds=time.time()-start, pid=os.getpid()))
    feature.flush()
    del feature
    os.replace(temporary, data.stem.with_suffix('.npy'))
    write_json(data.stem.with_suffix('.json'), dict(status='FEATURES_READY', rows_sha256=data.rows_hash,
        features_sha256=sha256(data.stem.with_suffix('.npy')),
        preparation_sha256=sha256(PREP/'PREPARATION.json'), seconds=time.time()-start))
    print(f'FEATURES_READY {role} {epoch} {time.time()-start:.1f}s', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--role', choices=ROLES, required=True)
    p.add_argument('--epoch', type=int, default=1)
    args = p.parse_args()
    prepare_features(args.role, args.epoch)
