"""Deterministic complex mixtures from the pinned RFUAV development schedule.

Normalize with each complete cached context's power, then crop. In particular,
never equalize the individual short windows. Source receiver noise remains in
its recorded component; a zero background target does not mean noise-free RF.
This loader prepares candidate data and does not authorize a training run.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np


DEVELOPMENT_CATEGORIES = {
    'DJI AVATA2', 'DJI FPV COMBO', 'DJI MAVIC3 PRO', 'DJI MINI3', 'DJI MINI4 PRO',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def long_context_mixture(first, second, powers, sir_db, phases):
    """SIR = 10 log10(P_first / P_second) in the *long* contexts.

    The common scale makes nominal total component power one; actual mixture
    power also contains the complex cross term. No target-dependent adjustment
    is made after selecting the short windows.
    """
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError('Two equally sized nonempty complex windows required')
    if not np.iscomplexobj(first) or not np.iscomplexobj(second):
        raise ValueError('Complex I/Q required')
    if not np.isfinite([*powers, sir_db, *phases]).all() or min(powers) <= 0:
        raise ValueError('Invalid power or mixing parameters')
    ratio = 10. ** (float(sir_db) / 10.)
    gains = np.sqrt(np.array([ratio, 1.]) / ((1. + ratio) * np.array(powers)))
    gains = gains * np.exp(1j * np.array(phases))
    sources = np.stack([first * gains[0], second * gains[1]]).astype(np.complex64)
    if not np.isfinite(sources).all():
        raise ValueError('Nonfinite synthesized I/Q')
    return sources.sum(axis=0), sources


class ScheduledMixtures:
    """Map-style CPU dataset; forward models receive only sample['mixture'].

    Full cache hashes are checked on first use in each worker. Subsequent reads
    check stat identity; mmap LRU bounds open files. All caches remain read-only.
    """
    def __init__(self, manifest_path, schedule_dir, role, max_open=8):
        if role not in {'train_pack', 'validation_pack'} or max_open < 1:
            raise ValueError('Development role and positive mmap limit required')
        directory = Path(schedule_dir)
        receipt = json.loads((directory / 'COMPLETE.json').read_text())
        if receipt['status'] != 'CPU_CACHE_REPLAY_AND_MIXTURE_SCHEDULE_COMPLETE':
            raise ValueError('Incomplete schedule preparation')
        if sha256(manifest_path) != receipt['source_manifest_sha256']:
            raise ValueError('Manifest changed since cache replay')
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest['exact_cross_role_duplicate_groups']:
            raise ValueError('Source copies cross roles')
        name = 'TRAIN_SCHEDULE.npy' if role == 'train_pack' else 'VALIDATION_SCHEDULE.npy'
        for filename in (name, 'CLIP_INDEX.json'):
            if sha256(directory / filename) != receipt['files'][filename]:
                raise ValueError('Pinned schedule or clip index changed')
        self.clips = manifest['clips']
        expected = json.loads((directory / 'CLIP_INDEX.json').read_text())['clips']
        if [{k: v for k, v in c.items() if k != 'cache_path'} for c in self.clips] != expected:
            raise ValueError('Clip index does not describe this manifest')
        packs = {}
        ids = set()
        for clip in self.clips:
            if clip['category'] not in DEVELOPMENT_CATEGORIES or clip['role'] not in {'train_pack', 'validation_pack'}:
                raise ValueError('Controller, held-out, or unexpected category')
            if clip['clip_id'] in ids:
                raise ValueError('Duplicate clip ID')
            ids.add(clip['clip_id'])
            if packs.setdefault(clip['pack_id'], clip['role']) != clip['role']:
                raise ValueError('Recording group crosses roles')
            if not np.isfinite(clip['mean_power']) or clip['mean_power'] <= 0:
                raise ValueError('Invalid context power')
        self.rows = np.load(directory / name, allow_pickle=False)
        self.window_samples = int(receipt['window_samples'])
        self.role, self.max_open = role, max_open
        self._maps, self._verified = OrderedDict(), {}
        for row in self.rows:
            a, b = int(row['first']), int(row['second'])
            if not (0 <= a < len(self.clips) and 0 <= b < len(self.clips)):
                raise ValueError('Clip index outside manifest')
            first, second = self.clips[a], self.clips[b]
            if first['role'] != role or second['role'] != role or first['category'] >= second['category']:
                raise ValueError('Role leakage or noncanonical source pair')
            if first['fs_hz'] != second['fs_hz']:
                raise ValueError('Sample rates differ; explicit resampling required')
            start = int(row['crop_start'])
            if start < 0 or start + self.window_samples > min(first['samples'], second['samples']):
                raise ValueError('Crop outside source context')
            if not np.isfinite([row['sir_db'], row['phase_first'], row['phase_second']]).all():
                raise ValueError('Nonfinite mixture parameters')

    def __len__(self):
        return len(self.rows)

    def _array(self, index):
        clip = self.clips[index]
        path = Path(clip['cache_path'])
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if index in self._verified and self._verified[index] != identity:
            raise ValueError('Source cache changed during use')
        if index not in self._verified:
            if sha256(path) != clip['cache_sha256']:
                raise ValueError('Source cache hash mismatch')
            self._verified[index] = identity
        if index not in self._maps:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.dtype != np.dtype('complex64') or array.shape != (clip['samples'],):
                raise ValueError('Unexpected I/Q cache format')
            self._maps[index] = array
            while len(self._maps) > self.max_open:
                self._maps.popitem(last=False)
        self._maps.move_to_end(index)
        return self._maps[index]

    def __getitem__(self, index):
        row = self.rows[index]
        indices = [int(row['first']), int(row['second'])]
        clips = [self.clips[i] for i in indices]
        start = int(row['crop_start'])
        windows = [self._array(i)[start:start + self.window_samples] for i in indices]
        mixture, sources = long_context_mixture(*windows,
            [c['mean_power'] for c in clips], float(row['sir_db']),
            [float(row['phase_first']), float(row['phase_second'])])
        # Nonzero waveform is a numerical loss flag, NOT a drone-count label.
        active = np.any(sources != 0, axis=1)
        return dict(mixture=mixture, references=sources, active=active,
            background_reference=np.zeros_like(mixture), index=int(index),
            epoch=int(row['epoch']), nominal_sir_db=float(row['sir_db']),
            categories=tuple(c['category'] for c in clips),
            pack_ids=tuple(c['pack_id'] for c in clips),
            fs_hz=clips[0]['fs_hz'],
            center_frequencies_hz=tuple(c['center_hz'] for c in clips))
