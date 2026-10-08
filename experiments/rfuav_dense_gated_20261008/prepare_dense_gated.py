"""Prepare ONE dense-data comparison: baseline U-Net vs gated U-Net.

CPU only, two bounded workers. Does not alter the prior corpus or GPU queue.
Both models receive the exact same new mixtures and old validation examples.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from dense_data import DenseContextMixtures, DenseLibrary
from drone_rf.context_training_data import write_json
from drone_rf.context_data import contextual_mixture
from drone_rf.data import sha256
from drone_rf.mixture_constraints import validate_sources


LOCAL = Path('/home/pyj/문서/GitHub/Drone_RF_Estimation/local')
OLD = LOCAL / 'within_dataset_context_20261007_v1/preparation'
DENSE = LOCAL / 'dense_training_corpus_20261007_v1/audit'
CHECKPOINT = LOCAL / 'within_dataset_context_20261007_v1/run/mean/SELECTED_050.pt'


def build_preparation(root):
    root.mkdir(parents=True, exist_ok=True)
    ready = root / 'PREPARATION.json'
    if ready.exists():
        raise ValueError('Preparation already exists; use --features-only to resume')
    receipt = json.loads((DENSE / 'COMPLETE.json').read_text())
    if receipt['status'] != 'DENSE_TRAIN_CORPUS_READY_FOR_REVIEW':
        raise ValueError('Incomplete dense corpus')
    for name, digest in receipt['files'].items():
        if sha256(DENSE / name) != digest:
            raise ValueError('Dense intake audit changed: ' + name)
    if any(receipt[k] for k in ('exact_cross_role_clip_copies', 'zero_power_contexts', 'similarity_flags', 'heldout_iq_read')):
        raise ValueError('Dense intake failed an admission check')
    config = json.loads((OLD / 'PREPARATION.json').read_text())
    old_manifest = json.loads(Path(config['manifest']).read_text())
    manifest = json.loads((DENSE / 'CACHE_MANIFEST.json').read_text())
    if manifest['parent_manifest_sha256'] != sha256(config['manifest']):
        raise ValueError('Different parent split')
    clips, old_clips = manifest['clips'], old_manifest['clips']
    if Counter(c['role'] for c in clips) != {'train_pack': 3902, 'validation_pack': 144}:
        raise ValueError('Unexpected dense cohort')
    old_validation = {c['clip_id']: c for c in old_clips if c['role'] == 'validation_pack'}
    if old_validation != {c['clip_id']: c for c in clips if c['role'] == 'validation_pack'}:
        raise ValueError('Validation source cache entries changed')
    if {c['pack_id'] for c in clips if c['role'] == 'train_pack'} != {c['pack_id'] for c in old_clips if c['role'] == 'train_pack'}:
        raise ValueError('Training pack split changed')
    # Hash-checked parent schedule; preserve class/count/SIR/phase/crop sequence.
    for name in ('TRAIN.npy', 'VALIDATION.npy'):
        if sha256(OLD / name) != config['files'][name]:
            raise ValueError('Parent schedule changed')
    train = np.load(OLD / 'TRAIN.npy', allow_pickle=False)
    train = train[train['epoch'] <= 5].copy()
    validation = np.load(OLD / 'VALIDATION.npy', allow_pickle=False).copy()
    groups = defaultdict(list)
    for i, clip in enumerate(clips):
        if clip['role'] == 'train_pack':
            groups[clip['category']].append(i)
    rng = np.random.default_rng(20261008)
    decks, positions = {}, defaultdict(int)
    used = set()
    for row in train:
        for j in range(int(row['count'])):
            old_clip = old_clips[int(row['indices'][j])]
            category = old_clip['category']
            if category not in decks or positions[category] == len(decks[category]):
                decks[category] = rng.permutation(groups[category])
                positions[category] = 0
            index = int(decks[category][positions[category]])
            positions[category] += 1
            if clips[index]['center_hz'] != old_clip['center_hz']:
                raise ValueError('Receiver center changed within category')
            row['indices'][j] = index
            used.add(index)
        validate_sources([dict(clips[int(i)], dataset='RFUAV') for i in row['indices'][:int(row['count'])]])
    index_by_id = {c['clip_id']: i for i, c in enumerate(clips)}
    for row in validation:
        for j in range(int(row['count'])):
            row['indices'][j] = index_by_id[old_clips[int(row['indices'][j])]['clip_id']]
    manifest_path = root / 'CACHE_MANIFEST.json'
    shutil.copyfile(DENSE / 'CACHE_MANIFEST.json', manifest_path)
    np.save(root / 'TRAIN.npy', train, allow_pickle=False)
    np.save(root / 'VALIDATION.npy', validation, allow_pickle=False)
    stage = json.loads((CHECKPOINT.parent / 'STAGE_050.json').read_text())
    if stage['selected_epoch'] != 22 or sha256(CHECKPOINT) != stage['checkpoint_sha256']:
        raise ValueError('Warm-start checkpoint failed verification')
    config.update(status='DENSE_GATED_COMPARISON_ADMITTED', manifest=str(manifest_path.resolve()),
        manifest_sha256=sha256(manifest_path), epochs=5, comparison_arms=['unet_mean', 'unet_gated'],
        arms=['unet_mean', 'unet_gated'], updates_per_arm=375, checkpoints=[1, 5],
        dense_corpus_used=True, initialization_checkpoint=str(CHECKPOINT),
        initialization_checkpoint_sha256=stage['checkpoint_sha256'], initialization_selected_epoch=22,
        initialization='same prior mean-U-Net e22 checkpoint; new gate adapter initially identity',
        optimizer_reset=True, learning_rate=1e-5, training_clips=3902, validation_clips=144,
        dense_scheduled_unique_training_clips=len(used), dense_parent_admission=str((DENSE / 'COMPLETE.json')),
        admission_scope='same-band development architecture comparison; no claim of physical-aircraft count',
        base_schedule=None, gpu_training_started=False,
        files={n: sha256(root / n) for n in ('TRAIN.npy', 'VALIDATION.npy')})
    write_json(ready, config)
    # Reuse validation FEATURES bit-for-bit, remapping only manifest indices.
    feature_dir = root / 'features'
    feature_dir.mkdir(exist_ok=True)
    old_stem = OLD / 'features/validation_pack_001'
    old_receipt = json.loads(old_stem.with_suffix('.json').read_text())
    if sha256(old_stem.with_suffix('.npy')) != old_receipt['feature_sha256']:
        raise ValueError('Validation feature hash changed')
    stem = feature_dir / 'validation_pack_001'
    shutil.copyfile(old_stem.with_suffix('.npy'), stem.with_suffix('.npy'))
    write_json(stem.with_suffix('.json'), dict(old_receipt,
        rows_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
        reused_features_from=str(old_stem), reuse='exact same recordings, gains, phase and crop; indices remapped'))
    # Validate the entire new schedules without reading any IQ payload.
    for epoch in range(1, 6):
        DenseContextMixtures(root, 'train_pack', epoch, use_features=False)
    DenseContextMixtures(root, 'validation_pack', 1)
    result = dict(status='DENSE_TWO_ARM_SCHEDULE_READY', training_clips=3902,
        scheduled_unique_training_clips=len(used), examples=12000, epochs=5, validation_examples=630,
        validation_features_unchanged=True, heldout_iq_read=False, raw_iq_read=False,
        mixture_policy=config['mixture_policy'], native_center_offsets_preserved=False,
        checkpoint_sha256=stage['checkpoint_sha256'], gpu_training_started=False)
    write_json(root / 'SCHEDULE_CHECK.json', result)
    print(result, flush=True)


def init_worker(manifest):
    global LIBRARY
    LIBRARY = DenseLibrary(manifest, 'train_pack')


def extract(task):
    index, row = task
    count = int(row['count'])
    indices = row['indices'][:count]
    clips = [LIBRARY.clips[int(i)] for i in indices]
    result = contextual_mixture([LIBRARY._array(int(i)) for i in indices],
        [c['mean_power'] for c in clips], row['levels'][:count], row['phases'][:count], int(row['crop_start']))
    return index, result['context_features']


def prepare_features(root):
    config = json.loads((root / 'PREPARATION.json').read_text())
    started = time.time()
    # Explicit spawn; no inherited mutable model or GPU state, two CPU workers.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker, initargs=(config['manifest'],)) as pool:
        for epoch in range(1, 6):
            data = DenseContextMixtures(root, 'train_pack', epoch, use_features=False)
            stem = data.cache_stem
            if stem.with_suffix('.json').exists():
                data.load_features()
                continue
            temporary = stem.with_suffix('.partial.npy')
            output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(data), 65, 256))
            # map queues only small schedule rows; worker data and mmap LRU bounded.
            for completed, (index, feature) in enumerate(pool.map(extract, enumerate(data.rows), chunksize=8), 1):
                if feature.shape != (65, 256) or not np.isfinite(feature).all():
                    raise ValueError('Invalid long mixture features')
                output[index] = feature
                if completed % 100 == 0:
                    write_json(root / 'FEATURE_PROGRESS.json', dict(stage='DENSE_FEATURE_PREPARATION_CPU',
                        epoch=epoch, completed=completed, total=len(data), elapsed_seconds=time.time()-started,
                        time=time.time(), pid=os.getpid(), gpu_training_started=False))
            output.flush()
            del output
            os.replace(temporary, stem.with_suffix('.npy'))
            write_json(stem.with_suffix('.json'), dict(status='FEATURES_READY', rows_sha256=data.rows_hash,
                feature_sha256=sha256(stem.with_suffix('.npy')), feature_sources=config['feature_sources'],
                time=time.time(), component_count_task='synthetic recorded contributions, not physical aircraft'))
            print(dict(epoch=epoch, features=len(data), elapsed_seconds=time.time()-started), flush=True)
    write_json(root / 'FEATURES_COMPLETE.json', dict(status='ALL_DENSE_FEATURES_READY', epochs=5,
        training_mixtures=12000, validation_mixtures=630, elapsed_seconds=time.time()-started,
        gpu_training_started=False, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--schedule-only', action='store_true')
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[-2:]))
    os.nice(10)
    if not args.features_only:
        build_preparation(args.output)
    if not args.schedule_only:
        prepare_features(args.output)
