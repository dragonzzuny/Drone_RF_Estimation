"""Seal a matched synthetic 1/2/3-component design and prepare CPU features."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from drone_rf.context_training_data import ContextMixtures, make_schedule, write_json
from drone_rf.data import ScheduledMixtures, sha256
import drone_rf.context_data as feature_module
import drone_rf.temporal as temporal_module


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    library = ScheduledMixtures(args.manifest, args.base_schedule, 'train_pack')
    ScheduledMixtures(args.manifest, args.base_schedule, 'validation_pack')
    training = make_schedule(library.clips, 'train_pack', epochs=50, examples_per_count=800)
    validation = make_schedule(library.clips, 'validation_pack', epochs=1, examples_per_count=210)
    np.save(args.output / 'TRAIN.npy', training, allow_pickle=False)
    np.save(args.output / 'VALIDATION.npy', validation, allow_pickle=False)
    config = dict(status='PINNED_SYNTHETIC_COMPONENT_EXPERIMENT', time=time.time(), seed=0,
        manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
        base_schedule=str(args.base_schedule.resolve()), epochs=50, examples_per_epoch=2400,
        effective_batch=32, updates_per_epoch=75, updates_per_arm=3750, validation_examples=630,
        window_samples=63872, context_samples=2097152, fs_hz=100000000,
        nfft=512, hop=128, stft_center=True, stft_window='sqrt_periodic_hann',
        arms=['ordered', 'mean'], context_kind='tcn', optimizer='AdamW', learning_rate=1e-4,
        weight_decay=1e-4, gradient_clip_norm=1., construction_count_loss_weight=.1,
        precision='float32', microbatch=1,
        checkpoints=[5, 10, 25, 50], validation_every_epochs=1,
        selection='minimum equally-count-weighted active component validation NMSE; include epoch0',
        count_target='number_of_synthetically_added_recorded_components_NOT_physical_aircraft',
        physical_count_eligible=False, same_category_multiple_aircraft=False,
        split_claim='original-pack-disjoint development; validation bandwidth differs; not independent dates',
        transmitter_claim='aircraft-labeled recording folders; no proof of airframe-only emission',
        controller_classes_excluded=True, heldout_models_read=False,
        mini3_pairing='unique XML in directory; pack1 XML / pack2 IQ discrepancy retained',
        power_normalization='per long component, then crop; candidate, not established improvement',
        files={n: sha256(args.output / n) for n in ('TRAIN.npy', 'VALIDATION.npy')},
        feature_sources={Path(p).name: sha256(p) for p in (feature_module.__file__, temporal_module.__file__)},
        authorizes='bounded development comparison under explicitly limited source/count/split claims',
        gpu_training_started=False)
    write_json(args.output / 'PREPARATION.json', config)
    print(json.dumps(config, ensure_ascii=False), flush=True)


def prepare_features(root, role, epoch):
    dataset = ContextMixtures(root, role, epoch, use_features=False)
    for module in (feature_module, temporal_module):
        if sha256(module.__file__) != dataset.config['feature_sources'][Path(module.__file__).name]:
            raise ValueError('Feature implementation changed after preparation')
    stem = dataset.cache_stem
    stem.parent.mkdir(exist_ok=True)
    if stem.with_suffix('.json').exists():
        dataset.load_features()
        return
    temporary = stem.with_suffix('.tmp.npy')
    features = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                        shape=(len(dataset), 65, 256))
    started = time.time()
    max_error = 0.
    for index in range(len(dataset)):
        item = dataset.full_example(index)
        features[index] = item['context_features']
        if not np.isfinite(features[index]).all():
            raise ValueError('Nonfinite long-mixture feature')
        error = float(np.abs(item['mixture'] - item['references'].sum(0)).max())
        max_error = max(max_error, error)
        if index % 20 == 0:
            write_json(root / 'FEATURE_PROGRESS.json', dict(status='PREPARING_CONTEXT_FEATURES',
                time=time.time(), role=role, epoch=epoch, example=index + 1, examples=len(dataset),
                elapsed_seconds=time.time() - started, pid=os.getpid(), gpu_used=False))
    features.flush()
    del features
    os.replace(temporary, stem.with_suffix('.npy'))
    write_json(stem.with_suffix('.json'), dict(status='FEATURES_READY', role=role, epoch=epoch,
        rows_sha256=dataset.rows_hash, feature_sha256=sha256(stem.with_suffix('.npy')),
        count=len(dataset), checked_source_clips=len(dataset.library._verified),
        max_mixture_replay_error=max_error, elapsed_seconds=time.time() - started,
        feature_sources=dataset.config['feature_sources']))


def features(args):
    root = args.output
    config = json.loads((root / 'PREPARATION.json').read_text())
    prepare_features(root, 'validation_pack', 1)
    for epoch in range(1, config['epochs'] + 1):
        while args.run and epoch > 2:
            if (args.run / 'FAILURE.json').exists():
                raise RuntimeError('Training failed; stop speculative feature preparation')
            progress = args.run / 'COMPLETED_EPOCHS.json'
            complete = json.loads(progress.read_text()) if progress.exists() else {'ordered': 0, 'mean': 0}
            if epoch <= min(complete.values()) + 2:
                break
            write_json(root / 'FEATURE_PROGRESS.json', dict(status='WAITING_TRAINING_PROGRESS',
                next_epoch=epoch, prepared_through=epoch - 1, time=time.time(), pid=os.getpid(), gpu_used=False))
            time.sleep(5)
        prepare_features(root, 'train_pack', epoch)
    write_json(root / 'FEATURE_COMPLETE.json', dict(status='ALL_CONTEXT_FEATURES_READY',
        epochs=config['epochs'], validation_examples=config['validation_examples'], time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'features'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--base-schedule', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    os.nice(10)
    os.sched_setaffinity(0, {14, 15})
    try:
        {'prepare': prepare, 'features': features}[args.action](args)
    except Exception:
        if args.output.exists():
            write_json(args.output / 'FEATURE_FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
