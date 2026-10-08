"""Fixed-checkpoint training/future-mixture/development-record diagnosis."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from native_data import ROOT, NativeMixtures, sha256, write_json
from phase_native import engine, native_inference_batch


def read(path):
    return json.loads(path.read_text())


def strata(data):
    return [tuple(data.library.clips[int(i)]['category'] for i in row['indices'][:int(row['count'])])
            for row in data.rows]


def mixture_key(row):
    copy = row.copy()
    copy['epoch'] = 0
    return copy.tobytes()


def register(parent, root):
    source = read(parent / 'phase/PROTOCOL.json')
    sources = dict(source['source_sha256'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(__file__)
    val = NativeMixtures(parent / 'preparation', 'validation_pack', 1)
    counts = Counter(strata(val))
    if any(n % 3 for n in counts.values()):
        raise ValueError('Cannot match category composition exactly')
    quotas = {key: n // 3 for key, n in counts.items()}
    subsets = {}
    data_hashes = dict(source['data_sha256'])
    for epoch in (1, 5):
        data = NativeMixtures(parent / 'preparation', 'train_pack', epoch)
        groups = defaultdict(list)
        for i, key in enumerate(strata(data)):
            groups[key].append(i)
        if any(len(groups[key]) < quota for key, quota in quotas.items()):
            raise ValueError('Training epoch has insufficient category strata')
        indices = sorted(i for key, quota in quotas.items() for i in groups[key][:quota])
        if len(indices) != 210 or Counter(int(data.rows[i]['count']) for i in indices) != {1:70, 2:70, 3:70}:
            raise ValueError('Wrong diagnostic composition')
        subsets[str(epoch)] = indices
        for path in (data.cache_stem.with_suffix('.json'), data.cache_stem.with_suffix('.npy')):
            data_hashes[str(path)] = sha256(path)
    config = read(parent / 'preparation/PREPARATION.json')
    train_path = Path(config['original_preparation']) / 'TRAIN.npy'
    data_hashes[str(train_path)] = sha256(train_path)
    plan = dict(status='REGISTERED_ADAPTIVE_FIT_GAP_DIAGNOSIS', source_sha256=sources, data_sha256=data_hashes,
        subsets=subsets, subset_rule='first TRAIN rows per ordered category tuple, quota=validation tuple count/3; indices sealed before model inference',
        examples_per_subset=210, count_balance=[70,70,70], category_composition_matches_validation=True,
        checkpoint='native SELECTED_005, unchanged one-pass selection', model_updates=0,
        comparisons=['TRAIN epoch1 mixtures','TRAIN epoch5 mixtures','saved630development validation mixtures'],
        exposure_rule='epoch1/5 mixtures are seen in native adaptation only if their epoch <= selected epoch; same original TRAIN recordings remain shared',
        caution='not an independent test; differences also include power/activity composition and validation recording/BW shift; no claim of proving absence of memorization',
        inference_reference_access=False, heldout_read=False, physical_aircraft_count=False)
    for rel, digest in sources.items():
        if sha256(ROOT / rel) != digest:
            raise ValueError('Changed fit-gap source')
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    write_json(root / 'PROTOCOL.json', plan)
    return plan


def verify(root, plan):
    for rel, digest in plan['source_sha256'].items():
        if sha256(ROOT / rel) != digest or sha256(root / 'source_snapshot' / rel) != digest:
            raise ValueError('Fit-gap source changed')
    for path, digest in plan['data_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Fit-gap data changed')


def main(parent):
    root = parent / 'fit_gap'
    root.mkdir(exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Refuse duplicate diagnostic')
    plan = register(parent, root)
    write_json(root / 'STATE.json', dict(status='WAITING_FOR_PHASE_CHECK', pid=os.getpid(), time=time.time()))
    deadline = time.monotonic() + 2 * 3600
    while not (parent / 'phase/COMPLETE.json').exists():
        if (parent / 'phase/FAILURE.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Prior phase failed or queue deadline reached')
        time.sleep(10)
    with engine.LOCK.open('r') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError('GPU queue deadline reached')
                time.sleep(10)
        # The lock is released at interpreter teardown; allow CUDA cleanup.
        time.sleep(2)
        engine.gpu_check()
        verify(root, plan)
        engine.torch.set_num_threads(2)
        engine.torch.backends.cudnn.benchmark = False
        engine.torch.backends.cuda.matmul.allow_tf32 = False
        engine.torch.backends.cudnn.allow_tf32 = False
        checkpoint = parent / 'gpu/SELECTED_005.pt'
        digest = sha256(checkpoint)
        saved = engine.torch.load(checkpoint, map_location='cpu', weights_only=False)
        selected = saved['best']['epoch']
        seen = set()
        for epoch in range(1, selected + 1):
            history = NativeMixtures(parent / 'preparation', 'train_pack', epoch, use_features=False)
            seen.update(mixture_key(row) for row in history.rows)
        net = engine.stft_build('unet_mean').cuda().eval()
        net.load_state_dict(saved['model'])
        del saved
        write_json(root / 'STATE.json', dict(status='GPU_EVALUATION_RUNNING', pid=os.getpid(), time=time.time()))
        summaries = {}
        with engine.torch.no_grad():
            for epoch in (1,5):
                data = NativeMixtures(parent / 'preparation', 'train_pack', epoch)
                rows = []
                for index in plan['subsets'][str(epoch)]:
                    item = native_inference_batch(data[index], 'cuda')
                    source = data.rows[index]
                    clips = [data.library.clips[int(i)] for i in source['indices'][:int(source['count'])]]
                    inputs = {k:item[k] for k in ('mixture','context_features','crop_start')}
                    prediction, logits = engine.stft_predict(net, inputs)
                    row = engine.row_metrics(prediction, logits, item, source, clips, index, {})
                    if row['sum_relative_error'] > 1e-9:
                        raise ValueError('Fit-gap mixture sum failed')
                    rows.append(row)
                result = engine.aggregate(rows, selected)
                repeats = sum(mixture_key(data.rows[i]) in seen for i in plan['subsets'][str(epoch)])
                if repeats != (210 if epoch <= selected else 0):
                    raise ValueError('Mixture-exposure assumption failed')
                result.update(training_schedule_epoch=epoch, native_adaptation_seen=epoch <= selected,
                    exact_repeated_native_mixtures=repeats, shared_train_recordings=True,
                    selected_epoch=selected, diagnostic_only=True)
                write_json(root / f'TRAIN_EPOCH_{epoch:03d}.json', result)
                summaries[str(epoch)] = {k:v for k,v in result.items() if k != 'rows'}
                print(json.dumps(dict(stage='FIT_GAP_SUBSET_COMPLETE', **summaries[str(epoch)])), flush=True)
        verify(root, plan)
        if sha256(checkpoint) != digest:
            raise ValueError('Fit-gap checkpoint changed')
        validation = read(parent / 'gpu' / f'VALIDATION_{selected:03d}.json')
        write_json(root / 'COMPLETE.json', dict(status='COMPLETE', protocol_sha256=sha256(root / 'PROTOCOL.json'),
            checkpoint_sha256=digest, selected_epoch=selected, subsets=summaries,
            validation_by_count=validation['by_count'], model_updates=0, heldout_read=False, time=time.time()))
        write_json(root / 'STATE.json', dict(status='COMPLETED', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {14,15})
    os.nice(10)
    try:
        main(args.run.resolve())
    except Exception:
        write_json(args.run / 'fit_gap/FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        write_json(args.run / 'fit_gap/STATE.json', dict(status='FAILED', time=time.time()))
        raise
