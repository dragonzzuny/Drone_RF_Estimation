"""Fixed four-phase inference on the native study's preselected checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from native_data import ROOT, NativeMixtures, sha256, write_json
sys.path.insert(0, str(ROOT / 'experiments/rfuav_phase_average_20261009'))
import run_phase as engine


def native_inference_batch(item, device):
    """The STFT branch consumes native features, not a raw long-IQ tensor."""
    keys = ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')
    return {key: engine.torch.as_tensor(item[key], device=device)[None] for key in keys}


def read(path):
    return json.loads(path.read_text())


def verify(root, plan):
    for rel, digest in plan['source_sha256'].items():
        if sha256(ROOT / rel) != digest or sha256(root / 'source_snapshot' / rel) != digest:
            raise ValueError(f'Changed phase inference source: {rel}')
    for path, digest in plan['data_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Changed phase inference input: {path}')


def run(parent):
    root = parent / 'phase'
    root.mkdir(exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Refuse duplicate phase run')
    old = read(ROOT / 'local/phase_average_20261009_v1/PROTOCOL.json')
    sources = dict(old['source_sha256'])
    for rel, digest in read(parent / 'GPU_PROTOCOL.json')['source_sha256'].items():
        if rel in sources and sources[rel] != digest:
            raise ValueError('Source version conflict')
        sources[rel] = digest
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(__file__)
    paths = [parent / 'GPU_PROTOCOL.json', parent / 'PREP_PROTOCOL.json',
             parent / 'preparation/PREPARATION.json', parent / 'preparation/NATIVE_MANIFEST.json',
             parent / 'preparation/features/validation_pack_001.json',
             parent / 'preparation/features/validation_pack_001.npy']
    plan = dict(status='REGISTERED_NATIVE_SELECTED_PHASE_INFERENCE', source_sha256=sources,
        data_sha256={str(p): sha256(p) for p in paths}, parent_run=str(parent),
        candidate='native SELECTED_005.pt, selected by base-prediction count2/3 NMSE including e0',
        checkpoint_reselection=False, angles_degrees=[0, 90, 180, 270], model_updates=0,
        baseline_forward_passes=1, candidate_forward_passes=4,
        aggregation='equal complex mean after inverse global phase and prediction-only whole-window source-slot alignment; background fixed',
        validation_cases=630, native_center_offsets_preserved=True, heldout_read=False,
        inference_reference_access=False, physical_aircraft_count=False, independent_test=False,
        acceptance='NMSE lower AND complex SI-SDR higher at each count2/3 versus same fixed model one-pass; inspect weakest NMSE too',
        scope='adaptive development evaluation; does not establish a new architecture or generalization')
    plan['input_adapter'] = 'native six-field batch; inject batching adapter into unchanged phase evaluator; no raw long-IQ input required by STFT model'
    for rel, digest in sources.items():
        if sha256(ROOT / rel) != digest:
            raise ValueError(f'Old phase source changed: {rel}')
        target = root / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    write_json(root / 'PROTOCOL.json', plan)
    frozen = sha256(root / 'PROTOCOL.json')
    verify(root, plan)
    write_json(root / 'STATE.json', dict(status='WAITING_FOR_NATIVE_TRAINING', pid=os.getpid(), time=time.time()))
    parent_pid = read(parent / 'GPU_PROGRESS.json')['pid']
    deadline = time.monotonic() + 2 * 3600
    while not (parent / 'GPU_COMPLETE.json').exists():
        if (parent / 'GPU_FAILURE.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Training failed or two-hour queue limit reached')
        time.sleep(10)
    with engine.LOCK.open('r') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError('GPU queue limit reached')
                time.sleep(10)
        for _ in range(15):
            if not Path('/proc', str(parent_pid)).exists():
                break
            time.sleep(1)
        engine.gpu_check()
        verify(root, plan)
        engine.torch.set_num_threads(2)
        engine.torch.backends.cudnn.benchmark = False
        engine.torch.backends.cuda.matmul.allow_tf32 = False
        engine.torch.backends.cudnn.allow_tf32 = False
        checkpoint = parent / 'gpu/SELECTED_005.pt'
        saved = engine.torch.load(checkpoint, map_location='cpu', weights_only=False)
        summaries = [read(parent / 'gpu' / f'VALIDATION_{e:03d}.json') for e in range(6)]
        expected = min(summaries, key=lambda r: r['selection_nmse'])['epoch']
        if saved['best']['epoch'] != expected or saved['protocol_sha256'] != sha256(parent / 'GPU_PROTOCOL.json'):
            raise ValueError('Native checkpoint selection changed')
        del saved
        checkpoint_hash = sha256(checkpoint)
        write_json(root / 'CHECKPOINT.json', dict(path=str(checkpoint), sha256=checkpoint_hash, selected_epoch=expected))
        write_json(root / 'STATE.json', dict(status='GPU_EVALUATION_RUNNING', pid=os.getpid(), time=time.time()))
        data = NativeMixtures(parent / 'preparation', 'validation_pack', 1)
        engine.batch = native_inference_batch
        result = engine.evaluate('native_selected', checkpoint, lambda: engine.stft_build('unet_mean'),
            engine.stft_predict, parent / 'gpu', data, root, frozen)
        verify(root, plan)
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError('Selected checkpoint changed during evaluation')
        write_json(root / 'COMPLETE.json', dict(status='COMPLETE', result=result, protocol_sha256=frozen,
                                               time=time.time(), heldout_read=False, model_updates=0))
        write_json(root / 'STATE.json', dict(status='COMPLETED', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {14, 15})
    os.nice(10)
    try:
        run(args.run.resolve())
    except Exception:
        write_json(args.run / 'phase/FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        write_json(args.run / 'phase/STATE.json', dict(status='FAILED', time=time.time()))
        raise
