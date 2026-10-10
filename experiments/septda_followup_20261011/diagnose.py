"""Bounded CPU diagnoses of data distribution and a fixed SepTDA checkpoint.

No optimizer, GPU, held-out recording or change to the running experiment.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'local/native_frequency_20261009_v1/preparation'
STUDY = ROOT / 'local/septda_continuation_20261010_v1'
PUBLIC = ROOT / 'reports/2026-10-11'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def state(run, status, **kwargs):
    write(run / 'STATE.json', dict(status=status, pid=os.getpid(), time=time.time(), **kwargs))


def register(run, mode, files, extra):
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'PROTOCOL.json').exists():
        raise ValueError('Use a fresh run directory; never overwrite a registered diagnosis')
    snapshot = run / 'source_snapshot/diagnose.py'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, snapshot)
    pins = {str(p): sha(p) for p in [Path(__file__), *files]}
    write(run / 'PROTOCOL.json', dict(mode=mode, pins=pins, heldout_read=False,
          gpu_use=False, new_optimizer_updates=0, started=time.time(), **extra))
    return pins


def verify(pins):
    for path, expected in pins.items():
        assert sha(path) == expected, path


def data_audit(run):
    import numpy as np
    prep = read(PREP / 'PREPARATION.json')
    manifest_path = PREP / 'NATIVE_MANIFEST.json'
    assert sha(manifest_path) == prep['manifest_sha256']
    clips = read(manifest_path)['clips']
    schedule_path = Path(prep['original_preparation']) / 'TRAIN.npy'
    assert sha(schedule_path) == prep['schedule_sha256']['TRAIN.npy']
    schedule = np.load(schedule_path, allow_pickle=False)
    groups = defaultdict(set)
    for clip in clips:
        assert clip['role'] in ('train_pack', 'validation_pack')
        groups[clip['role']].add(clip['pack_id'])
    assert not (groups['train_pack'] & groups['validation_pack'])
    rng = np.random.default_rng(20261011)
    chosen = []
    for epoch in range(1, 6):
        rows = schedule[schedule['epoch'] == epoch]
        assert len(rows) == 2400
        for count in (1, 2, 3):
            candidates = np.flatnonzero(rows['count'] == count)
            for index in sorted(rng.choice(candidates, size=16, replace=False).tolist()):
                chosen.append(dict(schedule_epoch=epoch, index=index, count=count))
    dev_path = STUDY / 'septda/VALIDATION_000.json'
    pins = register(run, 'DATA_POWER_AUDIT', [manifest_path, schedule_path, dev_path,
                    PREP / 'PREPARATION.json'], dict(seed=20261011, indices=chosen,
                    selection='16 uniformly sampled examples per count per sealed schedule; fixed before I/Q reads',
                    train_cases=240, dev_cases=630))
    state(run, 'READING_TRAIN_POWER', cases=0, total=240)
    verified = {}
    train = []
    for number, choice in enumerate(chosen, 1):
        rows = schedule[schedule['epoch'] == choice['schedule_epoch']]
        row = rows[choice['index']]
        n = int(row['count'])
        selected = [clips[int(i)] for i in row['indices'][:n]]
        assert all(c['role'] == 'train_pack' for c in selected)
        assert len({c['common_center_hz'] for c in selected}) == 1
        levels = np.asarray(row['levels'][:n], dtype=np.float64)
        weights = 10. ** ((levels - levels.max()) / 10.)
        weights /= weights.sum()
        phases = np.asarray(row['phases'][:n], dtype=np.float64)
        gains = np.sqrt(weights / [c['mean_power'] for c in selected]) * np.exp(1j * phases)
        start = min(max(int(row['crop_start']), 4096), 2**21 - 4096 - 63872) - 4096
        powers, ratios = [], []
        for clip, gain, weight in zip(selected, gains, weights):
            path = Path(clip['cache_path'])
            if str(path) not in verified:
                assert sha(path) == clip['cache_sha256']
                verified[str(path)] = clip['cache_sha256']
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            assert x.shape == (2088960,) and x.dtype == np.complex64
            y = (x[start:start + 63872] * gain).astype(np.complex64)
            assert np.isfinite(y).all()
            power = float(np.mean(np.abs(y.astype(np.complex128))**2))
            powers.append(power)
            ratios.append(float(10 * np.log10(max(power / weight, 1e-30))))
            del x, y
        train.append(dict(**choice, categories=[c['category'] for c in selected],
                     pack_ids=[c['pack_id'] for c in selected], reference_power=powers,
                     nominal_levels_db=levels.tolist(), local_over_context_power_db=ratios))
        if number % 16 == 0:
            state(run, 'READING_TRAIN_POWER', cases=number, total=240)
    dev = read(dev_path)['rows']
    summaries = []
    for label, rows in [('train_sample', train), ('dev_all', dev)]:
        for count in (2, 3):
            values = [r for r in rows if r['count'] == count]
            contrasts, local_ratios = [], []
            for r in values:
                p = np.asarray(r['reference_power'])
                contrasts.append(float(10 * np.log10(p.max() / p.min())))
                levels = np.asarray(r['nominal_levels_db'])
                weights = 10 ** ((levels - levels.max()) / 10.)
                weights /= weights.sum()
                local_ratios.extend((10 * np.log10(np.maximum(p / weights, 1e-30))).tolist())
            summaries.append(dict(role=label, count=count, cases=len(values),
                contrast_db_quantiles=dict(zip(['p10', 'p50', 'p90', 'max'],
                     np.quantile(contrasts, [.1, .5, .9, 1]).tolist())),
                contrast_above_20_db=sum(x > 20 for x in contrasts),
                source_windows=len(local_ratios),
                local_power_below_context_by_20_db=sum(x < -20 for x in local_ratios)))
    verify(pins)
    result = dict(status='COMPLETE_CHECKED', protocol_sha256=sha(run / 'PROTOCOL.json'),
        pack_counts={k: len(v) for k, v in groups.items()}, cross_role_pack_overlap=0,
        summaries=summaries, train_rows=train, verified_cache_sha256=verified,
        train_dev_same_gain_function=True, whole_context_normalization=True,
        heldout_read=False, gpu_use=False, new_optimizer_updates=0,
        limitations=['Train is a stratified 240-case sample, DEV is all630 fixed cases.',
                     'Low local power is not a verified noise-only label or SNR estimate.',
                     'Recording groups, not windows, limit independent data diversity.',
                     'Distribution differences do not establish the cause of model deterioration.'])
    write(run / 'COMPLETE.json', result)
    write(PUBLIC / 'DATA_POWER_AUDIT.json', result)
    state(run, 'COMPLETE_CHECKED', cases=240)
    print(json.dumps(dict(status='COMPLETE_CHECKED', summaries=summaries)), flush=True)


def branch_audit(run):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(ROOT / 'experiments/septda_rf_20261010'))
    import train as t
    from drone_rf.waveform import waveform_metrics
    protocol = read(STUDY / 'PROTOCOL.json')
    for rel, expected in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == expected
    checkpoint = STUDY / 'septda/ACTUAL_020.pt'
    epoch_receipt = read(STUDY / 'septda/EPOCH_020.json')
    assert sha(checkpoint) == epoch_receipt['actual_weights_sha256']
    prior_path = ROOT / 'reports/2026-10-10/SEPTDA_RF_TRAIN_DIAGNOSIS.json'
    prior = read(prior_path)
    parents = [r for r in prior['rows'] if r['model'] == 'parent/e0']
    indices = [r['index'] for n in (1, 2, 3) for r in [v for v in parents if v['count'] == n][:2]]
    dev_path = STUDY / 'septda/VALIDATION_020.json'
    assert sha(dev_path) == epoch_receipt['validation_sha256']
    dev_expected = [next(r for r in read(dev_path)['rows'] if r['count'] == n) for n in (1, 2, 3)]
    pins = register(run, 'CPU_BRANCH_AUDIT_E020', [checkpoint, prior_path, dev_path,
                    STUDY / 'PROTOCOL.json', STUDY / 'septda/EPOCH_020.json'],
                    dict(train_indices=indices, validation_indices=[r['index'] for r in dev_expected],
                    source_sha256=protocol['source_sha256'], checkpoint_epoch=20,
                    modes=['full', 'branch_bypassed'], total_forward_passes=18,
                    interpretation='Inference intervention on jointly trained weights, not a separately trained ablation'))
    snapshot_root = run / 'source_snapshot'
    for rel in protocol['source_sha256']:
        target = snapshot_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    state(run, 'LOADING_MODEL', completed_forward_passes=0, total=18)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['epoch'] == 20
    with torch.device('meta'):
        net = t.make_net()
    net.load_state_dict(saved['model'], assign=True)
    del saved
    net.eval()
    assert sum(p.numel() for p in net.parameters()) == 54596107
    assert all(p.device.type == 'cpu' for p in net.parameters())
    outputs, backend = [], []
    done = 0
    for role, cases in [('validation_pack', [r['index'] for r in dev_expected]), ('train_pack', indices)]:
        dataset = t.worker.NativeMixtures(PREP, role, 1)
        dataset.library.max_open = 1
        for index in cases:
            raw = dataset[index]
            item = {k: torch.as_tensor(np.asarray(raw[k])[None]) for k in
                    ('mixture', 'references', 'active', 'context_features', 'crop_start', 'construction_count')}
            for mode in ('full', 'branch_bypassed'):
                state(run, 'INFERENCE', role=role, index=index, mode=mode,
                      completed_forward_passes=done, total=18)
                original = net._forward
                try:
                    if mode == 'branch_bypassed':
                        net._forward = net._septda_parent_forward
                    with torch.inference_mode():
                        prediction, logits = t.worker.predict(net, item)
                        metrics = waveform_metrics(prediction, item['references'], item['active'], item['mixture'])
                finally:
                    net._forward = original
                n = int(raw['construction_count'])
                row = dict(role=role, index=index, count=n, mode=mode,
                     nmse=metrics['nmse'][0, :n].tolist(), si_sdr=metrics['si_sdr'][0, :n].tolist(),
                     reference_power=metrics['reference_power'][0, :n].tolist(),
                     sum_relative_error=float(metrics['sum_relative_error'][0]))
                assert np.isfinite(row['nmse'] + row['si_sdr']).all()
                assert row['sum_relative_error'] < 1e-9
                if role == 'validation_pack' and mode == 'full':
                    expected = next(r for r in dev_expected if r['index'] == index)
                    dn = max(abs(a-b) for a,b in zip(row['nmse'], expected['nmse']))
                    ds = max(abs(a-b) for a,b in zip(row['si_sdr'], expected['si_sdr']))
                    assert dn < 2e-5 and ds < 2e-3
                    backend.append(dict(index=index, max_nmse_difference=dn, max_si_sdr_difference=ds))
                if role == 'train_pack':
                    expected = next(r for r in parents if r['index'] == index)
                    assert np.allclose(row['reference_power'], expected['reference_power'], rtol=1e-6, atol=1e-12)
                outputs.append(row)
                done += 1
                write(run / 'PARTIAL.json', dict(rows=outputs, backend=backend))
                del prediction, logits, metrics
        del dataset
    verify(pins)
    for rel, expected in protocol['source_sha256'].items():
        assert sha(ROOT / rel) == sha(snapshot_root / rel) == expected
    assert not torch.cuda.is_initialized()
    result = dict(status='COMPLETE_CHECKED', checkpoint_epoch=20, rows=outputs, backend=backend,
        prior_rows=[r for r in prior['rows'] if r['index'] in indices],
        protocol_sha256=sha(run / 'PROTOCOL.json'), heldout_read=False,
        cuda_initialized=False, new_optimizer_updates=0,
        limitations=['Six fixed TRAIN cases and three DEV backend checks are diagnostic only.',
                     'Branch bypass keeps the jointly trained backbone; not a trained architecture ablation.'])
    write(run / 'COMPLETE.json', result)
    write(PUBLIC / 'BRANCH_AUDIT_E020.json', result)
    state(run, 'COMPLETE_CHECKED', completed_forward_passes=done, total=18)
    print(json.dumps(dict(status='COMPLETE_CHECKED', forwards=done)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['data', 'branch'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        (data_audit if args.mode == 'data' else branch_audit)(args.run.resolve())
    except Exception:
        write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        state(args.run, 'FAILED')
        raise
