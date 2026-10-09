"""TRAIN-only complex lag correlation in existing native-RF cached records.

Ordinary autocorrelation is not a test proving cyclostationarity, FHSS, CP,
source separability, or an optimal reconstruction bound.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import scipy
from scipy import fft, signal

ROOT = Path(__file__).resolve().parents[2]
MAX_LAG = 32768


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def correlation(values, max_lag=MAX_LAG):
    x = np.array(values, dtype=np.complex128, copy=True)
    x -= x.mean()
    if len(x) <= max_lag or not np.isfinite(x).all():
        raise ValueError('Invalid correlation input')
    n = len(x)
    spectrum = fft.fft(x, fft.next_fast_len(2*n - 1), workers=2)
    sums = fft.ifft(spectrum * spectrum.conj(), workers=2)[:max_lag + 1]
    cumulative = np.r_[0., np.cumsum(np.abs(x) ** 2)]
    lags = np.arange(max_lag + 1)
    energy = cumulative[n - lags] * (cumulative[n] - cumulative[lags])
    if np.any(energy <= 0):
        raise ValueError('Constant or zero-power record')
    result = sums / np.sqrt(energy)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1 + 1e-10:
        raise ValueError('Invalid normalized correlation')
    return result


def checks():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    actual = correlation(x, 64)
    x -= x.mean()
    for lag in (1, 7, 64):
        expected = np.vdot(x[:-lag], x[lag:]) / np.sqrt(np.vdot(x[:-lag], x[:-lag]).real * np.vdot(x[lag:], x[lag:]).real)
        np.testing.assert_allclose(actual[lag], expected, rtol=1e-10, atol=1e-12)
    tone = np.exp(2j * np.pi * np.arange(4096) / 64)
    np.testing.assert_allclose(np.abs(correlation(tone, 64)), 1, rtol=0, atol=1e-10)
    return dict(status='PASS', direct_complex_sum_lags=[1, 7, 64], periodic_tone=True)


def run(study, root, public):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PROTOCOL.json').exists():
        raise ValueError('Duplicate lag diagnosis')
    test = checks()
    p = json.loads((study / 'PROTOCOL.json').read_text())
    manifest_path = Path(p['preparation']) / 'NATIVE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    groups = defaultdict(list)
    for clip in manifest['clips']:
        if clip['role'] == 'train_pack':
            groups[clip['pack_id']].append(clip)
    if len(groups) != 8:
        raise ValueError('Expected eight existing TRAIN groups')
    selected = []
    for key, clips in sorted(groups.items()):
        clips.sort(key=lambda c: (c['source_path'], c['offset_samples']))
        selected.extend(clips[i] for i in (0, len(clips)//2, len(clips)-1))
    if len({c['clip_id'] for c in selected}) != 24 or any(c['fs_hz'] != 100_000_000 for c in selected):
        raise ValueError('Unexpected fixed TRAIN scope')
    protocol = dict(status='REGISTERED_CPU_TRAIN_COMPLEX_LAGS',
        source_sha256=digest(Path(__file__)), manifest_sha256=digest(manifest_path),
        study_protocol_sha256=digest(study / 'PROTOCOL.json'),
        clips=[{k: c[k] for k in ('clip_id', 'cache_sha256', 'category', 'pack_id', 'samples', 'fs_hz')} for c in selected],
        selection='Lexicographically first, middle, last existing TRAIN clip per pack; fixed before waveform reads',
        max_lag_samples=MAX_LAG, lag_bands=[[32,512],[513,4096],[4097,32768]],
        estimator='Subtract segment mean; complex autocorrelation normalized by the two overlapping segment energies',
        within_clip_check='Four contiguous quarters, same lag; not independent recording replication',
        numpy_version=np.__version__, scipy_version=scipy.__version__, checks=test,
        model_updates=0, heldout_read=False, validation_read=False, gpu_use=False,
        interpretation='Descriptive ordinary autocorrelation; no protocol label or separation guarantee', registered_at=time.time())
    write(root / 'PROTOCOL.json', protocol)
    rows, curves = [], []
    for clip in selected:
        path = Path(clip['cache_path'])
        if digest(path) != clip['cache_sha256']:
            raise ValueError('Native TRAIN cache changed')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (clip['samples'],) or not np.iscomplexobj(values):
            raise ValueError('Wrong native waveform')
        whole = np.abs(correlation(values))
        quarters = [np.abs(correlation(part)) for part in np.array_split(values, 4)]
        peaks = []
        for low, high in protocol['lag_bands']:
            candidates, _ = signal.find_peaks(whole[low:high+1], distance=16)
            candidates = candidates + low
            if len(candidates) == 0:
                candidates = np.array([low + int(np.argmax(whole[low:high+1]))])
            for lag in sorted(candidates, key=lambda j: whole[j], reverse=True)[:3]:
                peaks.append(dict(lag_band_samples=[low,high], lag_samples=int(lag),
                    lag_microseconds=float(lag/clip['fs_hz']*1e6), magnitude=float(whole[lag]),
                    quarter_magnitudes=[float(q[lag]) for q in quarters]))
        rows.append(dict(clip_id=clip['clip_id'], category=clip['category'], pack_id=clip['pack_id'],
            fs_hz=clip['fs_hz'], samples=clip['samples'], peaks=peaks))
        curves.append(whole.astype(np.float32))
        write(root / 'STATE.json', dict(status='CPU_TRAIN_COMPLEX_LAGS', clips=len(rows), total=24,
            pid=os.getpid(), time=time.time()))
        del values, whole, quarters
    np.savez_compressed(root / 'CURVES.npz', magnitudes=np.stack(curves), lags=np.arange(MAX_LAG+1))
    if digest(manifest_path) != protocol['manifest_sha256'] or digest(Path(__file__)) != protocol['source_sha256']:
        raise ValueError('Frozen inputs changed')
    result = dict(status='COMPLETE', protocol_sha256=digest(root / 'PROTOCOL.json'), checks=test,
        rows=rows, clips=24, training_groups=8, distinct_categories=5,
        model_updates=0, heldout_read=False, validation_read=False, gpu_use=False,
        limitation=protocol['interpretation'])
    write(root / 'COMPLETE.json', result)
    write(public, result)
    write(root / 'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(dict(status='COMPLETE', clips=24), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('study', 'run', 'public'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.study.resolve(), args.run.resolve(), args.public.resolve())
    except Exception:
        import traceback
        write(args.run / 'FAILURE.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
