"""Bounded TRAIN-only RF time-scale diagnosis; no architecture tuning on test.

Complex I/Q autocorrelation and power-envelope autocorrelation answer different
questions. Neither decodes hopping, proves an OFDM symbol period, or establishes
which features permit separation. Filter-induced short-lag correlation remains.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeLibrary, write_json, sha256


def autocorrelation(x):
    value = np.array(x, dtype=np.complex128, copy=True)
    value -= value.mean()
    energy = np.abs(value)**2
    if energy.sum() == 0:
        return None
    size = next_fast_len(2*len(value)-1)
    spectrum = fft(value, size, workers=2)
    numerator = ifft(spectrum*spectrum.conj(), workers=2)[:len(value)//2+1]
    prefix = np.r_[0.,np.cumsum(energy)]
    lag = np.arange(len(numerator))
    denominator = np.sqrt(prefix[len(value)-lag]*(prefix[-1]-prefix[lag]))
    return numerator/np.maximum(denominator,np.finfo(float).tiny)


def run(preparation, output):
    manifest = preparation/'NATIVE_MANIFEST.json'
    library = NativeLibrary(manifest,'train_pack')
    categories = sorted({c['category'] for c in library.clips if c['role']=='train_pack'})
    indices = []
    for name in categories:
        eligible = [i for i,c in enumerate(library.clips) if c['role']=='train_pack' and c['category']==name]
        indices += sorted(eligible,key=lambda i:library.clips[i]['clip_id'])[:4]
    records = []
    for index in indices:
        meta = library.clips[index]
        iq = library._array(index)
        corr = autocorrelation(iq)
        magnitude = np.abs(corr)
        crossing = np.flatnonzero(magnitude < .1)
        stride = 2048
        envelope = np.abs(np.asarray(iq[:len(iq)//stride*stride])).reshape(-1,stride)**2
        envelope = envelope.mean(-1,dtype=np.float64)
        env = autocorrelation(envelope)
        peaks = []
        if env is not None:
            locations, prop = find_peaks(env.real, height=.2, prominence=.1, distance=10)
            eligible = locations[locations*stride/100_000 >= .4]
            for lag in sorted(eligible,key=lambda i:env.real[i],reverse=True)[:3]:
                peaks.append(dict(lag_samples=int(lag*stride),lag_ms=float(lag*stride/100_000),correlation=float(env.real[lag])))
        lags = [1,8,32,128,512,2048,6139,16384,30592,63872,250000,400000,500000]
        row = dict(clip_id=meta['clip_id'],category=meta['category'],pack_id=meta['pack_id'],
                   samples=len(iq),cache_sha256=meta['cache_sha256'],source_sha256=meta['source_sha256'],
                   first_abs_complex_corr_below_0p1_samples=int(crossing[0]) if len(crossing) else None,
                   complex_abs_correlation={str(l):float(magnitude[l]) for l in lags},
                   envelope_stride_samples=stride,envelope_peak_candidates=peaks)
        records.append(row)
        print(f'{len(records)}/{len(indices)} {meta["category"]}',flush=True)
    result = dict(status='COMPLETE',source_sha256=sha256(Path(__file__)),
        manifest_sha256=sha256(manifest),selection='first 4 TRAIN clips per category sorted by clip_id; fixed before signal analysis',
        source_count=len(records),recording_group_count=len({r['pack_id'] for r in records}),
        sample_rate_hz=100_000_000,heldout_read=False,validation_iq_read=False,
        interpretation='descriptive correlations, not confirmed protocol/hopping labels or evidence of a sufficient receptive field',
        qualifications=['native common-band filtering contributes short-lag correlation',
            '20.8896ms contains few repeats of millisecond patterns',
            'near-zero mean complex autocorrelation does not exclude cyclic/higher-order structure',
            'clips from a recording group are not independent acquisitions'],
        rows=records,time=time.time())
    write_json(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.preparation,args.output)
