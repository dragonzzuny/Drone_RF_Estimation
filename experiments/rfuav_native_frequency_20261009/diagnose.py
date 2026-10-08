"""TRAIN-only, prespecified RF replay audit. No network or heldout access."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from native import BANDS, FS, GUARD, LENGTH, WINDOW, TAPS, BETA, transform, crop_start, mean_power, spectral_overlap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0, str(LEGACY))
from dense_data import DenseContextMixtures
from drone_rf.context_data import component_gains
from drone_rf.context_training_data import write_json
from drone_rf.data import sha256

PREPARATION = Path('/home/pyj/문서/GitHub/uav_analysis/rf_detection/rfuav_architecture_20261008/dense_preparation')


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'RESULT.json').exists():
        raise RuntimeError('Completed diagnostic; do not overwrite')
    data = DenseContextMixtures(PREPARATION, 'train_pack', 1, use_features=False)
    used, selected = Counter(), []
    for i, row in enumerate(data.rows):
        count = int(row['count'])
        names = tuple(data.library.clips[int(j)]['category'] for j in row['indices'][:count])
        key = names + tuple(map(float, row['levels'][:count]))
        if used[key] < 2:
            selected.append(i); used[key] += 1
    protocol = dict(status='REGISTERED_TRAIN_ONLY_DIAGNOSTIC', selected_indices=selected,
        selection='first two epoch1 TRAIN rows per category tuple and nominal level tuple; before IQ access',
        bands=BANDS, fs_hz=FS, taps=TAPS, kaiser_beta=BETA, guard_each_end_samples=GUARD,
        context_samples=LENGTH, window_samples=WINDOW, data_sha256={str(p):sha256(p) for p in
            [PREPARATION/'PREPARATION.json', PREPARATION/'TRAIN.npy', PREPARATION/'CACHE_MANIFEST.json']},
        code_sha256={p.name:sha256(p) for p in HERE.glob('*.py')},
        heldout_read=False, validation_iq_read=False,
        pass_criteria='finite positive filtered power; retained fraction <= 1.001; reference sum relative error <1e-12',
        overlap='intersection of normalized mean PSD histograms, including receiver noise; not active-bin occupancy')
    write_json(output/'PROTOCOL.json', protocol)
    cache, clips_log, results = {}, {}, []
    start_time = time.time()
    for i in selected:
        row = data.rows[i]; count = int(row['count']); indices = row['indices'][:count]
        clips = [data.library.clips[int(j)] for j in indices]
        raw, native, retentions = [], [], []
        for j, clip in zip(indices, clips):
            j = int(j)
            if j not in cache:
                x = data.library._array(j)
                y = transform(x, int(clip['center_hz']), int(clip['offset_samples']))
                p = mean_power(y); ratio = p/mean_power(x[GUARD:-GUARD])
                if not np.isfinite(p) or p <= 0 or not 0 < ratio <= 1.001:
                    raise RuntimeError('Invalid filtered context energy')
                # Keep only bounded short diagnostic crops; subsequent references
                # to a context are recomputed instead of growing a large RAM cache.
                cache[j] = (p, ratio)
                clips_log[j] = dict(clip_id=clip['clip_id'], category=clip['category'],
                    pack_id=clip['pack_id'], source_sha256=clip['source_sha256'],
                    cache_sha256=clip['cache_sha256'], center_hz=clip['center_hz'],
                    offset_samples=clip['offset_samples'], retained_power_fraction=ratio)
            else:
                x = data.library._array(j)
                y = transform(x, int(clip['center_hz']), int(clip['offset_samples']))
            new_start = crop_start(int(row['crop_start']))
            raw.append(np.array(x[new_start+GUARD:new_start+GUARD+WINDOW]))
            native.append(y[new_start:new_start+WINDOW].copy())
            retentions.append(cache[j][1])
        powers = [cache[int(j)][0] for j in indices]
        gains = component_gains(powers, row['levels'][:count], row['phases'][:count])
        refs = np.asarray([x*g for x,g in zip(native,gains)], dtype=np.complex64)
        mixed = refs.sum(0)
        sum_error = mean_power(mixed.astype(np.complex128)-refs.astype(np.complex128).sum(0))/mean_power(mixed)
        if not np.isfinite(mixed).all() or sum_error >= 1e-12:
            raise RuntimeError('Mixture/reference inconsistency')
        pairs = [dict(first=a, second=b, center_aligned_psd_overlap=spectral_overlap(raw[a],raw[b]),
            native_psd_overlap=spectral_overlap(native[a],native[b])) for a,b in itertools.combinations(range(count),2)]
        local_power = [mean_power(x) for x in refs]
        results.append(dict(index=i,count=count,categories=[c['category'] for c in clips],
            pack_ids=[c['pack_id'] for c in clips], nominal_levels_db=row['levels'][:count].tolist(),
            original_crop_start=int(row['crop_start']), adjusted_original_crop_start=new_start+GUARD,
            retained_power_fractions=retentions, local_reference_power=local_power,
            local_power_gap_db=10*np.log10(max(local_power)/min(local_power)),
            sum_relative_error=sum_error, pairs=pairs))
        print(json.dumps(dict(stage='TRAIN_AUDIT',complete=len(results),total=len(selected),
                              seconds=time.time()-start_time)),flush=True)
    result = dict(status='PASS',protocol_sha256=sha256(output/'PROTOCOL.json'),
        cases=len(results), unique_contexts=len(clips_log), seconds=time.time()-start_time,
        clips=list(clips_log.values()),rows=results,heldout_read=False,validation_iq_read=False)
    write_json(output/'RESULT.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
