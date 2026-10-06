"""Validate frozen source roles and same-band classes; inspect TRAIN IQ only."""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(center):
    # Dataset grouping by receiver center, not proof that occupied spectra overlap.
    if 2.4e9 <= center < 2.5e9:
        return '2.4GHz'
    if 5.7e9 <= center < 5.9e9:
        return '5.8GHz'
    raise ValueError(f'Unreviewed center frequency: {center}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def power(x):
    return float(np.mean(np.abs(np.asarray(x, dtype=np.complex128)) ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), 'Preserve existing audit; use a new output')
    seal = json.loads((args.selection / 'FREEZE.json').read_text())
    for name, digest in seal['files'].items():
        require(sha(args.selection / name) == digest, 'Changed sealed file: ' + name)
    spec = json.loads((args.selection / 'SELECTION.json').read_text())
    rf = json.loads((args.selection / 'RFUAV_EXISTING_ROLES.json').read_text())
    dd = json.loads((args.selection / 'DRONEDETECT_ROLES.json').read_text())
    probe = json.loads((args.probe / 'AUDIT.json').read_text())
    receipt = json.loads((args.probe / 'RECEIPT.json').read_text())
    require(receipt['selection_sha256'] == sha(args.selection / 'SELECTION.json'),
            'Probe belongs to another selection')
    require(len({r['member'] for r in dd}) == len(dd), 'Duplicate recording roles')
    train = {r['member']: r for r in dd if r['role'] == 'train_candidate'}
    require(len(train) == 51, 'Unexpected train population')
    require(all(r['code'] != 'DIS' and r['repeat'] < 3 and r['condition'] == 'CLEAN'
                for r in train.values()), 'Heldout in training')
    for r in dd:
        if r['code'] == 'DIS':
            require(r['role'].startswith('heldout_type_'), 'DIS role contamination')
        elif r['condition'] == 'CLEAN':
            expected = ('train_candidate' if r['repeat'] < 3 else
                        'validation_candidate' if r['repeat'] == 3 else
                        'known_type_confirmation')
            require(r['role'] == expected, 'Incorrect recording role')
    category_bands, centers = collections.defaultdict(set), collections.defaultdict(set)
    for r in rf + dd:
        category_bands[r['source_id']].add(band(r['center_hz']))
        centers[r['source_id']].add(r['center_hz'])
    require(all(len(v) == 1 for v in category_bands.values()),
            'A source spans bands; mixture selection must become recording-specific')
    withheld = {'DroneDetect:MP1', 'DroneDetect:MP2'}
    unknown = set(spec['unknown_type_ids'])
    compact_classes = {}
    for field in ['training_classes', 'withheld_combination_classes', 'unknown_type_classes']:
        rows = spec[field]
        require(len({tuple(sorted(r['sources'])) for r in rows}) == len(rows),
                'Duplicate mixture classes')
        for r in rows:
            require(len(r['sources']) == r['count'] == len(set(r['sources'])),
                    'Wrong mixture count')
            actual = {next(iter(category_bands[s])) for s in r['sources']}
            require(actual == {r['rf_band']}, 'Cross-band or incorrect band class')
            if field == 'training_classes':
                require(not withheld.issubset(r['sources']), 'Heldout pair in training')
                require(not unknown.intersection(r['sources']), 'Unknown type in training')
            elif field == 'withheld_combination_classes':
                require(withheld.issubset(r['sources']), 'Wrong unseen-combination class')
            else:
                require(len(unknown.intersection(r['sources'])) == 1, 'Wrong unknown type class')
        compact_classes[field] = [r['sources'] for r in rows]
    require({r['member'] for r in probe['records']} == set(train),
            'Probe includes missing, validation, or heldout recordings')
    qualities = []
    for r in probe['records']:
        path = args.probe / 'cache' / r['cache']
        require(sha(path) == r['cache_sha256'], 'Changed IQ cache')
        x = np.load(path, mmap_mode='r', allow_pickle=False)
        require(x.shape == (2097152,) and np.iscomplexobj(x) and np.isfinite(x).all(),
                'Invalid IQ cache')
        p = power(x)
        mean = x.mean(dtype=np.complex128)
        ac = power(np.asarray(x, dtype=np.complex128) - mean)
        require(np.isclose(p, abs(mean)**2 + ac, rtol=1e-10), 'DC/AC energy identity failed')
        require(np.isclose(p, r['resampled_power'], rtol=1e-10), 'Power replay failed')
        require(ac > 0, 'Constant record')
        qualities.append(dict(member=r['member'], cache_sha256=r['cache_sha256'],
            complex_mean=[float(mean.real), float(mean.imag)], total_power=p,
            fluctuation_power=ac, dc_power_fraction=float(abs(mean)**2/p),
            rail_proxy_fraction=r['rail_proxy_fraction'],
            prefix_resampling_power_delta_db=r['resampled_to_native_power_db']))
    # Distinguish the filter from a change in measured time interval for the
    # largest prefix power discrepancy; this reads one already-used TRAIN prefix.
    r = max(probe['records'], key=lambda r: abs(r['resampled_to_native_power_db']))
    with zipfile.ZipFile(spec['dronedetect_archive']) as z:
        with z.open(r['member']) as f:
            body = f.read(1260600 * 8)
    require(hashlib.sha256(body).hexdigest() == r['prefix_sha256'], 'Native prefix changed')
    x = np.frombuffer(body, dtype='<c8')
    y = resample_poly(x, 5, 3, window=('kaiser', 5.0), padtype='line')
    crop = (len(y)-2097152)//2
    lo, hi = int(np.ceil(crop*3/5)), int(np.floor((crop+2097152)*3/5))
    power_check = dict(member=r['member'],
        full_resampling_delta_db=float(10*np.log10(power(y)/power(x))),
        approximately_time_aligned_delta_db=float(10*np.log10(r['resampled_power']/power(x[lo:hi]))),
        native_aligned_indices=[lo, hi], native_full_power=power(x),
        native_aligned_power=power(x[lo:hi]), resampled_full_power=power(y),
        resampled_cropped_power=r['resampled_power'],
        interpretation='Time-aligned comparison is within one native sample; no filter-fidelity claim.')
    dc = [r['dc_power_fraction'] for r in qualities]
    result = dict(status='SOURCE_ROLES_AND_SAME_BAND_CLASSES_VERIFIED', time=time.time(),
        script_sha256=sha(Path(__file__)), selection_sha256=sha(args.selection/'SELECTION.json'),
        probe_sha256=sha(args.probe/'AUDIT.json'), selection_seal=seal,
        rf_roles=dict(collections.Counter(r['role'] for r in rf)),
        dronedetect_roles=dict(collections.Counter(r['role'] for r in dd)),
        centers_hz={k: sorted(v) for k, v in centers.items()},
        source_bands={k: next(iter(v)) for k, v in category_bands.items()},
        class_counts={k: dict(collections.Counter(len(r) for r in v)) for k,v in compact_classes.items()},
        mixture_classes=compact_classes, cross_band_classes=0,
        checked_train_prefixes=len(qualities), iq_quality=qualities,
        dc_fraction=dict(min=min(dc), median=float(np.median(dc)), max=max(dc),
            above_10_percent=sum(v>.1 for v in dc), above_90_percent=sum(v>.9 for v in dc)),
        resampling_outlier_check=power_check,
        validation_or_heldout_payloads_read=False, new_gpu_training_started=False,
        training_admission='PENDING_DC_BANDWIDTH_AND_FULL_CONTEXT_LOADER_CHECK',
        interpretation='51 TRAIN prefixes only. DC may be receiver offset or a coherent component; '
                       'do not treat its source as diagnosed. No quality-based recording exclusion '
                       'or preprocessing adopted. Common sample rate does not equal common bandwidth. '
                       'Same-band center-aligned synthesis is not native RF offset replay.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','class_counts','checked_train_prefixes',
        'dc_fraction','resampling_outlier_check','training_admission']},ensure_ascii=False))


if __name__ == '__main__':
    main()
