"""Audit matched development results without opening IQ or selecting checkpoints.

Adds separation-only summaries, a training-fitted band/count metadata baseline,
and local recorded-component power diagnostics to the frozen primary metric.
These are descriptive audits, not independent tests or new selection rules.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from drone_rf.data import DEVELOPMENT_CATEGORIES, sha256
from drone_rf.mixture_constraints import band_group


def mean(values):
    values = list(values)
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return float(np.mean(values))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite, nonempty diagnostic values required')
    return dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                    np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()))


def source_power_diagnostics(powers, levels):
    """Power includes receiver noise; small power is not an inactivity label."""
    powers, levels = np.asarray(powers, float), np.asarray(levels, float)
    if powers.ndim != 1 or powers.shape != levels.shape or not 1 <= len(powers) <= 3:
        raise ValueError('One to three powers and levels required')
    if not np.isfinite(np.r_[powers, levels]).all() or np.any(powers <= 0):
        raise ValueError('Positive finite reference powers required')
    weights = 10 ** ((levels - levels.max()) / 10)
    weights /= weights.sum()
    local_relative_long_db = 10 * np.log10(powers / weights)
    if len(powers) == 1:
        return dict(local_relative_long_db=local_relative_long_db.tolist(),
                    local_sir_db=None, nominal_sir_db=None, sir_difference_db=None)
    # Sum only other powers explicitly to avoid cancellation for very weak refs.
    others = np.array([np.delete(powers, i).sum() for i in range(len(powers))])
    nominal_others = np.array([np.delete(weights, i).sum() for i in range(len(weights))])
    local_sir = 10 * np.log10(powers / others)
    nominal_sir = 10 * np.log10(weights / nominal_others)
    return dict(local_relative_long_db=local_relative_long_db.tolist(),
                local_sir_db=local_sir.tolist(), nominal_sir_db=nominal_sir.tolist(),
                sir_difference_db=(local_sir - nominal_sir).tolist())


def band_count_baseline(training, validation):
    """Fit using train metadata ONLY. Band is not an input to the count head."""
    counts = defaultdict(Counter)
    for row in training:
        counts[row['band']][row['count']] += 1
    # Prespecified deterministic tie break: smaller synthetic count.
    predictions = {band: min(c, key=lambda n: (-c[n], n)) for band, c in counts.items()}
    if any(r['band'] not in predictions for r in validation):
        raise ValueError('Validation band absent from training')
    confusion = np.zeros((3, 3), dtype=int)
    for row in validation:
        confusion[row['count'] - 1, predictions[row['band']] - 1] += 1
    return dict(prediction_by_band=predictions,
                training_counts_by_band={b: dict(sorted(c.items())) for b, c in sorted(counts.items())},
                validation_accuracy=float(np.trace(confusion) / len(validation)),
                validation_confusion=confusion.tolist(),
                interpretation='Metadata diagnostic fitted on training counts; not RF source separation. '
                'Receiver band is known to this baseline but is not directly supplied to the neural head.')


def row_summary(rows):
    if not rows:
        raise ValueError('Empty stratum')
    multi = all(r['construction_count'] > 1 for r in rows)
    return dict(cases=len(rows),
        nmse=mean(mean(r['nmse']) for r in rows),
        si_sdr_db=mean(mean(r['si_sdr']) for r in rows),
        si_sdr_improvement_db=mean(mean(r['si_sdr_improvement']) for r in rows) if multi else None,
        weakest_local_component_nmse=mean(r['nmse'][int(np.argmin(r['reference_power']))] for r in rows),
        all_components_positive_improvement=mean(r['all_components_positive_improvement'] for r in rows) if multi else None,
        inactive_slot_energy_over_mixture=mean(r['inactive_leak'] for r in rows),
        background_energy_over_mixture=mean(r['background_nmse'] for r in rows),
        construction_count_accuracy=mean(r['construction_count'] == r['predicted_construction_count'] for r in rows))


def summarize_rows(rows):
    by_count = {str(c): row_summary([r for r in rows if r['construction_count'] == c]) for c in (1, 2, 3)}
    return dict(by_count=by_count,
                original_selection_macro_nmse=mean(r['nmse'] for r in by_count.values()),
                separation_only_count_macro_nmse=mean(by_count[str(c)]['nmse'] for c in (2, 3)),
                selection_changed=False)


def schedule_metadata(schedule, clips, role):
    result = []
    for row in schedule:
        count = int(row['count'])
        if count not in (1, 2, 3) or np.any(row['indices'][count:] != -1):
            raise ValueError('Invalid scheduled count/slots')
        indices = row['indices'][:count]
        if np.any(indices < 0) or np.any(indices >= len(clips)):
            raise ValueError('Invalid clip index')
        sources = [clips[int(i)] for i in indices]
        if any(c['role'] != role or c['category'] not in DEVELOPMENT_CATEGORIES for c in sources):
            raise ValueError('Unexpected role or category')
        if len({c['category'] for c in sources}) != count:
            raise ValueError('Duplicate category in mixture')
        bands = {band_group(c['center_hz']) for c in sources}
        if len(bands) != 1:
            raise ValueError('Cross-band case in same-band audit')
        result.append(dict(count=count, band=next(iter(bands)),
            categories=[c['category'] for c in sources],
            packs=[c['pack_id'] for c in sources], clip_ids=[c['clip_id'] for c in sources]))
    return result


def validate_rows(document, schedule, epoch):
    if document['epoch'] != epoch:
        raise ValueError('Wrong validation epoch')
    rows = document['rows']
    if len(rows) != len(schedule) or sorted(r['index'] for r in rows) != list(range(len(schedule))):
        raise ValueError('Missing or duplicate validation indices')
    rows = sorted(rows, key=lambda r: r['index'])
    for row, item in zip(rows, schedule):
        count = int(item['count'])
        if row['construction_count'] != count or row['active'] != [i < count for i in range(3)]:
            raise ValueError('Invalid reference labels')
        for key in ('nmse', 'reference_power', 'si_sdr', 'si_sdr_improvement'):
            if len(row[key]) != count:
                raise ValueError('Missing component metrics')
        if not np.isfinite(row['nmse']).all() or min(row['nmse']) < 0:
            raise ValueError('Invalid NMSE')
        source_power_diagnostics(row['reference_power'], item['levels'][:count])
    summary = summarize_rows(rows)
    if not np.isclose(summary['original_selection_macro_nmse'], document['macro_component_nmse'], rtol=1e-12, atol=1e-12):
        raise ValueError('Saved aggregate does not replay')
    return rows, summary


def audit(experiment, epoch):
    if epoch < 1:
        raise ValueError('A completed positive epoch is required')
    preparation, run = experiment / 'preparation', experiment / 'run'
    freeze = json.loads((run / 'FREEZE.json').read_text())
    for name, digest in freeze['files'].items():
        if sha256(name) != digest:
            raise ValueError('Frozen experiment changed')
    config = json.loads((preparation / 'PREPARATION.json').read_text())
    if config.get('mixture_policy') != 'within_one_dataset_and_one_native_RF_band' or config['dataset'] != 'RFUAV':
        raise ValueError('This audit requires the same-band RFUAV experiment')
    manifest = Path(config['manifest'])
    if sha256(manifest) != config['manifest_sha256']:
        raise ValueError('Manifest changed')
    clips = json.loads(manifest.read_text())['clips']
    train = np.load(preparation / 'TRAIN.npy', allow_pickle=False)
    train = train[train['epoch'] <= epoch]
    validation = np.load(preparation / 'VALIDATION.npy', allow_pickle=False)
    if len(train) != epoch * config['examples_per_epoch'] or len(validation) != config['validation_examples']:
        raise ValueError('Unexpected audit budget')
    train_meta = schedule_metadata(train, clips, 'train_pack')
    val_meta = schedule_metadata(validation, clips, 'validation_pack')
    train_packs = {p for r in train_meta for p in r['packs']}
    val_packs = {p for r in val_meta for p in r['packs']}
    if train_packs & val_packs:
        raise ValueError('Original pack leakage')
    arms, saved = {}, {}
    for arm in ('ordered', 'mean'):
        path = run / arm / f'VALIDATION_{epoch:03d}.json'
        entry_path = run / arm / f'EPOCH_{epoch:03d}.json'
        entry = json.loads(entry_path.read_text())
        if entry['epoch'] != epoch or entry['updates'] != epoch * config['updates_per_epoch']:
            raise ValueError('Mismatched optimization budget')
        doc = json.loads(path.read_text())
        rows, summary = validate_rows(doc, validation, epoch)
        if not np.isclose(entry['validation']['macro_component_nmse'], summary['original_selection_macro_nmse'], rtol=1e-12, atol=1e-12):
            raise ValueError('Epoch summary does not match validation')
        groups = defaultdict(list)
        for row, meta in zip(rows, val_meta):
            groups[(meta['count'], meta['band'], tuple(meta['categories']))].append(row)
        summary['by_combination'] = [dict(count=count, band=band, categories=list(cats), **row_summary(items))
            for (count, band, cats), items in sorted(groups.items())]
        summary.update(epoch=epoch, updates=entry['updates'],
            validation_sha256=sha256(path), epoch_receipt_sha256=sha256(entry_path))
        arms[arm], saved[arm] = summary, rows
    for first, second in zip(saved['ordered'], saved['mean']):
        if not np.array_equal(first['reference_power'], second['reference_power']):
            raise ValueError('Comparison references differ')
    power_rows, components = [], []
    for row, source, meta in zip(saved['ordered'], validation, val_meta):
        n = meta['count']
        diag = source_power_diagnostics(row['reference_power'], source['levels'][:n])
        power_rows.append(dict(index=row['index'], count=n, **diag))
        for j, category in enumerate(meta['categories']):
            components.append(dict(count=n, category=category, pack=meta['packs'][j],
                local_sir_db=None if n == 1 else diag['local_sir_db'][j],
                local_relative_long_db=diag['local_relative_long_db'][j],
                **{arm: dict(nmse=saved[arm][row['index']]['nmse'][j],
                             si_sdr_db=saved[arm][row['index']]['si_sdr'][j]) for arm in arms}))
    bins = [(-np.inf, -20, 'below_-20'), (-20, -10, '-20_to_-10'), (-10, 0, '-10_to_0'),
            (0, 10, '0_to_10'), (10, 20, '10_to_20'), (20, np.inf, '20_and_above')]
    sir_strata = []
    for count in (2, 3):
        for lo, hi, label in bins:
            items = [c for c in components if c['count'] == count and lo <= c['local_sir_db'] < hi]
            if items:
                sir_strata.append(dict(count=count, local_sir_bin_db=label, components=len(items),
                    arms={arm: dict(nmse=mean(c[arm]['nmse'] for c in items),
                                    si_sdr_db=mean(c[arm]['si_sdr_db'] for c in items)) for arm in arms}))
    power_summary = []
    for count in (1, 2, 3):
        items = [r for r in power_rows if r['count'] == count]
        power_summary.append(dict(count=count, cases=len(items),
            local_relative_long_db=quantiles([v for r in items for v in r['local_relative_long_db']]),
            any_component_below_own_long_mean_by_20db_fraction=mean(min(r['local_relative_long_db']) < -20 for r in items),
            any_sir_difference_over_3db_fraction=mean(max(abs(v) for v in r['sir_difference_db']) > 3 for r in items) if count > 1 else None,
            local_sir_db=quantiles([v for r in items for v in r['local_sir_db']]) if count > 1 else None))
    pair_rows = [i for i, r in enumerate(val_meta) if r['count'] == 2]
    untied = [i for i in pair_rows if validation[i]['levels'][0] != validation[i]['levels'][1]]
    reversal = mean(np.sign(validation[i]['levels'][0] - validation[i]['levels'][1]) !=
                    np.sign(saved['ordered'][i]['reference_power'][0] - saved['ordered'][i]['reference_power'][1]) for i in untied)
    per_category = []
    for category in sorted({r['category'] for r in components}):
        items = [r for r in components if r['category'] == category and r['count'] > 1]
        per_category.append(dict(category=category, components=len(items),
            validation_packs=len({r['pack'] for r in items}),
            arms={arm: dict(nmse=mean(r[arm]['nmse'] for r in items),
                            si_sdr_db=mean(r[arm]['si_sdr_db'] for r in items)) for arm in arms}))
    return dict(status='DESCRIPTIVE_MATCHED_EPOCH_AUDIT', time=time.time(),
        experiment=str(experiment), epoch=epoch, source_sha256=sha256(__file__),
        freeze_sha256=sha256(run / 'FREEZE.json'), manifest_sha256=config['manifest_sha256'],
        schedule_sha256=config['files'], arms=arms,
        count_band_metadata_baseline=band_count_baseline(train_meta, val_meta),
        validation_power_diagnostics=power_summary,
        pair_nominal_unequal_power_cases=len(untied), pair_local_strength_reversal_fraction=reversal,
        by_local_sir=sir_strata, separated_components_by_category=per_category,
        evidence_units=dict(training_packs=len(train_packs), validation_packs=len(val_packs),
            validation_mixtures=len(validation),
            note='Mixtures/crops share source packs and are not independent repetitions.'),
        iq_read=False, model_forward_run=False, checkpoint_selection_changed=False,
        heldout_evaluated=False, independent_test=False, inferential_test_performed=False,
        interpretation='Same-epoch diagnostics, not selected-best or final results. '
            'Local power ratios include recorded noise and do not identify RF activity or SNR. '
            'Two/three-count macro NMSE is supplemental; the frozen 1/2/3 macro selection is unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.nice(10)
    os.sched_setaffinity(0, {14, 15})
    result = audit(args.experiment.resolve(), args.epoch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'epoch', 'evidence_units')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
