"""Supersede pooled candidates with independent, within-dataset experiments."""
import argparse
import collections
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from drone_rf.mixture_constraints import validate_sources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new selection version')
    seal = json.loads((args.parent/'FREEZE.json').read_text())
    for name,digest in seal['files'].items():
        if sha(args.parent/name) != digest:
            raise ValueError('Changed parent selection: '+name)
    old = json.loads((args.parent/'SELECTION.json').read_text())
    rf = json.loads((args.parent/'RFUAV_EXISTING_ROLES.json').read_text())
    dd = json.loads((args.parent/'DRONEDETECT_ROLES.json').read_text())
    groups = collections.defaultdict(list)
    for r in rf+dd:
        groups[r['source_id']].append(r)
    # XML is metadata only, including the heldout type. Its I/Q stays unopened.
    xml_rows = []
    seen = set()
    for r in rf:
        if r['pack_id'] in seen:
            continue
        seen.add(r['pack_id'])
        p = Path(r['path']).parent/Path(r['pack_id']).name
        fields = {node.tag: node.text for node in ET.fromstring(p.read_bytes())}
        if float(fields['SampleRate']) != r['fs_hz'] or float(fields['CenterFrequency']) != r['center_hz']:
            raise ValueError('XML contradicts inventory')
        xml_rows.append(dict(pack_id=r['pack_id'], xml_sha256=sha(p), role=r['role'],
            device_type=fields.get('DeviceType'), fs_hz=float(fields['SampleRate']),
            center_hz=float(fields['CenterFrequency']), if_bandwidth_hz=float(fields['IFBandwidth']),
            scale_factor=fields.get('ScaleFactor'), reference_snr_level=fields.get('ReferenceSNRLevel'),
            receiver_serial_verified=False))
    experiments = []
    removed = collections.Counter()
    for dataset,priority in [('RFUAV','primary'),('DroneDetect_V2','independent_supplementary_candidate')]:
        entry = dict(dataset=dataset, priority=priority, joint_training=False,
            shared_weights_across_datasets=False, new_gpu_training_started=False,
            native_sample_rate_hz=100000000 if dataset=='RFUAV' else 60000000,
            source_roles_inherited_without_changes=True,
            required_before_execution=['frozen native-IQ loader','same-budget GPU preflight'],
            primary_count_is_recorded_components_not_physical_aircraft=True)
        if dataset=='DroneDetect_V2':
            entry['required_before_execution'] += ['DC/variable-component audit',
                'physical-time context geometry at native rate','transmitter provenance review']
        for field in ['training_classes','withheld_combination_classes','unknown_type_classes']:
            rows=[]
            for row in old[field]:
                all_records = [record for source in row['sources'] for record in groups[source]]
                if {r['dataset'] for r in all_records} != {dataset}:
                    continue
                # Check every recorded center/rate, rather than trusting a category name.
                alternatives = {}
                for source in row['sources']:
                    values={(r['dataset'],r['center_hz'],r['fs_hz']) for r in groups[source]}
                    alternatives[source]=[dict(dataset=a,center_hz=b,fs_hz=c) for a,b,c in values]
                import itertools
                for chosen in itertools.product(*alternatives.values()):
                    checked=validate_sources(chosen)
                    if checked['rf_band'] != row['rf_band']:
                        raise ValueError('Incorrect band in previous selection')
                rows.append(row)
            entry[field]=rows
        entry['class_counts']={field:dict(collections.Counter(r['count'] for r in entry[field]))
            for field in ['training_classes','withheld_combination_classes','unknown_type_classes']}
        experiments.append(entry)
    for field in ['training_classes','withheld_combination_classes','unknown_type_classes']:
        removed[field]=len(old[field])-sum(len(e[field]) for e in experiments)
    result=dict(schema='drone_rf.within_dataset_selection.v3',time=time.time(),seed=0,
        status='WITHIN_DATASET_SAME_BAND_SELECTION_FROZEN',
        supersedes='v1 pooled candidates and v2 metadata carrying an obsolete global resampling rate',
        reason='User requires one dataset per experiment to reduce receiver-confounded separation',
        parent_selection_sha256=sha(args.parent/'SELECTION.json'), parent_seal=seal,
        primary_dataset='RFUAV', cross_dataset_synthesis=False, pooled_training=False,
        preserve_native_sample_rate=True, no_cross_band_primary_mixtures=True,
        original_rf_center_offsets_preserved=False,
        placement='center-aligned synthetic mixtures within the original dataset and RF band',
        same_band_does_not_prove_overlap=True, experiments=experiments,
        removed_cross_dataset_classes=dict(removed), rf_receiver_metadata=xml_rows,
        receiver_interpretation='RFUAV XML DeviceType is a receiver model, not hardware identity. '
            'ScaleFactor and ReferenceSNRLevel vary. One dataset does not remove all acquisition confounds.',
        waveform_interpretation='Recorded UAS-link contributions; no airframe-only transmitter verification',
        mixing=dict({k:v for k,v in old['mixing'].items() if k not in ['sample_rate_hz','dronedetect_resample']},
            cross_dataset_mixtures=False,
            sample_rate_policy='native rate within each independent dataset',
            sample_rates_hz={'RFUAV':100000000,'DroneDetect_V2':60000000},
            resampling_required=False),
        next_primary_comparison=dict(arms=['ordered TCN','mean-context control'],
            dataset='RFUAV', same_native_band=True, same_initial_weights=True,
            seed=0, epochs_max=50, examples_per_epoch=2400,effective_batch=32,
            updates_per_epoch=75, checkpoints=[5,10,25,50],
            validation='same fixed within-band validation for both arms',
            band_counts={'2.4GHz':[1,2],'5.8GHz':[1,2,3]},
            current_cross_band_trial='retain as prior synthetic exploratory control; do not relabel'),
        lora_status='No verified local IQ; generic public LoRa remains outside drone primary experiment',
        crossfire_status='No verified local or public capture admitted',
        validation_or_heldout_iq_read_this_selection=False)
    args.output.mkdir(parents=True)
    p=args.output/'SELECTION.json'
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    freeze=dict(time=time.time(),files={'SELECTION.json':sha(p)},
        source_sha256=sha(Path(__file__)),constraint_sha256=sha(Path(__file__).parents[1]/'src/drone_rf/mixture_constraints.py'))
    (args.output/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],removed=dict(removed),
        experiments=[{k:e[k] for k in ['dataset','native_sample_rate_hz','class_counts']} for e in experiments]),ensure_ascii=False))


if __name__=='__main__':
    main()
