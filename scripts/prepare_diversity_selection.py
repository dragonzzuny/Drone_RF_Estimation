"""Seal candidate source roles and mixture classes before inspecting new IQ."""
import argparse
import collections
import hashlib
import itertools
import json
import time
import zipfile
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new version; never replace a sealed split')
    inv = json.loads(args.inventory.read_text())
    rf = [dict(r, dataset='RFUAV', source_id='RFUAV:' + r['category'])
          for r in inv['files'] if r['aircraft_category']]
    dd = []
    with zipfile.ZipFile(args.archive) as z:
        for member in sorted(z.infolist(), key=lambda i: i.filename):
            if not member.filename.endswith('.dat'):
                continue
            parts = Path(member.filename).parts
            assert len(parts) == 4 and parts[0] == 'DroneDetect_V2'
            code, mode = parts[2].split('_')
            condition = parts[1]
            repeat = int(Path(member.filename).stem.rsplit('_', 1)[1])
            assert code in {'AIR', 'DIS', 'INS', 'MIN', 'MP1', 'MP2', 'PHA'}
            assert mode in {'ON', 'HO', 'FY'} and repeat in range(5)
            assert condition in {'CLEAN', 'BLUE', 'WIFI', 'BOTH'}
            assert member.file_size % 8 == 0
            if code == 'DIS':
                role = 'heldout_type_clean' if condition == 'CLEAN' else 'heldout_type_interference'
            elif condition != 'CLEAN':
                role = 'reserved_interference_recording_check'
            else:
                role = {0: 'train_candidate', 1: 'train_candidate', 2: 'train_candidate',
                        3: 'validation_candidate', 4: 'known_type_confirmation'}[repeat]
            dd.append(dict(dataset='DroneDetect_V2', member=member.filename,
                code=code, source_id='DroneDetect:' + code, mode=mode, condition=condition,
                repeat=repeat, role=role, bytes=member.file_size, crc32=member.CRC,
                recording_group=member.filename,
                fs_hz=60_000_000, center_hz=2_437_500_000,
                physical_airframe_id=None, independent_session_verified=False,
                airframe_only_emission_verified=False,
                prior_exposure='Previously used in detector research; separator v1 uses no DroneDetect data or pretrained detector weights',
                format='<c8 under existing author-supported reader; local endianness assumption retained',
                payload_read_this_selection=False))
    assert len(dd) == 390 and sum(r['role']=='train_candidate' for r in dd) == 51
    assert sum(r['role']=='validation_candidate' for r in dd) == 17
    assert sum(r['role']=='known_type_confirmation' for r in dd) == 17
    assert sum(r['code']=='DIS' for r in dd) == 40
    ids = sorted({'RFUAV:' + r['category'] for r in rf if r['role']=='train_pack'} |
                 {r['source_id'] for r in dd if r['role']=='train_candidate'})
    bands = {i: ('5.8GHz' if any(s in i for s in ['AVATA2', 'FPV COMBO', 'MAVIC3 PRO'])
                 else '2.4GHz') for i in ids}
    withheld_pair = frozenset({'DroneDetect:MP1', 'DroneDetect:MP2'})
    train_classes, combination_test = [], []
    for count in (1, 2, 3):
        for group in itertools.combinations(ids, count):
            if len({bands[g] for g in group}) != 1:
                continue
            row = dict(sources=list(group), count=count, rf_band=bands[group[0]],
                       placement='center_aligned_same_ism_band_synthetic',
                       actual_simultaneous_capture=False)
            (combination_test if withheld_pair.issubset(group) else train_classes).append(row)
    unseen = {'RFUAV:DAUTEL EVO NANO': '5.8GHz', 'DroneDetect:DIS': '2.4GHz'}
    unknown_classes = []
    for unknown, band in unseen.items():
        companions = [i for i in ids if bands[i] == band]
        for count in (1, 2, 3):
            for known in itertools.combinations(companions, count - 1):
                unknown_classes.append(dict(sources=[unknown, *known], count=count, rf_band=band,
                    withheld_pair_also_present=withheld_pair.issubset(known),
                    placement='center_aligned_same_ism_band_synthetic'))
    assert all(not withheld_pair.issubset(r['sources']) for r in train_classes)
    assert all(not (set(r['sources']) & unseen.keys()) for r in train_classes)
    spec = dict(schema='drone_rf.diversity_selection.v1', time=time.time(), seed=0,
        status='CANDIDATE_SOURCE_ROLES_AND_MIXTURE_CLASSES_FROZEN',
        gpu_training_started=False, native_iq_audit_complete=False,
        current_frozen_trial_changed=False,
        training_source_ids=ids, unknown_type_ids=list(unseen),
        unknown_type_interpretation='Absent from this separation model training; not a claim of previously unopened datasets or new physical aircraft',
        source_truth='Contribution of a recorded UAS link; transmitter direction unresolved for RFUAV and DroneDetect',
        strict_airframe_only_admission=False,
        protected_existing_confirmation_opened=False,
        rf_inventory_sha256=sha(args.inventory), dronedetect_archive=str(args.archive),
        roles=dict(collections.Counter(r['role'] for r in dd)),
        validation_choice='repeat 3 known CLEAN; selection never uses repeat 4 or DIS/Autel',
        split_unit='whole DAT recording before windowing; repetitions do not prove independent dates',
        training_classes=train_classes, withheld_combination_classes=combination_test,
        unknown_type_classes=unknown_classes,
        mixing=dict(max_components=3, complex_addition=True, image_overlay=False,
            source_phase='independent uniform [0,2pi)', time_offsets='independent within parent recording role',
            two_source_db=[-10, 0, 10], three_source_db=[[0,0,0],[-10,0,0],[0,-10,0],[0,0,-10],[10,0,0],[0,10,0],[0,0,10]],
            power_reference='whole 20.97152ms context, then crop; report measured local power too',
            sample_rate_hz=100_000_000, dronedetect_resample=[5,3],
            no_cross_ism_band_mixtures_in_primary=True,
            native_band_replay='separate check preserving relative RF offsets and common observed bandwidth; references are identically filtered',
            same_model_mixtures='separate cross-record diagnostic, not a claim of multiple physical aircraft'),
        next_gpu_comparison=dict(architecture='full U-Net+ordered TCN', seed=0,
            arms=['RFUAV5_same_band', 'RFUAV5_plus_DroneDetect6_same_band'],
            same_initial_weights=True, epochs_max=50, examples_per_epoch=2400,
            effective_batch=32, updates_per_epoch=75, checkpoints=[5,10,25,50],
            new_recorded_source_cohort_comparison=True, frozen_loader_and_preflight_required=True,
            train_sampling='800 examples per count; expanded arm alternates RFUAV-only, DroneDetect-only, same-band cross-dataset classes when feasible; uniform cyclic class allocation',
            success='record-grouped NMSE and absolute SI-SDR of every component, weak-source strata, count confusion; fixed common validation; no heldout tuning'),
        lora=dict(local_verified_iq=False, public_source_found=True,
            source='https://research.engr.oregonstate.edu/hamdaoui/datasets',
            use='auxiliary LoRa waveform experiment; IoT records are not drone ground truth',
            future_train='OSU setup2 devices1-15 days1-3', future_validation='devices1-15 day4',
            future_unseen_devices='devices16-25 day5', acquisition_and_license_receipt_required=True),
        crossfire=dict(local_verified_iq=False, public_verified_iq_dataset_found=False,
            action='source evidence or separately collected air-side telemetry required; no substitution by generic LoRa or CRSF serial logs'))
    args.output.mkdir(parents=True)
    dump(args.output/'RFUAV_EXISTING_ROLES.json', rf)
    dump(args.output/'DRONEDETECT_ROLES.json', dd)
    dump(args.output/'SELECTION.json', spec)
    freeze = {p.name: sha(p) for p in sorted(args.output.glob('*.json'))}
    dump(args.output/'FREEZE.json', dict(time=time.time(), files=freeze,
        source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status=spec['status'], training_categories=len(ids),
        roles=spec['roles'], training_classes=dict(collections.Counter(r['count'] for r in train_classes)),
        withheld_combination_classes=len(combination_test), unknown_type_classes=len(unknown_classes)),ensure_ascii=False))


if __name__ == '__main__':
    main()
