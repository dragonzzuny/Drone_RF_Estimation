"""CPU integrity audit and schedules, only old TRAIN10 cache payloads opened."""
from collections import Counter
import copy
import time
import numpy as np
from data import *


def main():
    OUT.mkdir(exist_ok=True)
    PREP.mkdir(exist_ok=True)
    value = read(MANIFEST)
    assert value['status'] == 'METADATA_SPLIT_PREPARED_NOT_TRAINING_READY'
    clips = value['clips']
    validate_clips(clips)
    checks = []
    started = time.time()
    libraries = {role: Library(clips, role) for role in ROLES}
    for i, c in enumerate(clips):
        role = c['diagnostic_role']
        if role not in ROLES:
            continue
        array = libraries[role]._array(i)
        assert np.isfinite(array).all()
        power = float(np.mean(np.abs(array).astype(np.float64)**2))
        assert np.isclose(power, c['mean_power'], rtol=1e-5, atol=0)
        checks.append(dict(clip_id=c['clip_id'], role=role, cache_sha256=c['cache_sha256'], measured_power=power))
        if len(checks) % 100 == 0:
            write_json(OUT/'PREPARATION_STATE.json', dict(stage='CACHE_AUDIT', checked=len(checks), total=2020,
                seconds=time.time()-started, pid=os.getpid()))
    assert Counter(c['role'] for c in checks) == {'fit_block':1296, 'within_record_dev':724}
    paths = {}
    for role, epochs, per_count in ((ROLES[0],50,800), (ROLES[1],1,210)):
        for epoch in range(1, epochs+1):
            rows = schedule(clips, role, epoch, per_count)
            assert np.array_equal(rows, schedule(clips, role, epoch, per_count))
            path = PREP/f'{role}_{epoch:03d}.npy'
            if path.exists():
                assert np.array_equal(np.load(path, allow_pickle=False), rows)
            else:
                np.save(path, rows, allow_pickle=False)
            paths[path.name] = sha256(path)
    # Negative test: overlapping raw support must be rejected before payload reads.
    bad = copy.deepcopy(clips)
    index = next(i for i,c in enumerate(bad) if c['diagnostic_role'] == ROLES[1])
    bad[index]['raw_support_start_samples'] = 0
    bad[index]['raw_support_end_samples'] = LENGTH + 8192
    try:
        validate_clips(bad)
        raise RuntimeError('Overlap was accepted')
    except AssertionError:
        pass
    wrong = next(i for i,c in enumerate(clips) if c['diagnostic_role'] == ROLES[1])
    try:
        libraries[ROLES[0]]._array(wrong)
        raise RuntimeError('Wrong role was accepted')
    except AssertionError:
        pass
    config = dict(status='TIME_BLOCK_SCHEDULES_READY', manifest_sha256=sha256(MANIFEST),
        schedules=paths, fs_hz=FS, context_samples=LENGTH, window_samples=WINDOW,
        epochs=50, examples_per_epoch=2400, dev_examples=630,
        same_recording_diagnostic=True, independent_recording_validation=False,
        old_checkpoint_reuse=False, cross_band_synthesis=False, heldout_read=False)
    if (PREP/'PREPARATION.json').exists():
        assert config == read(PREP/'PREPARATION.json')
    else:
        write_json(PREP/'PREPARATION.json', config)
    report = dict(status='PASS', source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (HERE/'data.py', HERE/'prepare.py')},
        manifest_sha256=sha256(MANIFEST), preparation_sha256=sha256(PREP/'PREPARATION.json'),
        checked_contexts=len(checks), roles=dict(Counter(c['role'] for c in checks)),
        finite_power_hash_checks=True, cross_role_exact_duplicate_caches=0,
        full_raw_support_gap_at_least_seconds=1, role_overlap_negative_tests='PASS',
        arbitrary_time_or_frequency_shift_duplicate_test=False,
        old_dev20_read=False, heldout_read=False, checks=checks, seconds=time.time()-started)
    write_json(OUT/'DATA_AUDIT.json', report)
    write_json(ROOT/'reports/2026-10-11/TIMEBLOCK_DATA_AUDIT.json', {k:v for k,v in report.items() if k!='checks'})
    print(f'DATA_AUDIT_PASS {len(checks)} contexts {time.time()-started:.1f}s', flush=True)
    del libraries
    prepare_features(ROLES[1], 1)
    prepare_features(ROLES[0], 1)
    # Full long-context and fast short-window paths must agree for every source count.
    errors=[]
    for role in ROLES:
        dataset = Mixtures(role)
        for count in (1,2,3):
            index = int(np.flatnonzero(dataset.rows['count']==count)[0])
            full, fast = dataset.full_example(index), dataset[index]
            for key in ('mixture','references','context_features'):
                np.testing.assert_allclose(fast[key], full[key], rtol=2e-6, atol=2e-7)
            errors.append(dict(role=role,count=count,max_sum_error=float(np.max(np.abs(fast['mixture']-fast['references'].sum(0))))))
    write_json(OUT/'READY.json', dict(status='PASS', data_audit_sha256=sha256(OUT/'DATA_AUDIT.json'),
        preparation_sha256=sha256(PREP/'PREPARATION.json'), long_short_equivalence=errors))
    print('TIMEBLOCK_PREPARATION_READY', flush=True)


if __name__ == '__main__':
    main()
