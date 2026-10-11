"""Prepare evaluation roles from metadata only; never modify frozen loaders.

The time-block split is within the SAME recording, not a new-session test.
No existing checkpoint is eligible for its validation because earlier training
could have seen both blocks. No training or I/Q read is performed here.
"""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'local/bandwidth_evaluation_20261011_v1'
PUBLIC = ROOT / 'reports/2026-10-11'
INPUTS = {}
FS = 100_000_000
GAP = FS
FILTER_GUARD = 4096


def read(path):
    value = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(value).hexdigest()
    return json.loads(value)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    inventory = read(ROOT/'local/inventory/INVENTORY.json')
    admission = read(PUBLIC/'RECORDING_ADMISSION_AUDIT.json')
    native = read(ROOT/'local/native_frequency_20261009_v1/preparation/NATIVE_MANIFEST.json')
    files = {f['relative_path']: f for f in inventory['files']}
    raw = Path(admission['raw_root'])
    groups = collections.defaultdict(list)
    strata = []
    for category in admission['admission_ledger']:
        if category['status'] != 'EXISTING_RECORDING_CONTRIBUTION_TASK_KEEP_FROZEN':
            continue
        by_width = collections.defaultdict(list)
        recorded_xml = set()
        for pack in category['packs']:
            by_width[pack['nominal_width_mhz']].append(pack)
            folder = raw / Path(pack['pack_id']).parent
            xmls = {str(f.relative_to(raw)) for f in folder.glob('*.xml')}
            assert pack['pack_id'] in xmls
            recorded_xml.update(xmls)
        assert recorded_xml == {p['pack_id'] for p in category['packs']}
        assert {str(f.relative_to(raw)) for f in (raw/category['category']).rglob('*.xml')} == recorded_xml
        for width, packs in sorted(by_width.items()):
            strata.append(dict(category=category['category'], nominal_vtsbw_mhz=width,
                               recording_groups=len(packs),
                               separate_train_dev_recordings_available=len(packs)>=2,
                               pack_ids=[p['pack_id'] for p in packs]))
    assert len(strata)==13 and all(s['recording_groups']==1 for s in strata)
    for clip in native['clips']:
        if clip['role']=='train_pack' and '/VTSBW=10/' in clip['source_path']:
            groups[clip['category']].append(clip)
    assert len(groups)==5
    allocations=[]
    summary=[]
    assigned_ids=set()
    for category, clips in sorted(groups.items()):
        assert len({c['pack_id'] for c in clips})==1
        timed=[]
        for c in clips:
            assert c['fs_hz']==FS and c['offset_samples']==c['original_offset_samples']+FILTER_GUARD
            source=files[c['source_path']]
            assert source['role']=='train_pack' and source['category']==category
            match=re.search(r'_(\d+)-(\d+)s\.iq$', c['source_path'])
            assert match and int(match[2])==int(match[1])+1
            start_local=c['offset_samples']-FILTER_GUARD
            end_local=c['offset_samples']+c['samples']+FILTER_GUARD
            assert 0<=start_local<end_local<=source['samples_cf32']
            assert end_local-start_local==2_097_152
            start=int(match[1])*FS+start_local
            end=int(match[1])*FS+end_local
            timed.append((c,start,end))
        horizon=max(end for _,_,end in timed)
        cut=(3*horizon//5//FS)*FS
        current=[]
        for c,start,end in timed:
            role='fit_block' if end<=cut else 'within_record_dev' if start>=cut+GAP else 'gap_excluded'
            assert c['clip_id'] not in assigned_ids
            assigned_ids.add(c['clip_id'])
            current.append(dict(**c, original_role=c['role'], diagnostic_role=role,
                                raw_support_start_samples=start, raw_support_end_samples=end))
        fit=[c for c in current if c['diagnostic_role']=='fit_block']
        dev=[c for c in current if c['diagnostic_role']=='within_record_dev']
        assert len(fit)>0 and len(dev)>0
        gap=min(c['raw_support_start_samples'] for c in dev)-max(c['raw_support_end_samples'] for c in fit)
        assert gap>=GAP
        # The interval comparison includes all raw samples used by filtering and
        # long-context extraction, not only the fine I/Q prediction window.
        summary.append(dict(category=category, pack_id=clips[0]['pack_id'], nominal_vtsbw_mhz=10,
                            fit_before_seconds=cut/FS, dev_from_seconds=(cut+GAP)/FS,
                            cache_metadata_horizon_seconds=horizon/FS,
                            fit_contexts=len(fit), dev_contexts=len(dev),
                            excluded_contexts=len(current)-len(fit)-len(dev),
                            minimum_raw_support_gap_seconds=gap/FS,
                            same_recording_in_both_roles=True,
                            independent_recording_validation=False))
        allocations.extend(current)
    roles=collections.Counter(c['diagnostic_role'] for c in allocations)
    plan=dict(status='METADATA_SPLIT_PREPARED_NOT_TRAINING_READY', clips=allocations,
              policy=dict(nominal_vtsbw_mhz=10,fit_fraction_of_available_timeline=.6,
                          cutoff_rounding='floor to integer second',gap_seconds=1,
                          interval_basis='filename seconds plus raw cache support including filter guards',
                          no_existing_checkpoint_reuse=True,new_initialization_seed=0,
                          existing_dev20_excluded=True,heldout_excluded=True,
                          mixture_generation_after_source_split=True,
                          new_mixture_features_required=True,
                          same_native_rf_band_only=True,fs_hz=FS),
              input_sha256=INPUTS, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'TIME_BLOCK_MANIFEST.json'
    if path.exists():
        assert json.loads(path.read_text())==plan, 'Existing design differs: use a new version'
    else:
        write(path,plan)
    report=dict(status='METADATA_AND_INTERVAL_CHECK_PASS',strata=strata,
                same_bandwidth_independent_recording_split_available=False,
                within_record_time_block_summary=summary,context_counts=dict(roles),
                local_manifest=str(path.relative_to(ROOT)),
                local_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=plan['source_sha256'],input_sha256=INPUTS,
                raw_iq_read=False,cache_arrays_read=False,heldout_payload_read=False,
                existing_split_changed=False,existing_worker_changed=False,
                new_training_started=False,new_performance_measured=False,
                physical_recording_session_independence_verified=False,
                one_second_gap_proves_independence=False,
                pending=['Check cache integrity/duplicate or shifted waveform similarity.',
                         'Implement a separate explicit within-record diagnostic loader.',
                         'Generate mixtures and mixture context features within each new role.',
                         'Freeze learning budget/selection; initialize fresh full-capacity model.',
                         'Obtain additional same-condition recordings for independent evaluation.'])
    write(PUBLIC/'BANDWIDTH_EVALUATION_AUDIT.json',report)
    print(json.dumps(dict(status=report['status'],strata=len(strata),contexts=dict(roles),
                          time_blocks=summary),ensure_ascii=False))


if __name__=='__main__':
    main()
