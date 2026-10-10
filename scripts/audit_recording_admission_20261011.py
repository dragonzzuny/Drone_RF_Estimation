"""Reconcile original RFUAV recording groups without opening any I/Q payload."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT/'local/inventory/INVENTORY.json'
MANIFEST = ROOT/'local/native_frequency_20261009_v1/preparation/NATIVE_MANIFEST.json'
SCOPE = ROOT/'reports/2026-10-07/rfuav_signal_scope_audit.json'
OUTPUT = ROOT/'reports/2026-10-11'


def read_metadata(path):
    path = Path(path)
    assert path.suffix in ('.json', '.xml'), 'Payload reads prohibited in this audit'
    return path.read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def nominal_width(pack_id):
    match = re.search(r'VTSBW=(\d+)', pack_id)
    return int(match.group(1)) if match else None


def main():
    target = OUTPUT/'RECORDING_ADMISSION_AUDIT.json'
    if target.exists():
        raise FileExistsError('Preserve completed audit; register a new output for a rerun')
    inventory_bytes, manifest_bytes, scope_bytes = [read_metadata(p) for p in (INVENTORY, MANIFEST, SCOPE)]
    inv, manifest, scope = [json.loads(b) for b in (inventory_bytes, manifest_bytes, scope_bytes)]
    assert scope['inventory_sha256'] == sha(inventory_bytes)
    files = {f['relative_path']: f for f in inv['files']}
    packs = {p['pack_id']: p for p in inv['packs']}
    first = inv['files'][0]
    raw = Path(first['path'])
    for _ in Path(first['relative_path']).parts:
        raw = raw.parent
    assert raw/first['relative_path'] == Path(first['path'])
    on_disk = {str(p.relative_to(raw)): p for p in raw.rglob('*.iq')}
    xml_on_disk = {str(p.relative_to(raw)) for p in raw.rglob('*.xml')}
    missing_files = sorted(set(files)-set(on_disk))
    new_files = sorted(set(on_disk)-set(files))
    changed_stat = []
    for key in sorted(set(files) & set(on_disk)):
        expected, current = files[key], on_disk[key].stat()
        if (expected['bytes'], expected['mtime_ns']) != (current.st_size, current.st_mtime_ns):
            changed_stat.append(key)
    changed_xml = []
    for key, pack in packs.items():
        blob = read_metadata(pack['xml_path'])
        if sha(blob) != pack['xml_sha256']:
            changed_xml.append(key)
        # Confirm fields, rather than treating only a filename as receiver evidence.
        assert {node.tag: node.text for node in ET.fromstring(blob)} == pack['metadata']

    grouped = collections.defaultdict(list)
    source_rows = {}
    for clip in manifest['clips']:
        original = files[clip['source_path']]
        assert original['role'] == clip['role']
        assert original['pack_id'] == clip['pack_id']
        assert original['category'] == clip['category']
        assert original['fs_hz'] == clip['fs_hz'] == 100_000_000
        assert original['center_hz'] == clip['center_hz']
        assert clip['role'] in ('train_pack', 'validation_pack')
        grouped[clip['role']].append(clip)
        source_rows[clip['source_path']] = original

    roles = {}
    for role, clips in grouped.items():
        paths = sorted({c['source_path'] for c in clips})
        role_packs = sorted({c['pack_id'] for c in clips})
        roles[role] = dict(clips=len(clips), files=len(paths), recording_groups=len(role_packs),
            source_paths=paths, pack_ids=role_packs,
            original_seconds=sum(files[p]['samples_cf32']/files[p]['fs_hz'] for p in paths),
            nominal_widths_mhz=sorted({nominal_width(p) for p in role_packs}))
    assert not set(roles['train_pack']['pack_ids']) & set(roles['validation_pack']['pack_ids'])
    unused_current_files = sorted(k for k, v in files.items()
        if v['role'] in ('train_pack', 'validation_pack') and k not in source_rows)
    unused_train_packs = sorted(k for k, v in packs.items()
        if v['role'] == 'train_pack' and k not in roles['train_pack']['pack_ids'])

    settings = []
    current_categories = sorted({x['category'] for x in source_rows.values()})
    for category in current_categories:
        record = dict(category=category)
        for role, label in [('train_pack', 'train'), ('validation_pack', 'dev')]:
            ps = [p for p in packs.values() if p['role'] == role and p['category'] == category]
            record[label] = dict(groups=len(ps), files=sum(p['iq_files'] for p in ps),
                nominal_widths_mhz=sorted({nominal_width(p['pack_id']) for p in ps}),
                centers_mhz=sorted({float(p['metadata']['CenterFrequency'])/1e6 for p in ps}),
                reference_snr_labels=sorted({p['metadata']['ReferenceSNRLevel'] for p in ps}),
                scale_factor_labels=sorted({p['metadata']['ScaleFactor'] for p in ps}))
        record['dev_nominal_width_absent_from_train'] = bool(
            set(record['dev']['nominal_widths_mhz']) - set(record['train']['nominal_widths_mhz']))
        settings.append(record)

    ledger = []
    scope_map = {c['category']: c for c in scope['categories']}
    for category in inv['categories']:
        name = category['category']
        ps = [p for p in packs.values() if p['category'] == name]
        if name in current_categories:
            status = 'EXISTING_RECORDING_CONTRIBUTION_TASK_KEEP_FROZEN'
        elif any(p['role'] == 'heldout_model_not_for_selection' for p in ps):
            status = 'HELDOUT_KEEP_UNOPENED'
        else:
            status = 'NOT_ADMITTED_TRANSMITTER_PROVENANCE_REVIEW_REQUIRED'
        ledger.append(dict(category=name, status=status, iq_files=category['iq_files'],
            recording_groups=len(ps), aircraft_only_capture_verified=False,
            new_admission=False, video_required=False,
            paper_video_bandwidth_mhz=scope_map[name]['paper_video_bandwidth_mhz'],
            packs=[dict(pack_id=p['pack_id'], device=p['metadata']['DeviceType'],
                        fs_hz=int(p['metadata']['SampleRate']), center_hz=float(p['metadata']['CenterFrequency']),
                        nominal_width_mhz=nominal_width(p['pack_id'])) for p in ps]))

    checks = dict(missing_iq_files=missing_files, new_iq_files=new_files,
                  changed_file_size_or_mtime=changed_stat, changed_xml=changed_xml,
                  new_xml=sorted(xml_on_disk-set(packs)), missing_xml=sorted(set(packs)-xml_on_disk))
    all_checks_pass = not any(checks.values())
    assert all_checks_pass, checks
    result = dict(status='METADATA_RECONCILIATION_COMPLETE',
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_sha256={str(p.relative_to(ROOT)): sha(b) for p, b in
                      [(INVENTORY,inventory_bytes),(MANIFEST,manifest_bytes),(SCOPE,scope_bytes)]},
        raw_root=str(raw), categories=len(inv['categories']), iq_files=len(files), recording_groups=len(packs),
        live_filesystem_checks=checks, active_roles=roles,
        unused_files_in_current_cohort=unused_current_files,
        unused_training_groups_in_current_cohort=unused_train_packs,
        new_approved_training_groups=0, new_admissions=0,
        protected_heldout_files=sum(f['role']=='heldout_model_not_for_selection' for f in files.values()),
        noncohort_review_categories=sum(c['status'].startswith('NOT_ADMITTED') for c in ledger),
        nominal_capture_settings_by_category=settings, admission_ledger=ledger,
        iq_payloads_opened=0, caches_opened=0, xml_files_opened=len(packs),
        full_iq_integrity_reverified=False, split_changed=False, heldout_payload_read=False,
        interpretation='All existing training recording groups are represented. DEV holds out VTSBW=20 for all five product labels, so its generalization gap also includes a capture-condition shift. Metadata establishes this shift, not its causal share of waveform error. VTSBW is a folder label, not measured occupied bandwidth. Current labels and pending video evidence do not establish aircraft-only emission.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines=['# 원녹음 확대 가능 여부와 분할 조건 재점검', '',
        '현재 5개 제품 라벨의 학습 원기록 8묶음·83파일은 모두 학습 캐시에 포함돼 있다. '
        '같은 조건에서 추가 편입할 미사용 원녹음 묶음은 0개다. 창이나 합성 조합을 늘리는 것은 가능하지만 새 녹음 확보와 구분한다.', '',
        '|역할|원기록 묶음|I/Q 파일|문맥 창|원파일 전체 길이|', '|---|---:|---:|---:|---:|']
    for role, label in [('train_pack','학습'),('validation_pack','개발 검증')]:
        v=roles[role]
        lines.append(f"|{label}|{v['recording_groups']}|{v['files']}|{v['clips']}|{v['original_seconds']:.3f}초|")
    lines += ['', '창 길이 합은 중복·가장자리 처리 때문에 원기록 길이와 다를 수 있다. '
        '검증 파일의 일부 창과 학습 파일의 확대 창을 비교한 집계다.', '',
        '**분할은 녹음뿐 아니라 명목 대역폭 조건도 다르다.** 검증의 `VTSBW=20` 폴더 조건은 다섯 제품 모두 학습에 없다. '
        '현재 검증은 새 녹음과 새 수집 조건에 대한 일반화를 함께 시험한다. 이 차이가 잔차 증가의 원인이라는 인과 결론은 아직 없다.', '',
        '|제품 라벨|학습 VTSBW 폴더(MHz)|검증 VTSBW 폴더(MHz)|학습/검증 중심주파수(MHz)|', '|---|---|---|---|']
    for row in settings:
        lines.append(f"|{row['category']}|{row['train']['nominal_widths_mhz']}|{row['dev']['nominal_widths_mhz']}|{row['train']['centers_mhz']} / {row['dev']['centers_mhz']}|")
    lines += ['', '`VTSBW`는 데이터 폴더의 표기이며 이 작업에서 실제 점유 대역폭을 측정하지 않았다. '
        'ReferenceSNRLevel도 XML 라벨이며 성분별 실제 SNR로 간주하지 않는다. 수신기 모델은 같아도 물리 수신기·수집 세션이 동일하다는 증거는 아니다.', '',
        '외장하드 RFUAV의 37범주·358 I/Q 파일을 파일 크기·수정시각·XML로 기존 목록과 대조했다. '
        '새 파일, 사라진 파일, 변경된 XML은 없었다. I/Q 본문과 캐시는 읽지 않았으므로 전체 파형 무결성을 재검사한 결과는 아니다.', '',
        'Autel 13파일은 보류 상태를 유지한다. 나머지 31범주도 제품명만으로 기체 측 송신이라고 편입하지 않는다. '
        'Herelink Hx4와 YUNZHUO H16의 합계 12파일은 우선 검토 후보이지만, 원문 영상전송 표기만으로 개별 녹음의 송신 방향이 확정되지는 않는다. '
        '영상 유무는 편입 조건이 아니며 조종기 측 또는 명시적 결합 라벨을 새 드론 정답으로 사용하지 않는다.', '',
        '후속 신규 자료는 같은 RFUAV·대역·표본률에서 송신 방향과 원기록 묶음이 확인돼야 한다. '
        '기존 학습과 검증 양쪽에 같은 명목 대역폭의 서로 다른 새 녹음이 있어야 녹음 일반화와 대역폭 변화를 분리할 수 있다. '
        '현재 검증 묶음을 학습으로 옮기거나 그 조각을 무작위 재분할해 이 조건을 대신하지 않는다.', '',
        '현재 GPU 비교는 봉인된 분할로 계속한다. **추가 학습 자료 0개 승인, 0개 편입**이 이번 감사의 결론이다. '
        '외부에 더 많은 적격 자료가 없다는 결론은 아니다. 현재 실패 원인 전체를 데이터 부족으로 확정하지 않는다.', '',
        '[37범주별 편입 명세와 입력 지문](RECORDING_ADMISSION_AUDIT.json) · '
        '[기존 송신원 검토](../2026-10-07/RFUAV_SCOPE_CORRECTION_KO.md) · '
        '[학습/개발 파형 진단](DATA_AND_ARCHITECTURE_DIAGNOSIS_KO.md)', '']
    (OUTPUT/'RECORDING_ADMISSION_KO.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('status','categories','iq_files','recording_groups',
        'new_approved_training_groups','new_admissions','iq_payloads_opened','xml_files_opened')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
