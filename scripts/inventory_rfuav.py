"""Inventory metadata and enforce aircraft-category admission; no RF array reads."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile

P = None
ROOT = None
RAW = None
AIRCRAFT = frozenset({'DAUTEL EVO NANO', 'DJI AVATA2', 'DJI FPV COMBO',
    'DJI MAVIC3 PRO', 'DJI MINI3', 'DJI MINI4 PRO'})
HELDOUT_MODEL = 'DAUTEL EVO NANO'

def write(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert not (P/'INVENTORY.json').exists(), 'Do not replace an existing inventory'
    categories, packs, files = [], [], []
    for directory in sorted(x for x in RAW.iterdir() if x.is_dir()):
        admitted = directory.name in AIRCRAFT
        xmls = sorted(directory.rglob('*.xml'))
        iqfiles = sorted(directory.rglob('*.iq'))
        for xml in xmls:
            meta = {n.tag: n.text for n in ET.parse(xml).getroot()}
            group = str(xml.relative_to(RAW))
            siblings = list(xml.parent.glob('*.xml'))
            member_files = sorted(f for f in iqfiles if f.parent == xml.parent and
                                  (len(siblings) == 1 or f.name.startswith(xml.stem+'_')))
            assert member_files, xml
            role = 'excluded_controller_or_nonaircraft'
            if admitted:
                assert re.sub(r'\s+', '', meta['Drone']).upper() == re.sub(r'\s+', '', directory.name).upper(), (directory, meta)
                if directory.name == HELDOUT_MODEL:
                    role = 'heldout_model_not_for_selection'
                elif 'VTSBW=20' in str(xml):
                    role = 'validation_pack'
                else:
                    role = 'train_pack'
            row = dict(pack_id=group, xml_path=str(xml), xml_sha256=digest(xml),
                category=directory.name, aircraft_category=admitted, role=role,
                metadata=meta, iq_files=len(member_files), bytes=sum(f.stat().st_size for f in member_files),
                physical_airframe_id=None, independent_session_verified=False,
                transmitter_direction_verified=False, prior_research_exposure='not_fully_audited',
                xml_stem_mismatched_members=[f.name for f in member_files if not f.name.startswith(xml.stem+'_')])
            packs.append(row)
            for f in member_files:
                st = f.stat()
                assert st.st_size > 0 and st.st_size % 8 == 0
                record = dict(path=str(f), relative_path=str(f.relative_to(RAW)),
                    pack_id=group, category=directory.name, role=role,
                    aircraft_category=admitted, bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                    samples_cf32=st.st_size//8, fs_hz=int(meta['SampleRate']),
                    center_hz=float(meta['CenterFrequency']), data_type=meta['DataType'],
                    payload_read=False, whole_file_hash_verified=False)
                files.append(record)
        categories.append(dict(category=directory.name, aircraft_category=admitted,
            iq_files=len(iqfiles), packs=len(xmls), bytes=sum(f.stat().st_size for f in iqfiles)))
    assert len({r['path'] for r in files}) == len(files)
    assert len(files) == sum(r['iq_files'] for r in categories)
    admitted = [r for r in files if r['aircraft_category']]
    assert {r['category'] for r in admitted} == AIRCRAFT
    assert all(r['fs_hz'] == 100_000_000 and r['data_type'] == 'Complex Float' for r in admitted)
    archive_summary=[]
    for path in sorted((ROOT/'archives').glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            members=[i for i in z.infolist() if not i.is_dir()]
            entry=dict(path=str(path), members=len(members),
                uncompressed_bytes=sum(i.file_size for i in members),
                extensions=dict(collections.Counter(Path(i.filename).suffix.lower() for i in members)),
                payload_read=False, full_crc_verified=False)
            if path.name == 'CARDRF.zip':
                entries=[]
                for i in members:
                    parts=Path(i.filename).parts
                    if parts[0]!='CARDRF' or not i.filename.endswith('.mat'): continue
                    start=3 if parts[1]=='LOS' else 2
                    entries.append((parts[start],parts[start+1]))
                entry['category_counts']=dict(collections.Counter(c for c,d in entries))
                entry['uav_device_counts']=dict(collections.Counter(d for c,d in entries if c=='UAV'))
                entry['format_note']='real passband measurements; cached IQ is digitally downconverted, not native I/Q'
                entry['controller_rule']='Only UAV eligible; UAV_Controller excluded'
            if path.name == 'DroneDetect_V2.zip':
                entry['device_codes']=dict(collections.Counter(Path(i.filename).parts[2].split('_')[0] for i in members))
                entry['format_note']='native cf32 per previously audited manifest; transmitter purity not established'
            archive_summary.append(entry)
    by_serial=collections.defaultdict(list)
    for r in packs: by_serial[r['metadata'].get('SerialNumber')].append(r['pack_id'])
    anomalies=dict(label_mismatches=[dict(pack_id=r['pack_id'],folder=r['category'],xml=r['metadata'].get('Drone'))
        for r in packs if r['category'] != r['metadata'].get('Drone')],
        duplicate_pack_serials={k:v for k,v in by_serial.items() if len(v)>1},
        serial_is_pack_not_physical_airframe=True)
    summary=dict(time=time.time(),status='METADATA_INVENTORY_AND_CONTROLLER_EXCLUSION_COMPLETE',
        all_categories=len(categories),all_iq_files=len(files),all_iq_bytes=sum(r['bytes'] for r in files),
        aircraft_categories=len(AIRCRAFT),aircraft_iq_files=len(admitted),aircraft_iq_bytes=sum(r['bytes'] for r in admitted),
        aircraft_packs=sum(r['aircraft_category'] for r in packs),
        excluded_categories=len(categories)-len(AIRCRAFT),
        aircraft_pack_roles=dict(collections.Counter(r['role'] for r in packs if r['aircraft_category'])),
        metadata_only=True, new_rf_arrays_read=False, transmitter_purity_established=False,
        archive_payload_integrity_verified=False, protected_confirmation_opened=False)
    write('INVENTORY.json',dict(summary=summary,categories=categories,packs=packs,files=files,
        other_archives=archive_summary,metadata_anomalies=anomalies))
    write('AIRCRAFT_CANDIDATES.json',[r for r in packs if r['aircraft_category']])
    write('EXCLUDED_CONTROLLERS.json',[r for r in packs if not r['aircraft_category']])
    with (P/'RFUAV_FILES.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
    write('INVENTORY_AUDIT.json',dict(summary,inventory_sha256=digest(P/'INVENTORY.json'),
        all_files_mapped_to_one_pack=True,controller_in_aircraft_list=False))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.data_root.resolve(); RAW=ROOT/'extracted/rfuav'; P=args.output.resolve()
    P.mkdir(parents=True,exist_ok=True)
    main()
