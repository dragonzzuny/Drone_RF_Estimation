"""Bounded metadata search for named LoRa/Crossfire recordings, not RF decoding."""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path

KEY = re.compile(r'(?i)(?<![a-z])(lora|crossfire|cross[-_ ]fire|expresslrs|elrs|crsf)(?![a-z])')
RAW = {'.iq', '.dat', '.bin', '.mat', '.h5', '.hdf5', '.cfile', '.sigmf-data', '.npy', '.npz'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path('/media/pyj/Elements/uav_rf_research_20260914')
    scans, matches, errors = [], [], []
    for p in [Path('/home/pyj/datasets'), root]:
        result = subprocess.run(['rg', '--files', '--hidden', '-g', '!.git/**',
            '-g', '!**/site-packages/**', '-g', '!**/node_modules/**',
            '-g', '!**/source_snapshot/**', '-g', '!**/rf_detection_references/**', str(p)],
            text=True, capture_output=True, timeout=180)
        paths = result.stdout.splitlines()
        scans.append(dict(root=str(p), files=len(paths), returncode=result.returncode,
            exclusions=['.git', 'site-packages', 'node_modules', 'source_snapshot', 'rf_detection_references'],
            symlink_directories_followed=False))
        if result.returncode not in (0, 1):
            errors.append(dict(root=str(p), error=result.stderr[:1000]))
        for path in paths:
            if KEY.search(path):
                matches.append(dict(origin='filesystem_name', path=path,
                    raw_extension_candidate=Path(path).suffix.lower() in RAW))
    # The acquisition backup has large software trees: inspect its existing file
    # catalogues, not private admin files, credentials, or archive payloads.
    for folder in ['acquisition_pc_archive_20260915_v1', 'acquisition_pc_supplement_20260915_v1']:
        p = Path('/media/pyj/Elements/uav_analysis') / folder / 'SOURCE_PLAN.json'
        body = p.read_bytes()
        rows = json.loads(body)['files']
        scans.append(dict(catalogue=str(p), files=len(rows), sha256=hashlib.sha256(body).hexdigest()))
        for row in rows:
            path = row['path']
            if KEY.search(path):
                matches.append(dict(origin='backup_catalogue_name', path=path,
                    raw_extension_candidate=Path(path).suffix.lower() in RAW))
    archives = []
    for p in sorted((root / 'archives').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            members = [i for i in z.infolist() if not i.is_dir()]
            rows = [(i.filename, i.file_size, i.CRC) for i in members]
            archives.append(dict(archive=p.name, members=len(rows),
                central_index_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                extensions=dict(collections.Counter(Path(i.filename).suffix for i in members))))
            for i in members:
                if KEY.search(i.filename):
                    matches.append(dict(origin='zip_member_name', archive=p.name,
                        path=i.filename, raw_extension_candidate=Path(i.filename).suffix.lower() in RAW))
    meta_paths = subprocess.run(['rg', '--files', '/home/pyj/datasets/rf_detection_captures',
        '-g', '*.sigmf-meta', '-g', '!**/source_snapshot/**'], capture_output=True, text=True, check=True).stdout.splitlines()
    centers, rates = collections.Counter(), collections.Counter()
    metadata_matches = []
    for name in meta_paths:
        p = Path(name)
        if p.stat().st_size > 2_000_000:
            errors.append(dict(path=name, error='Metadata size cap exceeded'))
            continue
        body = p.read_text()
        d = json.loads(body)
        for c in d.get('captures', []):
            if 'core:frequency' in c:
                centers[str(c['core:frequency'])] += 1
        if 'core:sample_rate' in d.get('global', {}):
            rates[str(d['global']['core:sample_rate'])] += 1
        if KEY.search(body):
            metadata_matches.append(name)
    result = dict(status='NAMED_SIGNAL_METADATA_SEARCH', time=time.time(),
        scans=scans, archives=archives, name_matches=matches,
        raw_extension_name_matches=[r for r in matches if r['raw_extension_candidate']],
        sigmf=dict(files=len(meta_paths), center_frequency_counts=dict(centers),
                   sample_rate_counts=dict(rates), keyword_metadata_paths=metadata_matches),
        errors=errors, iq_payloads_read=False, protected_confirmation_payloads_read=False,
        interpretation='No name match is not proof of RF absence. Unlabelled captures and module variants remain unverified; repository prose calling TX16S ELRS is not a capture annotation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(scans=scans, raw_name_matches=result['raw_extension_name_matches'],
        all_keyword_name_matches=len(matches), sigmf=result['sigmf'], errors=errors), ensure_ascii=False))


if __name__ == '__main__':
    main()
