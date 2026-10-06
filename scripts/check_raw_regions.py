"""Read bounded cf32 regions of aircraft train/validation candidates only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from inventory_rfuav import AIRCRAFT


def check(inventory,output):
    if output.exists():raise FileExistsError(output)
    x=json.loads(inventory.read_text());rows=[]
    for item in x['files']:
        if not item['aircraft_category'] or item['role']=='heldout_model_not_for_selection':continue
        if item['category'] not in AIRCRAFT or item['role'] not in {'train_pack','validation_pack'}:
            raise ValueError('Unknown category/role; no waveform read')
        path=Path(item['path']);stat=path.stat()
        if stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError(f'Raw source changed: {path}')
        if item['data_type']!='Complex Float' or stat.st_size%8:
            raise ValueError('Not declared interleaved float32 I/Q')
        n=stat.st_size//8;offsets=sorted({0,max(0,n//2-8192),max(0,n-16384)});checks=[]
        with path.open('rb') as stream:
            for off in offsets:
                stream.seek(off*8);raw=stream.read(min(16384,n-off)*8)
                arr=np.frombuffer(raw,dtype='<c8')
                checks.append(dict(offset_samples=off,bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),finite=bool(np.isfinite(arr).all()),
                    mean_power=float(np.mean(np.abs(arr.astype(np.complex128))**2))))
        rows.append(dict(path=item['relative_path'],pack_id=item['pack_id'],category=item['category'],
            role=item['role'],samples=n,checks=checks,finite=all(c['finite'] for c in checks),
            three_region_digest=hashlib.sha256(''.join(c['sha256'] for c in checks).encode()).hexdigest()))
    groups={}
    for row in rows:groups.setdefault(row['three_region_digest'],[]).append(row)
    duplicates=[[dict(path=r['path'],role=r['role'],pack_id=r['pack_id']) for r in g]
                 for g in groups.values() if len(g)>1]
    result=dict(status='BOUNDED_RAW_QC_COMPLETE',time=time.time(),files_read=len(rows),
        bytes_read=sum(c['bytes'] for r in rows for c in r['checks']),all_finite=all(r['finite'] for r in rows),
        matched_three_region_groups=duplicates,full_file_identity_verified=False,
        heldout_model_payload_read=False,protected_confirmation_opened=False,rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=check(args.inventory,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
