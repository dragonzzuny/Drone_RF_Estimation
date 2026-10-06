"""CPU-only format/resampling probe of the sealed TRAIN candidates, never heldouts."""
import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from scipy.signal import resample_poly


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def power(x):
    return float(np.mean(x.real.astype(np.float64)**2 + x.imag.astype(np.float64)**2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output; preserve previous probe')
    seal = json.loads((args.selection/'FREEZE.json').read_text())
    for name, digest in seal['files'].items():
        if sha(args.selection/name) != digest:
            raise ValueError('Selection changed: ' + name)
    spec = json.loads((args.selection/'SELECTION.json').read_text())
    all_rows = json.loads((args.selection/'DRONEDETECT_ROLES.json').read_text())
    rows = [r for r in all_rows if r['role']=='train_candidate']
    assert len(rows)==51
    assert all(r['code']!='DIS' and r['condition']=='CLEAN' and r['repeat']<3 for r in rows)
    args.output.mkdir(parents=True)
    (args.output/'cache').mkdir()
    dump(args.output/'RECEIPT.json', dict(pid=os.getpid(), started=time.time(),
        source_sha256=sha(Path(__file__)), selection_sha256=sha(args.selection/'SELECTION.json'),
        scipy=scipy_version, numpy=np.__version__, cuda_used=False,
        native_samples=1_260_600, target_samples=2_097_152, output_fs_hz=100_000_000,
        resample=dict(up=5,down=3,window=['kaiser',5.0],padtype='line',crop='symmetric center'),
        prefix_only=True, independent_session_claimed=False, new_training_admission=False))
    results=[]
    started=time.time()
    with zipfile.ZipFile(spec['dronedetect_archive']) as z:
        for index,row in enumerate(rows):
            info=z.getinfo(row['member'])
            assert info.CRC==row['crc32'] and info.file_size==row['bytes']
            with z.open(row['member']) as stream:
                body=stream.read(1_260_600*8)
            if len(body)!=1_260_600*8:
                raise ValueError('Short IQ prefix')
            x=np.frombuffer(body,dtype='<c8')
            if not np.isfinite(x).all() or power(x)<=0:
                raise ValueError('Invalid or zero prefix: '+row['member'])
            y=resample_poly(x,5,3,window=('kaiser',5.0),padtype='line')
            crop=(len(y)-2_097_152)//2
            y=np.ascontiguousarray(y[crop:crop+2_097_152],dtype=np.complex64)
            assert len(y)==2_097_152 and np.isfinite(y).all()
            name=f'{index:03d}_{row["code"]}_{row["mode"]}_{row["repeat"]}.npy'
            target=args.output/'cache'/name
            np.save(target,y,allow_pickle=False)
            p=power(y)
            mean=y.mean(dtype=np.complex128)
            results.append(dict(member=row['member'], source_id=row['source_id'],
                role=row['role'], prefix_sha256=hashlib.sha256(body).hexdigest(),
                cache=name, cache_sha256=sha(target), native_samples=len(x), output_samples=len(y),
                native_power=power(x), resampled_power=p,
                resampled_to_native_power_db=float(10*np.log10(p/power(x))),
                dc_fraction=float(abs(mean)**2/p),
                rail_proxy_fraction=float(np.mean((np.abs(x.real)>=0.999)|(np.abs(x.imag)>=0.999))),
                center_hz=row['center_hz'], fs_hz=100_000_000,
                phase_preserved_by_complex_resampling=True,
                airframe_only_emission_verified=False))
            dump(args.output/'PROGRESS.json',dict(stage='CPU_TRAIN_PREFIX_PROBE',
                completed=index+1,total=len(rows),time=time.time(),seconds=time.time()-started))
    # Verify addition after conversion at common sample rate; no truth is fed to a model.
    examples=[]
    for code in ['AIR','INS','MIN']:
        r=next(r for r in results if r['source_id']=='DroneDetect:'+code)
        x=np.load(args.output/'cache'/r['cache'],mmap_mode='r',allow_pickle=False)
        examples.append(np.asarray(x[:63872],dtype=np.complex128)/np.sqrt(r['resampled_power']))
    rng=np.random.default_rng(0)
    probes=[]
    for count,levels in [(1,[0]),(2,[0,-10]),(2,[0,10]),(3,[0,-10,10])]:
        references=np.stack([examples[i]*10**(levels[i]/20)*np.exp(1j*rng.uniform(0,2*np.pi))
                             for i in range(count)])
        mixed=references.sum(axis=0)
        recovered=mixed-references[1:].sum(axis=0)
        rel=power(recovered-references[0])/power(references[0])
        assert rel<1e-24
        probes.append(dict(count=count,relative_db=levels,oracle_subtraction_nmse=rel,
            same_native_center_hz=2_437_500_000, separation_model_evaluated=False))
    result=dict(status='CPU_NATIVE_IQ_AND_COMPLEX_MIXTURE_INPUTS_CHECKED', time=time.time(),
        seconds=time.time()-started, files=len(results), records=results, mixture_probes=probes,
        validation_or_heldout_payloads_read=False, full_archive_crc_verified=False,
        no_new_training_admission=True, separation_model_evaluated=False,
        interpretation='Format and input arithmetic only. Prefix quality does not prove whole-file quality, airframe-only transmission, independent sessions, or learned separation.')
    dump(args.output/'AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
