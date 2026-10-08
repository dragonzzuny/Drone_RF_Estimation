"""Bounded CPU audit of long crops and source-bandwidth metadata.

Reads TRAIN I/Q only. Validation grouping is metadata only; no Autel access.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from long_data import admitted_dataset
from run_comparison import PREPARATION
from waveform_models import LONG_SAMPLES,TARGET_SAMPLES
from drone_rf.data import sha256
from drone_rf.context_training_data import write_json


def run(root):
    if any(root.iterdir()): raise RuntimeError('Fresh audit directory required')
    start=time.time()
    data=admitted_dataset(PREPARATION,'train_pack',1)
    selected=[]
    for count in (1,2,3):
        indices=np.flatnonzero(data.rows['count']==count)
        selected.extend(indices[np.linspace(0,len(indices)-1,6,dtype=int)].tolist())
    rows=[]
    for index in selected:
        item=data[index]; row=data.rows[index]; k=int(row['count'])
        assert all(data.library.clips[int(j)]['role']=='train_pack' for j in row['indices'][:k])
        offset=int(item['crop_start'])-int(item['long_start'])
        assert np.array_equal(item['long_mixture'][offset:offset+TARGET_SAMPLES],item['mixture'])
        assert np.array_equal(item['references'].sum(0),item['mixture'])
        rows.append(dict(index=index,count=k,long_start=int(item['long_start']),target_offset=offset,
            left_context_ms=offset/1e5,right_context_ms=(LONG_SAMPLES-offset-TARGET_SAMPLES)/1e5,
            mixture_sha256=hashlib.sha256(item['mixture'].tobytes()).hexdigest(),
            long_mixture_sha256=hashlib.sha256(item['long_mixture'].tobytes()).hexdigest()))
    manifest=json.loads((PREPARATION/'CACHE_MANIFEST.json').read_text())
    packs=collections.defaultdict(set)
    for clip in manifest['clips']: packs[clip['role']].add(clip['pack_id'])
    # Actual schedule coordinates across both roles; no validation I/Q read.
    position_summary={}
    for filename in ('TRAIN.npy','VALIDATION.npy'):
        schedule=np.load(PREPARATION/filename,allow_pickle=False)
        starts=np.clip(schedule['crop_start']-(LONG_SAMPLES-TARGET_SAMPLES)//2,0,2097152-LONG_SAMPLES)
        left=schedule['crop_start']-starts
        right=LONG_SAMPLES-left-TARGET_SAMPLES
        position_summary[filename]=dict(cases=len(schedule),left_ms_min=float(left.min()/1e5),
            right_ms_min=float(right.min()/1e5),symmetric_context_cases=int(np.sum(left==right)))
    result=dict(status='LONG_INPUT_CPU_AUDIT_COMPLETE',seconds=time.time()-start,rows=rows,
        training_iq_cases=len(rows),validation_iq_read=False,heldout_iq_read=False,
        pack_metadata={k:sorted(v) for k,v in packs.items()},position_summary=position_summary,
        source_sha256=sha256(Path(__file__)),manifest_sha256=sha256(PREPARATION/'CACHE_MANIFEST.json'),
        scope='18 TRAIN mixtures plus all-schedule coordinate checks; not a waveform-generalization result')
    write_json(root/'COMPLETE.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pack_metadata')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-4:-2]));os.nice(10)
    args.run.mkdir(parents=True,exist_ok=True);run(args.run)
