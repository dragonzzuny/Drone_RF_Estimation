"""Prepare RFUAV-only/same-band schedules without changing an active trial."""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np

from drone_rf.context_training_data import write_json
from drone_rf.data import ScheduledMixtures,sha256
from drone_rf.mixture_constraints import validate_sources
from drone_rf.within_dataset_schedule import make_within_dataset_schedule


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--source-roles',type=Path,required=True)
    ap.add_argument('--previous-preparation',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    seal=json.loads((args.selection/'FREEZE.json').read_text())
    for name,digest in seal['files'].items():
        if sha256(args.selection/name)!=digest:raise ValueError('Selection changed')
    selection=json.loads((args.selection/'SELECTION.json').read_text())
    if sha256(args.source_roles)!=selection['parent_seal']['files']['RFUAV_EXISTING_ROLES.json']:
        raise ValueError('Source inventory changed')
    roles=json.loads(args.source_roles.read_text())
    source={r['relative_path']:r for r in roles}
    old=json.loads((args.previous_preparation/'PREPARATION.json').read_text())
    library=ScheduledMixtures(old['manifest'],old['base_schedule'],'train_pack')
    ScheduledMixtures(old['manifest'],old['base_schedule'],'validation_pack')
    if sha256(old['manifest'])!=old['manifest_sha256']:raise ValueError('Manifest changed')
    clips=[]
    for clip in library.clips:
        r=source[clip['source_path']]
        for key in ('category','role','fs_hz','center_hz','pack_id'):
            if r[key]!=clip[key]:raise ValueError('Cache/inventory mismatch: '+key)
        if r['dataset']!='RFUAV' or 'heldout' in r['role']:raise ValueError('Not RFUAV development')
        clips.append(dict(clip,dataset='RFUAV',source_id=r['source_id'],receiver_model='USRPX310'))
    if {r['device_type'] for r in selection['rf_receiver_metadata']}!={'USRPX310'}:
        raise ValueError('Unreviewed receiver model')
    experiment=next(e for e in selection['experiments'] if e['dataset']=='RFUAV')
    training=make_within_dataset_schedule(clips,experiment['training_classes'],'train_pack')
    validation=make_within_dataset_schedule(clips,experiment['training_classes'],'validation_pack',
                                           epochs=1,examples_per_count=210)
    counts={}
    for name,rows in [('train',training),('validation',validation)]:
        counter=collections.Counter()
        for row in rows:
            chosen=[clips[int(i)] for i in row['indices'][:int(row['count'])]]
            meta=validate_sources(chosen)
            if any(c['role']!=('train_pack' if name=='train' else 'validation_pack') for c in chosen):
                raise ValueError('Role leakage')
            counter[(meta['rf_band'],int(row['count']))]+=1
        counts[name]=[dict(rf_band=b,count=c,examples=n) for (b,c),n in sorted(counter.items())]
    args.output.mkdir(parents=True)
    np.save(args.output/'TRAIN.npy',training,allow_pickle=False)
    np.save(args.output/'VALIDATION.npy',validation,allow_pickle=False)
    config=dict(old,time=time.time(),gpu_training_started=False,
        mixture_policy='within_one_dataset_and_one_native_RF_band',dataset='RFUAV',
        cross_dataset_synthesis=False,cross_band_synthesis=False,
        native_center_offsets_preserved=False,
        placement='center-aligned synthetic same-band complex mixtures',
        sample_rate_conversion=False,
        selection_sha256=sha256(args.selection/'SELECTION.json'),
        source_roles_sha256=sha256(args.source_roles),
        previous_preparation_sha256=sha256(args.previous_preparation/'PREPARATION.json'),
        class_counts=experiment['class_counts'],schedule_strata=counts,
        controller_classes_excluded=True,heldout_models_read=False,
        files={n:sha256(args.output/n) for n in ['TRAIN.npy','VALIDATION.npy']})
    write_json(args.output/'PREPARATION.json',config)
    print(json.dumps(dict(status=config['status'],policy=config['mixture_policy'],
        training_examples=len(training),validation_examples=len(validation),strata=counts,
        gpu_training_started=False),ensure_ascii=False))


if __name__=='__main__':main()
