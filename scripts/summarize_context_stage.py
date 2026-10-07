"""Summarize saved validation rows by original receiver band; never read IQ."""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from drone_rf.data import sha256
from drone_rf.mixture_constraints import band_group


def mean(values):
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return float(np.mean(values))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--epoch',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    r=args.experiment
    config=json.loads((r/'preparation/PREPARATION.json').read_text())
    manifest=Path(config['manifest'])
    if sha256(manifest)!=config['manifest_sha256']:raise ValueError('Manifest changed')
    clips=json.loads(manifest.read_text())['clips']
    path=r/'preparation/VALIDATION.npy'
    if sha256(path)!=config['files']['VALIDATION.npy']:raise ValueError('Validation schedule changed')
    schedule=np.load(path,allow_pickle=False)
    arms={}
    for arm in config['arms']:
        folder=r/'run'/arm
        stage=json.loads((folder/f'STAGE_{args.epoch:03d}.json').read_text())
        selected=folder/f'SELECTED_{args.epoch:03d}.json'
        d=json.loads(selected.read_text())
        if stage['checkpoint_sha256']!=sha256(folder/f'SELECTED_{args.epoch:03d}.pt'):
            raise ValueError('Selected checkpoint changed')
        if len(d['rows'])!=len(schedule) or {row['index'] for row in d['rows']}!=set(range(len(schedule))):
            raise ValueError('Missing/duplicate validation indices')
        groups=collections.defaultdict(list)
        for row in d['rows']:
            source=schedule[row['index']]
            count=int(source['count'])
            if count!=row['construction_count'] or len(row['nmse'])!=count:
                raise ValueError('Incorrect construction labels')
            bands={band_group(clips[int(i)]['center_hz']) for i in source['indices'][:count]}
            label=next(iter(bands)) if len(bands)==1 else 'cross_band_synthetic'
            groups[(count,label)].append(row)
        strata=[]
        for (count,label),rows in sorted(groups.items()):
            strata.append(dict(count=count,band=label,cases=len(rows),
                component_nmse=mean([v for row in rows for v in row['nmse']]),
                component_si_sdr=mean([v for row in rows for v in row['si_sdr']]),
                si_sdr_improvement=None if count==1 else mean([v for row in rows for v in row['si_sdr_improvement']]),
                count_accuracy=mean([int(row['construction_count']==row['predicted_construction_count']) for row in rows])))
        reconstructed=mean([mean([v for row in d['rows'] if row['construction_count']==count for v in row['nmse']])
                            for count in (1,2,3)])
        if not np.isclose(reconstructed,d['macro_component_nmse'],rtol=1e-12):
            raise ValueError('Saved aggregate failed replay')
        arms[arm]=dict(selected_epoch=d['epoch'],completed_budget=args.epoch,
            updates=stage['updates'],macro_component_nmse=d['macro_component_nmse'],
            strata=strata,selection_sha256=sha256(selected),checkpoint_sha256=stage['checkpoint_sha256'])
    result=dict(status='SAVED_STAGE_STRATIFIED_BY_ORIGINAL_RF_BAND',time=time.time(),
        source_sha256=sha256(Path(__file__)),validation_schedule_sha256=sha256(path),
        budget_epochs=args.epoch,arms=arms,iq_read=False,heldout_evaluated=False,
        independent_test=False,physical_drone_count_evaluated=False,
        interpretation='Descriptive strata of a prior mixed-band validation. Different strata are not a causal band ablation. '
            'One seed; selected checkpoints minimize the original whole-validation NMSE, not each subgroup.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
