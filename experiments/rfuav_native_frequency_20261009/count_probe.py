"""Exploratory CPU count diagnosis triggered by native e1 count3 accuracy0.

Uses mixture-only cached features; no waveform network or threshold is changed.
Construction count is not a physical-aircraft-count annotation.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from native_data import HERE,ROOT,LEGACY,NativeMixtures,sha256,write_json
from native import band_for


def features(data):
    means=[];stds=[];bands=[]
    for start in range(0,len(data),100):
        x=np.asarray(data.features[start:start+100],dtype=np.float64)
        means.append(x.mean(-1));stds.append(x.std(-1))
    for row in data.rows:
        bands.append(band_for(data.library.clips[int(row['indices'][0])]['center_hz']))
    return np.concatenate(means),np.concatenate(stds),data.rows['count'].astype(np.int64),np.asarray(bands)


def score(true,predicted):
    matrix=confusion_matrix(true,predicted,labels=[1,2,3])
    return dict(accuracy=float(np.mean(true==predicted)),confusion_matrix=matrix.tolist(),labels=[1,2,3],
        accuracy_by_count={str(k):float(np.mean(predicted[true==k]==k)) for k in (1,2,3)},
        predictions=predicted.tolist())


def run(root):
    protocol_path=root/'COUNT_PROTOCOL.json'
    if protocol_path.exists():raise RuntimeError('Refuse ambiguous duplicate count probe')
    source_files=[HERE/'count_probe.py',HERE/'native_data.py',HERE/'native.py',
                  *LEGACY.glob('*.py'),*(LEGACY/'vendor/drone_rf').glob('*.py')]
    sources={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    trigger=root/'gpu/VALIDATION_001.json'
    protocol=dict(status='REGISTERED_EXPLORATORY_COUNT_PROBE',source_sha256=sources,
        triggered_by='native e1 count3 accuracy0; adaptively chosen development diagnosis',
        trigger_sha256=sha256(trigger),preparation_protocol_sha256=sha256(root/'PREP_PROTOCOL.json'),
        arms=['mean65','mean_std130'],classifier='StandardScaler(TRAIN fit) + multinomial LogisticRegression C1 lbfgs max_iter2000 random_state0',
        training='all12000 scheduled mixtures; no independent-record multiplication claim',
        validation='same630 development mixtures; no model/threshold selection or heldout access',
        comparisons=['band-only majority from TRAIN','existing waveform-network count head'],
        inputs='only mean, or mean plus population std, of mixture power-context feature sequence',
        temporal_claim='std retains variability, not token order or hopping recurrence',
        physical_aircraft_count=False,waveform_improvement_claim=False)
    for rel,digest in sources.items():
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=digest:raise RuntimeError('Snapshot conflict')
        if not target.exists():shutil.copyfile(ROOT/rel,target)
    write_json(protocol_path,protocol);protocol_sha=sha256(protocol_path)
    while not (root/'PREP_COMPLETE.json').exists():
        if (root/'PREP_FAILURE.json').exists():raise RuntimeError('Preparation failed')
        time.sleep(10)
    started=time.time();train=[features(NativeMixtures(root/'preparation','train_pack',e)) for e in range(1,6)]
    mean,std,y,band=[np.concatenate([v[k] for v in train]) for k in range(4)]
    vm,vs,vy,vband=features(NativeMixtures(root/'preparation','validation_pack',1))
    by_band={b:Counter(y[band==b]).most_common(1)[0][0] for b in sorted(set(band))}
    results={'band_only_majority':score(vy,np.array([by_band[b] for b in vband]))}
    (root/'count').mkdir(exist_ok=True)
    for arm in protocol['arms']:
        x=np.concatenate([mean,std],1) if arm=='mean_std130' else mean
        vx=np.concatenate([vm,vs],1) if arm=='mean_std130' else vm
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);classifier.fit(x,y)
        predicted=classifier.predict(vx);results[arm]=score(vy,predicted)
        results[arm]['training_accuracy']=float(classifier.score(x,y))
        results[arm]['iterations']=classifier[-1].n_iter_.tolist()
        joblib.dump(classifier,root/'count'/f'{arm}.joblib')
    for rel,digest in sources.items():
        if sha256(ROOT/rel)!=digest:raise RuntimeError('Count source changed')
    write_json(root/'count/RESULT.json',dict(status='COMPLETE',protocol_sha256=protocol_sha,results=results,
        train_cases=len(y),validation_cases=len(vy),seconds=time.time()-started,
        no_waveform_change=True,physical_aircraft_count=False,independent_test=False,
        native_preparation_sha256=sha256(root/'preparation/PREPARATION.json')))
    write_json(root/'COUNT_COMPLETE.json',dict(status='COMPLETE',protocol_sha256=protocol_sha,time=time.time()))
    print(json.dumps({k:{q:v for q,v in r.items() if q!='predictions'} for k,r in results.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{12,13});os.nice(15)
    try:run(args.run.resolve())
    except Exception:
        write_json(args.run/'COUNT_FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
