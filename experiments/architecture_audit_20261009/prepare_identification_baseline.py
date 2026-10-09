"""Freeze a TRAIN-only spectral linear classifier before separated-output tests.

Only the existing RFUAV development roles are opened. This preliminary result
tests single-record category discrimination, not separation or physical-device
fingerprinting. No hyperparameter or feature selection uses validation data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures
import watch_epochs as watch


def features(wave):
    wave=np.asarray(wave,dtype=np.complex128)
    if wave.ndim!=1 or len(wave)!=63872 or not np.isfinite(wave).all():
        raise ValueError('Expected one finite native complex crop')
    frames=np.lib.stride_tricks.sliding_window_view(wave,512)[::128]
    window=np.hanning(513)[:-1]
    spectrum=np.fft.fftshift(np.fft.fft(frames*window,axis=-1),axes=-1)
    power=np.abs(spectrum)**2
    power=power.reshape(len(frames),128,4).mean(-1)
    scale=float(power.mean())
    if not scale>0:
        raise ValueError('Zero-energy recording')
    normalized=power/scale
    mean=normalized.mean(0)
    feature=np.concatenate((np.log10(np.maximum(mean,1e-8)),
                            np.log1p(normalized.std(0)/np.maximum(mean,1e-8))))
    if feature.shape!=(256,) or not np.isfinite(feature).all():
        raise ValueError('Invalid spectral features')
    return feature


def fit_classifier(x,y,classes,regularization=.01):
    counts=np.bincount(y,minlength=len(classes))
    if np.any(counts==0):raise ValueError('Missing training category')
    weights=1/counts[y];weights=weights/weights.sum()
    mean=(x*weights[:,None]).sum(0)
    scale=np.sqrt(((x-mean)**2*weights[:,None]).sum(0)).clip(1e-6)
    design=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    target=np.eye(len(classes))[y]
    penalty=np.eye(design.shape[1])*regularization;penalty[-1,-1]=0
    coefficients=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,
                                 design.T@(weights[:,None]*target))
    return dict(mean=mean,scale=scale,coefficients=coefficients,classes=np.asarray(classes))


def predict(classifier,x):
    design=np.column_stack(((x-classifier['mean'])/classifier['scale'],np.ones(len(x))))
    return (design@classifier['coefficients']).argmax(1)


def check_math():
    rng=np.random.default_rng(2909)
    wave=rng.normal(size=63872)+1j*rng.normal(size=63872)
    np.testing.assert_allclose(features(wave),features(wave*(.37+1.81j)),rtol=1e-9,atol=1e-9)
    x=np.repeat(np.eye(5),10,axis=0)+rng.normal(scale=.01,size=(50,5));y=np.repeat(np.arange(5),10)
    model=fit_classifier(x,y,list('abcde'));assert np.array_equal(predict(model,x),y)
    return dict(complex_gain_invariance=True,synthetic_class_mapping=True,recorded_reads=0)


def unique_examples(data):
    examples={}
    for row in data.rows:
        for index in row['indices'][:int(row['count'])]:
            index=int(index);start=int(row['crop_start']);clip=data.library.clips[index]
            examples[(index,start)]=dict(clip_index=index,crop_start=start,clip_id=clip['clip_id'],
                                       category=clip['category'],pack_id=clip['pack_id'])
    return [examples[key] for key in sorted(examples)]


def extract(data,examples,root,role):
    values=[];started=time.time()
    for i,row in enumerate(examples):
        wave=data.library._array(row['clip_index'])[row['crop_start']:row['crop_start']+63872]
        values.append(features(wave))
        if (i+1)%25==0 or i+1==len(examples):
            watch.write(root/'STATE.json',dict(status='CPU_REFERENCE_FEATURES',role=role,
                        examples=i+1,total=len(examples),seconds=time.time()-started,pid=os.getpid(),time=time.time()))
    return np.stack(values)


def score(classifier,x,labels,classes):
    truth=np.array([classes.index(label) for label in labels]);prediction=predict(classifier,x)
    confusion=np.zeros((len(classes),len(classes)),dtype=np.int64)
    np.add.at(confusion,(truth,prediction),1)
    recall=np.diag(confusion)/confusion.sum(1)
    return dict(examples=len(truth),accuracy=float(np.mean(truth==prediction)),
                macro_recall=float(recall.mean()),confusion=confusion.tolist(),
                per_class=[dict(category=c,examples=int(confusion[i].sum()),recall=float(recall[i]))
                           for i,c in enumerate(classes)])


def run(preparation,source_study,root,public):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate classifier preparation')
    root.mkdir(parents=True,exist_ok=True);check=check_math()
    parent=watch.read(source_study/'PROTOCOL.json')
    if watch.digest(preparation/'PREPARATION.json')!=parent['preparation_sha256']:
        raise ValueError('Different RF preparation')
    source=dict(parent['source_sha256'])
    for path in (Path(__file__),Path(watch.__file__)):
        source[str(path.relative_to(ROOT))]=watch.digest(path)
    protocol=dict(status='REGISTERED_REFERENCE_CLASSIFIER_BASELINE',source_sha256=source,
        preparation_sha256=parent['preparation_sha256'],parent_protocol_sha256=watch.digest(source_study/'PROTOCOL.json'),
        train_schedule='epoch1 only; unique(clip,crop) before feature extraction',
        validation_schedule='existing development630 mixtures; unique reference(clip,crop)',
        features='256 amplitude/constant-phase-invariant features: 128 log mean PSD bins and128 log1p temporal CV bins',
        stft='full complex FFT512 hop128 periodic Hann; valid frames only; fftshift; group4 frequency bins',
        sample_rate_hz=100_000_000,complex_samples=63872,
        classifier='class-balanced weighted linear ridge to one-hot labels; lambda0.01; unpenalized intercept',
        normalization='mean and standard deviation fitted only to TRAIN, balanced class weights',
        output='five known recorded-category labels, not individual-airframe fingerprints or calibrated probabilities',
        selection='fixed before examining development classification; no validation tuning',
        inference_inputs='only a single complex I/Q crop; no category, receiver-band label or recording ID',
        caveats='spectral band/receiver offsets remain observable and may be shortcuts; record/BW shift is confounded',
        source_separation_evaluated=False,heldout_read=False,synthetic_check=check,time=time.time())
    for rel,digest in source.items():
        if watch.digest(ROOT/rel)!=digest:raise ValueError('Frozen dependency changed')
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    watch.write(root/'PROTOCOL.json',protocol)
    train=NativeMixtures(preparation,'train_pack',1,use_features=False)
    train_examples=unique_examples(train);classes=sorted({r['category'] for r in train_examples})
    if len(classes)!=5:raise ValueError('Five-category development cohort changed')
    x=extract(train,train_examples,root,'train_pack')
    y=np.array([classes.index(row['category']) for row in train_examples])
    classifier=fit_classifier(x,y,classes)
    np.savez(root/'FROZEN_CLASSIFIER.npz',**classifier)
    classifier_sha=watch.digest(root/'FROZEN_CLASSIFIER.npz')
    # Freeze and receipt the classifier BEFORE opening development waveforms.
    training_score=score(classifier,x,[r['category'] for r in train_examples],classes)
    watch.write(root/'TRAIN_FROZEN.json',dict(status='FROZEN',classifier_sha256=classifier_sha,
                train_score=training_score,validation_waveforms_read=False,time=time.time()))
    np.savez(root/'TRAIN_FEATURES.npz',features=x,labels=y)
    validation=NativeMixtures(preparation,'validation_pack',1,use_features=False)
    val_examples=unique_examples(validation)
    train_packs={r['pack_id'] for r in train_examples};val_packs={r['pack_id'] for r in val_examples}
    if train_packs&val_packs:raise ValueError('Recording-group leakage')
    v=extract(validation,val_examples,root,'validation_pack')
    val_score=score(classifier,v,[r['category'] for r in val_examples],classes)
    np.savez(root/'VALIDATION_FEATURES.npz',features=v,
             labels=np.array([classes.index(r['category']) for r in val_examples]))
    watch.write(root/'EXAMPLE_METADATA.json',dict(train=train_examples,validation=val_examples))
    if watch.digest(root/'FROZEN_CLASSIFIER.npz')!=classifier_sha:raise ValueError('Classifier altered')
    for rel,digest in source.items():
        if watch.digest(ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Source changed during classifier preparation')
    complete=dict(protocol,status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),
        classifier_sha256=classifier_sha,classes=classes,train=training_score,validation=val_score,
        train_recording_groups=len(train_packs),validation_recording_groups=len(val_packs),
        reference_window_definition='deduplicated(clip_id,crop_start); windows are not independent recordings',
        feature_sha256={n:watch.digest(root/n) for n in ['TRAIN_FEATURES.npz','VALIDATION_FEATURES.npz','EXAMPLE_METADATA.json']})
    watch.write(root/'COMPLETE.json',complete);watch.write(public.with_suffix('.json'),complete)
    lines=['# 단독 기록 파형의 기종 라벨 구분: 고정 CPU 기준', '',
        '분리 출력 평가는 아직 수행하지 않았다. 기존 TRAIN 첫 epoch의 단독 정답 기여에서 '
        '중복 창을 제거해 선형 분류기를 학습하고, 가중치를 저장한 뒤 기존 개발검증의 단독 정답을 평가했다. '
        '수신 잡음도 포함된 기록 파형이며 실제 개별 기체 fingerprint 판정이 아니다.', '',
        f"학습/검증 창: {len(train_examples)}/{len(val_examples)}개; 원기록 묶음: {len(train_packs)}/{len(val_packs)}개. "
        '창 수를 독립 녹음 수로 해석하지 않는다.', '',
        '| 역할 | 정확도 | 평균 클래스 재현율 |', '|---|---:|---:|',
        f"|학습|{training_score['accuracy']:.6f}|{training_score['macro_recall']:.6f}|",
        f"|개발검증 단독 정답|{val_score['accuracy']:.6f}|{val_score['macro_recall']:.6f}|", '',
        '| 기종 라벨 | 검증 창 | 재현율 |','|---|---:|---:|']
    for row in val_score['per_class']:
        lines.append(f"|{row['category']}|{row['examples']}|{row['recall']:.6f}|")
    lines+=['','특징·정규화·정규화 계수는 검증 점수로 고르지 않았다. 스펙트럼 대역·수신 중심 차이가 '
        '분류 단서가 될 수 있으며 기종 고유 RF fingerprint의 학습을 입증하지 않는다. '
        '원기록/VTSBW 조건 차이와 한정된5범주 개발 자료의 결과다. '
        '이 기준기는 후속 분리 출력 비교에서 그대로 고정하고, 혼합·복원·정답 및 개수 선택 규칙을 구분해야 한다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',train=training_score,validation=val_score)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('preparation','source-study','run','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:run(a.preparation.resolve(),a.source_study.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
