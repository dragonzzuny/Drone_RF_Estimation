"""CPU audit of all scheduled TRAIN powers; saved validation metadata only.

No model inference, heldout waveforms, activity labels, or schedule changes.
The local/full power ratio describes fluctuations, not signal/noise truth.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import fit_diagnostic
import watch_epochs as watch
from native_data import NativeMixtures


def summary(rows):
    result = []
    for count in (1, 2, 3):
        sub = [r for r in rows if r['count'] == count]
        gaps = np.array([r['gap_db'] for r in sub])
        edges = [0, 10, 20, 30, 40, 50, math.inf]
        result.append(dict(count=count, cases=len(sub),
            median_gap_db=float(np.median(gaps)), p95_gap_db=float(np.quantile(gaps, .95)),
            maximum_gap_db=float(gaps.max()),
            bins=[dict(low_db=lo, high_db=hi if math.isfinite(hi) else None,
                       cases=int(((gaps>=lo)&(gaps<hi)).sum())) for lo, hi in zip(edges[:-1],edges[1:])]))
    return result


def run(study, root, public):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate audit registration')
    root.mkdir(parents=True, exist_ok=True)
    parent = watch.read(study/'PROTOCOL.json')
    validation = study/'unet_mean/VALIDATION_004.json'
    watch.validate(validation, None)
    source = dict(parent['source_sha256'])
    for path in (Path(__file__), Path(watch.__file__)):
        source[str(path.relative_to(watch.ROOT))] = watch.digest(path)
    protocol = dict(status='REGISTERED_ALL_TRAIN_POWER_AUDIT', source_sha256=source,
        study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        preparation_sha256=parent['preparation_sha256'], validation_metadata_sha256=watch.digest(validation),
        epochs=[1,2,3,4,5], expected_cases=12000, expected_by_count=4000,
        source='TRAIN scheduled crop references and mixture, exact native data loader',
        floor_rule='source_power < 1e-6*max(mixture_power,1e-8), float64 CPU reductions',
        local_full_ratio='crop source power / nominal normalized full-source power weight',
        heldout_read=False, validation_waveform_read=False, model_updates=0, post_hoc=True)
    for rel, digest in source.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Frozen source changed')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, target)
    watch.write(root/'PROTOCOL.json', protocol)
    rows, began = [], time.time()
    with (root/'TRAIN_ROWS.jsonl').open('w') as stream:
        for epoch in protocol['epochs']:
            train = NativeMixtures(parent['preparation'], 'train_pack', epoch)
            if len(train) != 2400:
                raise ValueError('Training schedule changed')
            for index in range(len(train)):
                item = train[index]; count = int(item['construction_count'])
                references = item['references'][:count].astype(np.complex128)
                power = np.mean(np.abs(references)**2, axis=-1)
                mix = float(np.mean(np.abs(item['mixture'].astype(np.complex128))**2))
                if not np.isfinite(power).all() or np.any(power<=0) or not math.isfinite(mix):
                    raise ValueError('Invalid reference power')
                levels = train.rows[index]['levels'][:count].astype(np.float64)
                weights = 10**((levels-levels.max())/10)
                weights /= weights.sum()
                ratios = power/max(mix,1e-8)
                clips = [train.library.clips[int(i)] for i in train.rows[index]['indices'][:count]]
                row = dict(epoch=epoch, index=index, count=count,
                    categories=[c['category'] for c in clips], pack_ids=[c['pack_id'] for c in clips],
                    nominal_levels_db=levels.tolist(), reference_power=power.tolist(), mixture_power=mix,
                    target_mix_ratio=ratios.tolist(), floor_applied=(ratios<1e-6).tolist(),
                    near_floor=(np.abs(ratios/1e-6-1)<1e-3).tolist(),
                    crop_to_full_power_db=(10*np.log10(power/weights)).tolist(),
                    gap_db=float(10*np.log10(power.max()/power.min())))
                stream.write(json.dumps(row)+'\n'); rows.append(row)
                if len(rows)%100 == 0:
                    stream.flush()
                    watch.write(root/'STATE.json', dict(status='CPU_TRAIN_POWER_AUDIT', cases=len(rows),
                        total=12000, epoch=epoch, seconds=time.time()-began, pid=os.getpid(), time=time.time()))
    if Counter(r['count'] for r in rows) != {1:4000,2:4000,3:4000}:
        raise ValueError('Unexpected count balance')
    categories = []
    for category in sorted({c for r in rows for c in r['categories']}):
        entries = [(r,j) for r in rows for j,c in enumerate(r['categories']) if c==category]
        values = np.array([r['crop_to_full_power_db'][j] for r,j in entries])
        categories.append(dict(category=category, contributions=len(entries),
            crop_to_full_db_quantiles={str(q):float(np.quantile(values,q)) for q in (0,.05,.5,.95,1)},
            floor_applied=sum(r['floor_applied'][j] for r,j in entries)))
    saved = watch.read(validation)['rows']
    validation_rows = [dict(count=r['count'], gap_db=10*math.log10(max(r['reference_power'])/min(r['reference_power']))) for r in saved]
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Source changed during audit')
    result = dict(protocol, status='COMPLETE', protocol_sha256=watch.digest(root/'PROTOCOL.json'),
        train_rows_sha256=watch.digest(root/'TRAIN_ROWS.jsonl'), train=summary(rows),
        validation=summary(validation_rows), by_epoch=[dict(epoch=e,summary=summary([r for r in rows if r['epoch']==e])) for e in range(1,6)],
        by_category=categories, floor_contributions=sum(sum(r['floor_applied']) for r in rows),
        near_floor_contributions=sum(sum(r['near_floor']) for r in rows),
        total_contributions=sum(r['count'] for r in rows),
        minimum_target_to_mix_ratio=min(v for r in rows for v in r['target_mix_ratio']), seconds=time.time()-began)
    watch.write(root/'COMPLETE.json', result); watch.write(public.with_suffix('.json'), result)
    lines = ['# 학습 전체의 국소 전력차와 손실 분모 바닥값 진단', '',
        'RFUAV의 고정 TRAIN 12,000개를 CPU에서 읽었다. 검증은 기존 630행의 저장된 전력 수치만 '
        '사용했으며 검증·보류 I/Q는 열지 않았다. 학습 순서·자료·손실·선택 규칙은 변경하지 않았다.', '',
        '| 자료 | 성분 수 | 혼합 수 | 전력차 중앙값 dB | 95백분위 dB | 최대 dB | 20dB 이상 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('train','validation'):
        for s in result[name]:
            high=sum(b['cases'] for b in s['bins'] if b['low_db']>=20)
            lines.append(f"|{name}|{s['count']}|{s['cases']}|{s['median_gap_db']:.2f}|{s['p95_gap_db']:.2f}|{s['maximum_gap_db']:.2f}|{high}|")
    lines += ['', f"학습의 총 {result['total_contributions']}개 기록 기여 중 분모 바닥값이 적용되는 것은 "
        f"{result['floor_contributions']}개다. 경계값의 ±0.1% 안에 드는 것은 {result['near_floor_contributions']}개이며 "
        f"최소 정답/혼합 전력비는 {result['minimum_target_to_mix_ratio']:.6g}다. CPU float64 집계다.", '',
        '전력차 분포의 차이는 학습·검증 조건 차이의 기술적 진단이다. 원인 효과를 입증하지 않으며 '
        '큰 전력차 사례를 평가에서 제거하지 않는다. 짧은 창/전체 기록 전력비는 시간 변동의 지표이며 '
        '송신 활동·수신 잡음의 정답 라벨로 해석하지 않는다. 상세 epoch·기종별 분포는 JSON에 보존했다.', '']
    watch.write(public.with_suffix('.md'), '\n'.join(lines))
    watch.write(root/'STATE.json', dict(status='COMPLETE', pid=os.getpid(), time=time.time()))
    print(json.dumps(dict(status='COMPLETE',train=result['train'],validation=result['validation'],
        floor_contributions=result['floor_contributions'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','run','public'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    try:run(a.study.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
