"""Frozen epoch-1 long-context models on already seen TRAIN48, CPU only."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback

import numpy as np
import torch

import phase_packing as pp
import phase_data as data
import phase_evaluation as evaluation
import watch_epochs as watch
from native_data import NativeMixtures


def run(study, root, public):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate diagnostic registration')
    root.mkdir(parents=True, exist_ok=True)
    parent = watch.read(study/'PROTOCOL.json')
    train = NativeMixtures(parent['preparation'], 'train_pack', 1)
    used, indices = Counter(), []
    for i, row in enumerate(train.rows):
        n = int(row['count'])
        names = tuple(train.library.clips[int(j)]['category'] for j in row['indices'][:n])
        key = names + tuple(map(float, row['levels'][:n]))
        if used[key] < 2:
            used[key] += 1; indices.append(i)
    if len(indices) != 48:
        raise ValueError('Expected existing TRAIN48 rule')
    source = dict(parent['source_sha256'])
    source[str(Path(__file__).relative_to(watch.ROOT))] = watch.digest(Path(__file__))
    checkpoints = {a: watch.digest(study/a/'ACTUAL_001.pt') for a in ('local','long')}
    validations = {a: watch.digest(study/a/'VALIDATION_001.json') for a in checkpoints}
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel) != digest:
            raise ValueError('Source changed')
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watch.ROOT/rel, target)
    protocol = dict(status='REGISTERED_FROZEN_PHASE_E1_TRAIN48_INFERENCE', source_sha256=source,
        study_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        preparation_sha256=parent['preparation_sha256'], checkpoints_sha256=checkpoints,
        validation_scores_sha256=validations, checkpoint_epoch=1, checkpoint_updates=75,
        parameters_per_arm=pp.PARAMETERS, train_indices=indices, seen_train_epoch=1,
        selection='Same prior TRAIN48: first two per ordered category tuple and nominal levels',
        purpose='Already seen training-case fit; not generalization or independent test',
        backend='CPU float32 two threads, full long input and original outputs',
        validation_use='First existing1/2/3 case per arm only for CPU/GPU original inference agreement',
        model_updates=0, heldout_read=False, one_forward_per_case=True)
    watch.write(root/'PROTOCOL.json', protocol)
    validation = NativeMixtures(parent['preparation'], 'validation_pack', 1)
    records, backend_checks, started = [], [], time.time()
    for arm in checkpoints:
        saved = torch.load(study/arm/'ACTUAL_001.pt', map_location='cpu', weights_only=False)
        if saved['epoch']!=1 or saved['arm']!=arm or saved['protocol_sha256']!=protocol['study_protocol_sha256']:
            raise ValueError('Wrong checkpoint')
        net=pp.PhasePackedWaveNet(arm).eval(); net.load_state_dict(saved['model']); del saved
        def calculate(dataset, index):
            item = data.batch(data.example(dataset,index),'cpu')
            row = dataset.rows[index]
            clips=[dataset.library.clips[int(i)] for i in row['indices'][:int(row['count'])]]
            return evaluation.row_metrics(data.predict(net,item), item, row, clips, index)
        with torch.inference_mode():
            expected=watch.read(study/arm/'VALIDATION_001.json')['rows']
            for count in (1,2,3):
                reference=next(r for r in expected if r['count']==count)
                actual=calculate(validation, reference['index'])
                np.testing.assert_allclose(actual['nmse'],reference['nmse'],rtol=1e-3,atol=1e-5)
                np.testing.assert_allclose(actual['si_sdr'],reference['si_sdr'],rtol=0,atol=.01)
                backend_checks.append(dict(arm=arm,index=reference['index'],passed=True))
            for index in indices:
                records.append(dict(arm=arm,**calculate(train,index)))
                watch.write(root/'PARTIAL.json',dict(rows=records,partial=True))
                watch.write(root/'STATE.json',dict(status='CPU_TRAIN_INFERENCE',arm=arm,
                    cases=len(records),total=96,seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        del net;gc.collect()
    summaries=[]
    def mean(values):
        return statistics.mean(values) if all(v is not None for v in values) else None
    for arm in checkpoints:
        for count in (1,2,3):
            rs=[r for r in records if r['arm']==arm and r['count']==count]
            summaries.append(dict(arm=arm,count=count,cases=len(rs),
                mean_nmse=mean([v for r in rs for v in r['nmse']]),
                mean_si_sdr=mean([v for r in rs for v in r['si_sdr']]),
                weakest_nmse=mean([r['nmse'][r['weakest_index']] for r in rs]),
                background_energy_to_mixture=mean([r['background_nmse'] for r in rs])))
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen source changed')
    for arm,digest in checkpoints.items():
        if watch.digest(study/arm/'ACTUAL_001.pt')!=digest:
            raise ValueError('Checkpoint changed')
    result=dict(protocol,status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),
        rows=records,summary=summaries,backend_checks=backend_checks,seconds=time.time()-started)
    watch.write(root/'COMPLETE.json',result);watch.write(public.with_suffix('.json'),result)
    fmt=lambda v: '미정의' if v is None else f'{v:.6f}'
    lines=['# 긴 복소문맥 모델 e1: 이미 본 TRAIN48 적합도', '',
        '두 군의 e1/75업데이트 가중치를 고정하고, 실제 첫 epoch에서 학습한 같은 48혼합을 CPU로 추론했다. '
        '성분 수·기종·정답은 추론 입력에 사용하지 않았다. 기존 검증 각 성분 수의 첫 행은 CPU/GPU '
        '원래 점수 재현 확인에만 사용했다. 학습을 추가하거나 보류 I/Q를 읽지 않았다.', '',
        '| 군 | 성분 수 | 혼합 수 | NMSE ↓ | 복소 SI-SDR ↑ dB | 약신호 NMSE ↓ | 배경 출력/혼합 에너지 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['arm']}|{s['count']}|{s['cases']}|{fmt(s['mean_nmse'])}|{fmt(s['mean_si_sdr'])}|{fmt(s['weakest_nmse'])}|{fmt(s['background_energy_to_mixture'])}|")
    lines+=['','자료 구성이 다른 TRAIN48 평균과 검증630 평균을 일반화 격차의 정량값으로 바로 빼지 않는다. '
        '이 진단은 이미 본 혼합의 적합 여부를 확인한다. 실패 원인을 출력 표현·최적화·시간 문맥 중 하나로 '
        '확정하지 않는다. 원래 630개 평가와 등록된 모델 선택 규칙은 유지한다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',summary=summaries,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','run','public'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.study.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
