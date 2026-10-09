"""Mixture-only no-separation baselines on existing development mixtures.

Fixed before this baseline evaluation. No learned parameters, ground-truth
count/identity/gain, output rescaling, or reserved data is used in prediction.
Nonfinite SI-SDR remains explicit, never averaged after dropping zero outputs.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import time
import traceback

import numpy as np
import torch

import fit_diagnostic as fit
import watch_epochs as watch
from native_data import NativeMixtures
from drone_rf.waveform import waveform_metrics
from study import finite_values, value_status

MODES=('copy_one','equal_three','equal_four','all_background')


def prediction(mixture, mode):
    out=torch.zeros((1,4,mixture.shape[-1]),dtype=mixture.dtype)
    if mode=='copy_one': out[:,0]=mixture
    elif mode=='equal_three': out[:,:3]=mixture[:,None]/3
    elif mode=='equal_four': out[:]=mixture[:,None]/4
    elif mode=='all_background': out[:,-1]=mixture
    else: raise ValueError('Unknown fixed baseline')
    return out


def score(item, mode):
    mixture=torch.as_tensor(item['mixture'])[None]
    references=torch.as_tensor(item['references'])[None]
    active=torch.as_tensor(item['active'])[None]
    m=waveform_metrics(prediction(mixture,mode),references,active,mixture)
    act=active[0];si=m['si_sdr'][0][act];input_si=m['input_si_sdr'][0][act]
    return dict(mode=mode,index=int(item['index']),count=int(item['construction_count']),
        nmse=finite_values(m['nmse'][0][act]),si_sdr=finite_values(si),si_status=value_status(si),
        input_si_sdr=finite_values(input_si),si_sdr_gain=finite_values(si-input_si),
        gain_status=value_status(si-input_si),reference_power=m['reference_power'][0][act].tolist(),
        assignment=m['assignment'][0].tolist(),weakest_index=int(m['reference_power'][0][act].argmin()),
        sum_relative_error=float(m['sum_relative_error'][0]),
        background_nmse=float(m['background_nmse'][0]),inactive_leak=float(m['inactive_leak'][0].sum()))


def run(study, root, public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists(): raise ValueError('Duplicate baseline run')
    parent=watch.read(study/'PROTOCOL.json')
    source=dict(parent['source_sha256'])
    source[str(Path(__file__).relative_to(watch.ROOT))]=watch.digest(Path(__file__))
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest: raise ValueError('Source changed')
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(watch.ROOT/rel,dest)
    template=study/'unet_mean/VALIDATION_004.json';watch.validate(template,None)
    protocol=dict(status='REGISTERED_CPU_MIXTURE_ONLY_BASELINES',modes=list(MODES),
        source_sha256=source,preparation=parent['preparation'],
        preparation_sha256=parent['preparation_sha256'],
        parent_protocol_sha256=watch.digest(study/'PROTOCOL.json'),
        validation_template_sha256=watch.digest(template),cases=630,model_updates=0,
        learned_parameters=0,heldout_read=False,prediction_inputs='mixture only',
        source_assignment='same whole-window NMSE+coherence PIT as learned models',
        zero_output_policy='preserve nonfinite SI-SDR and full-group null means',
        purpose='post-hoc sanity baseline, not an independent test or a new separator',
        numerical_gain_tolerance_db=.001)
    watch.write(root/'PROTOCOL.json',protocol)
    expected={r['index']:r for r in watch.read(template)['rows']}
    data=NativeMixtures(parent['preparation'],'validation_pack',1)
    records=[];start=time.time()
    with torch.inference_mode():
        for index in range(len(data)):
            item=data[index]
            if item['construction_count']!=expected[index]['count']:raise ValueError('Changed count')
            for mode in MODES:
                r=score(item,mode)
                np.testing.assert_allclose(r['reference_power'],expected[index]['reference_power'],rtol=1e-12,atol=1e-14)
                if r['sum_relative_error']>1e-12:raise ValueError('Mixture-sum failure')
                if mode=='all_background':
                    np.testing.assert_allclose(r['nmse'],1,rtol=0,atol=1e-12)
                    if any(v is not None for v in r['si_sdr']):raise ValueError('Zero output got finite SI')
                if mode in ('equal_three','equal_four') and r['count']>1:
                    if any(v is None or abs(v)>.001 for v in r['si_sdr_gain']):
                        raise ValueError('Scalar-copy SI invariance failed')
                records.append(r)
            if index%25==0:
                watch.write(root/'STATE.json',dict(status='CPU_BASELINE_VALIDATION',examples=index+1,
                    total=len(data),pid=os.getpid(),seconds=time.time()-start,time=time.time()))
    summaries=[]
    for mode in MODES:
        for count in (1,2,3):
            rows=[r for r in records if r['mode']==mode and r['count']==count]
            if len(rows)!=210:raise ValueError('Missing cases')
            si=[v for r in rows for v in r['si_sdr']]
            gain=[v for r in rows for v in r['si_sdr_gain']]
            summaries.append(dict(mode=mode,count=count,cases=210,
                mean_nmse=stats.mean(v for r in rows for v in r['nmse']),
                mean_si_sdr=stats.mean(si) if all(v is not None for v in si) else None,
                nonfinite_si_sdr=sum(v is None for v in si),
                si_status_counts=dict(collections.Counter(v for r in rows for v in r['si_status'])),
                mean_si_gain=stats.mean(gain) if all(v is not None for v in gain) else None,
                weakest_nmse=stats.mean(r['nmse'][r['weakest_index']] for r in rows)))
    for rel,digest in source.items():
        if watch.digest(watch.ROOT/rel)!=digest or watch.digest(root/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen baseline source changed')
    if watch.digest(Path(parent['preparation'])/'PREPARATION.json')!=parent['preparation_sha256']:
        raise ValueError('Preparation changed')
    watch.write(root/'ROWS.json',records)
    result=dict(status='COMPLETE',summary=summaries,cases_per_mode=630,
        protocol_sha256=watch.digest(root/'PROTOCOL.json'),source_sha256=source,
        preparation_sha256=parent['preparation_sha256'],validation_template_sha256=watch.digest(template),
        rows_sha256=watch.digest(root/'ROWS.json'),seconds=time.time()-start,heldout_read=False,
        independent_test=False,model_updates=0,learned_parameters=0,
        checks=dict(reference_powers_match=True,all_background_nmse_one=True,
                    zero_output_nonfinite_preserved=True,scalar_copy_si_invariance=True,mixture_sum=True))
    watch.write(root/'COMPLETE.json',result);watch.write(public.with_suffix('.json'),result)
    lines=['# 분리하지 않는 단순 기준선: 기존 개발 혼합 평가','',
        '예측에는 혼합 I/Q만 사용했다. 정답·합성 개수는 채점에만 사용하며 학습·배율 보정은 없다. '
        '각방법630혼합을 같은 전체창 PIT 규칙으로 평가했다. 모든 출력의 합은 혼합과 같다.','',
        '- `copy_one`: 첫 성분에 혼합 전체, 다른 두 성분과 배경은0.','- `equal_three`: 세 성분에 혼합/3, 배경0.',
        '- `equal_four`: 세 성분과 배경에 모두 혼합/4.','- `all_background`: 세 성분0, 배경에 혼합 전체.','',
        '| 방법 | 구성 성분 수 | NMSE ↓ | 약한 성분 NMSE ↓ | 복소 SI-SDR ↑ dB | 비유한 SI 성분 수 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        si='비유한 포함' if r['mean_si_sdr'] is None else f"{r['mean_si_sdr']:.6f}"
        lines.append(f"|{r['mode']}|{r['count']}|{r['mean_nmse']:.6f}|{r['weakest_nmse']:.6f}|{si}|{r['nonfinite_si_sdr']}|")
    lines+=['','0출력의 SI-SDR은 미정의다. 비유한 성분을 제외한 평균을 만들지 않았다. '
        '균등분배는 파형 모양을 분리하지 않으므로 SI-SDR은 혼합 입력과 같으며 수치오차0.001dB 이내를 검사했다. '
        '낮은 NMSE 하나만으로 모든 신호를 복원했다고 판단하지 않는다. 기존 모델 선택 규칙은 유지했다. '
        '단일 성분의 완전 복사는 무한 SI-SDR도 발생할 수 있으며 상태별 개수는 JSON에 남겼다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','run','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.study.resolve(),a.run.resolve(),a.public.resolve())
    except Exception:
        a.run.mkdir(parents=True,exist_ok=True)
        watch.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
