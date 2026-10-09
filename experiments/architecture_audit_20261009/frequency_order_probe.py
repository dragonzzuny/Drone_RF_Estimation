"""Frozen U-Net inference with reordered frequency cells: CPU TRAIN diagnosis.

An exploratory boundary/representation check, not retraining or a heldout test.
The inverse ordering is restored before iSTFT. No source labels enter forward.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch

import fit_diagnostic as fit
from native_data import NativeMixtures,sha256,write_json
from models import build
from drone_rf.waveform import analyze,synthesize,waveform_metrics

ROOT=Path(__file__).resolve().parents[2]


def evaluate(net,item,shift):
    batch={k:torch.as_tensor(np.asarray(item[k])[None]) for k in
           ('mixture','references','active','context_features','crop_start')}
    with torch.inference_mode():
        z=analyze(batch['mixture'])
        if shift:z=torch.fft.fftshift(z,dim=-2)
        predicted=net(z,batch['context_features'],batch['crop_start'])
        spec=predicted['estimates']
        if shift:spec=torch.fft.ifftshift(spec,dim=-2)
        estimate=synthesize(spec,batch['mixture'].shape[-1])
        score=waveform_metrics(estimate,batch['references'],batch['active'],batch['mixture'])
    count=int(item['construction_count'])
    if float(score['sum_relative_error'].max())>1e-9:raise ValueError('Mixture sum failed')
    nmse=score['nmse'][0,:count].tolist();si=score['si_sdr'][0,:count].tolist()
    if not np.isfinite(nmse).all() or not np.isfinite(si).all():raise ValueError('Nonfinite score')
    return dict(nmse=nmse,si_sdr=si,reference_power=score['reference_power'][0,:count].tolist(),
        assignment=score['assignment'][0,:count].tolist(),count_logits=predicted['count_logits'][0].tolist(),
        sum_relative_error=float(score['sum_relative_error'].max()))


def run(screen,output,public):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'PROTOCOL.json').exists():raise ValueError('Duplicate registration')
    plan=fit.read(screen/'PROTOCOL.json')
    train=NativeMixtures(plan['preparation'],'train_pack',1)
    used,indices=Counter(),[]
    for i,row in enumerate(train.rows):
        n=int(row['count']);names=tuple(train.library.clips[int(j)]['category'] for j in row['indices'][:n])
        key=names+tuple(map(float,row['levels'][:n]))
        if used[key]<2:indices.append(i);used[key]+=1
    if len(indices)!=48:raise ValueError('Expected48 fixed TRAIN cases')
    source=dict(plan['source_sha256']);source[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    checkpoint=screen/'unet_mean/SELECTED_003.pt';digest=sha256(checkpoint)
    copy=output/'model.pt';shutil.copyfile(checkpoint,copy)
    if sha256(copy)!=digest or sha256(checkpoint)!=digest:raise ValueError('Checkpoint copy changed')
    val_path=screen/'unet_mean/VALIDATION_003.json'
    saved_rows=fit.read(val_path)['rows']
    reproduction=[next(r['index'] for r in saved_rows if r['count']==n) for n in (1,2,3)]
    protocol=dict(status='REGISTERED_CPU_TRAIN_FREQUENCY_ORDER_DIAGNOSIS',source_sha256=source,
        screen_protocol_sha256=sha256(screen/'PROTOCOL.json'),preparation_sha256=plan['preparation_sha256'],
        checkpoint_sha256=digest,checkpoint_epoch=3,checkpoint_updates=225,train_indices=indices,
        backend_reproduction_indices=reproduction,backend_reference_sha256=sha256(val_path),
        modes=['original','frequency_centered'],change='fftshift of frequency cells before U-Net, inverse shift of all outputs before iSTFT; same context and weights',
        one_forward_per_mode=True,ensemble=False,model_updates=0,train_only_comparison=True,
        validation_use='Three already evaluated cases only for CPU/GPU baseline agreement',heldout_read=False,
        interpretation='Exploratory inference distribution/boundary change; not full validation benefit or new trained model')
    for rel,value in source.items():
        if sha256(ROOT/rel)!=value:raise ValueError('Source changed')
        target=output/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    write_json(output/'PROTOCOL.json',protocol)
    weights=torch.load(copy,map_location='cpu',weights_only=False)
    if weights['best']['epoch']!=3 or weights['protocol_sha256']!=protocol['screen_protocol_sha256']:raise ValueError('Wrong weights')
    net=build('unet_mean');net.load_state_dict(weights['model']);net.eval();del weights
    val=NativeMixtures(plan['preparation'],'validation_pack',1)
    by_index={r['index']:r for r in saved_rows};checks=[]
    for index in reproduction:
        actual=evaluate(net,val[index],False);expected=by_index[index]
        np.testing.assert_allclose(actual['nmse'],expected['nmse'],rtol=1e-3,atol=1e-5)
        np.testing.assert_allclose(actual['si_sdr'],expected['si_sdr'],rtol=0,atol=.01)
        if actual['assignment']!=expected['assignment'][:expected['count']]:raise ValueError('Active matching changed on CPU')
        checks.append(dict(index=index,max_nmse_difference=float(np.max(np.abs(np.array(actual['nmse'])-expected['nmse'])))))
    rows=[];start=time.time()
    for index in indices:
        item=train[index];n=int(item['construction_count'])
        modes={name:evaluate(net,item,shift) for name,shift in (('original',False),('frequency_centered',True))}
        if modes['original']['count_logits']!=modes['frequency_centered']['count_logits']:raise ValueError('Count context changed')
        clips=[train.library.clips[int(i)] for i in train.rows[index]['indices'][:n]]
        rows.append(dict(index=index,count=n,categories=[c['category'] for c in clips],
            weakest_index=int(np.argmin(modes['original']['reference_power'])),modes=modes))
        write_json(output/'STATE.json',dict(status='CPU_TRAIN_INFERENCE',cases=len(rows),total=48,pid=os.getpid(),seconds=time.time()-start,time=time.time()))
    summary=[]
    for n in (1,2,3):
        sub=[r for r in rows if r['count']==n]
        for mode in protocol['modes']:
            summary.append(dict(count=n,cases=len(sub),mode=mode,
                mean_nmse=float(np.mean([v for r in sub for v in r['modes'][mode]['nmse']])),
                mean_si_sdr=float(np.mean([v for r in sub for v in r['modes'][mode]['si_sdr']])),
                weakest_nmse=float(np.mean([r['modes'][mode]['nmse'][r['weakest_index']] for r in sub]))))
    for rel,value in source.items():
        if sha256(ROOT/rel)!=value or sha256(output/'source_snapshot'/rel)!=value:raise ValueError('Frozen source changed')
    result=dict(protocol,status='COMPLETE',protocol_sha256=sha256(output/'PROTOCOL.json'),summary=summary,
        rows=rows,backend_checks=checks,seconds=time.time()-start)
    write_json(output/'COMPLETE.json',result);write_json(public.with_suffix('.json'),result)
    text=['# STFT 주파수 배열 순서: TRAIN 추론 진단','',
        '동일 U-Net e3 가중치와 고정 TRAIN48혼합을 사용했다. 입력 주파수 배열을 중앙 정렬하고 출력에서 '
        '원래 순서로 돌린 뒤 iSTFT했다. 신호 값·수신 주파수 간격·정답·문맥 특징은 그대로다. '
        '정답은 평가에만 사용하고 모델 입력에는 전달하지 않는다. 재학습이나 검증 전체 성능 비교가 아니다.','',
        '|성분 수|혼합 수|배열|NMSE↓|복소 SI-SDR↑ dB|약신호 NMSE↓|',
        '|---:|---:|---|---:|---:|---:|']
    for s in summary:text.append(f"|{s['count']}|{s['cases']}|{s['mode']}|{s['mean_nmse']:.6f}|{s['mean_si_sdr']:.3f}|{s['weakest_nmse']:.6f}|")
    text+=['','기존 검증 중1/2/3성분 각1개에서 CPU 원래 추론과 저장된 GPU 점수의 일치부터 확인했다. '
        '중앙 정렬한 모델을 새로 학습한 실험이 아니므로, 추론 배열 변경의 부정적 결과를 정렬 표현 자체의 '
        '일반적인 실패로 해석하지 않는다. 보류 파일은 열지 않았다.','']
    public.with_suffix('.md').write_text('\n'.join(text));write_json(output/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(summary=summary,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('screen','output','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.screen.resolve(),a.output.resolve(),a.public.resolve())
    except Exception:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
