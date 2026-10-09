"""Paired CPU validation of one fixed checkpoint's frequency-array ordering.

Both modes run on all630development cases on the same CPU backend. The original
mode must also reproduce saved GPU scores. GPU training proceeds independently.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import frequency_order_probe as probe
from native_data import NativeMixtures,sha256,write_json
from models import build
import watch_epochs as watch


def run(screen,prerequisite,output,public):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'PROTOCOL.json').exists():raise ValueError('Duplicate registration')
    prior=watch.read(prerequisite/'COMPLETE.json')
    if prior['status']!='COMPLETE' or len(prior['rows'])!=48:raise ValueError('TRAIN diagnosis incomplete')
    plan=watch.read(screen/'PROTOCOL.json')
    val_file=screen/'unet_mean/VALIDATION_003.json'
    watch.validate(val_file,None)
    expected=watch.read(val_file)['rows']
    if sha256(val_file)!=prior['backend_reference_sha256']:raise ValueError('Wrong validation weights')
    source=dict(prior['source_sha256'])
    for p in (Path(__file__),Path(watch.__file__)):source[str(p.relative_to(probe.ROOT))]=sha256(p)
    checkpoint=prerequisite/'model.pt'
    if sha256(checkpoint)!=prior['checkpoint_sha256']:raise ValueError('Copied checkpoint changed')
    protocol=dict(status='REGISTERED_FULL_PAIRED_CPU_FREQUENCY_ORDER_VALIDATION',source_sha256=source,
        prerequisite_sha256=sha256(prerequisite/'COMPLETE.json'),validation_sha256=sha256(val_file),
        preparation_sha256=plan['preparation_sha256'],checkpoint_sha256=prior['checkpoint_sha256'],
        checkpoint_epoch=3,updates=225,indices=[r['index'] for r in expected],cases=630,
        modes=['original','frequency_centered'],backend='CPU float32, two threads, same backend both modes',
        original_gpu_reproduction='all630cases; NMSE rtol1e-3 atol1e-5, SI-SDR atol0.01dB; active assignment differences recorded',
        heldout_read=False,independent_test=False,model_updates=0,one_forward_per_mode=True,
        scope='Fixed e3 inference sensitivity; does not alter primary checkpoint selection or claim training improvement',
        inference_gate='Report both2/3 NMSE and SI-SDR plus weak/single/count conditions; do not adopt from TRAIN score')
    for rel,digest in source.items():
        if sha256(probe.ROOT/rel)!=digest:raise ValueError('Frozen source changed')
        dest=output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(probe.ROOT/rel,dest)
    write_json(output/'PROTOCOL.json',protocol)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if weights['best']['epoch']!=3 or weights['protocol_sha256']!=sha256(screen/'PROTOCOL.json'):raise ValueError('Wrong checkpoint')
    net=build('unet_mean');net.load_state_dict(weights['model']);net.eval();del weights
    data=NativeMixtures(plan['preparation'],'validation_pack',1)
    rows=[];start=time.time()
    for reference in expected:
        index=reference['index'];item=data[index];count=int(item['construction_count'])
        modes={name:probe.evaluate(net,item,shift) for name,shift in (('original',False),('frequency_centered',True))}
        base=modes['original']
        if count!=reference['count']:raise ValueError('Count changed')
        np.testing.assert_allclose(base['reference_power'],reference['reference_power'],rtol=1e-10,atol=1e-15)
        np.testing.assert_allclose(base['nmse'],reference['nmse'],rtol=1e-3,atol=1e-5)
        np.testing.assert_allclose(base['si_sdr'],reference['si_sdr'],rtol=0,atol=.01)
        if base['count_logits']!=modes['frequency_centered']['count_logits']:raise ValueError('Count context changed')
        rows.append(dict(index=index,count=count,categories=reference['categories'],pack_ids=reference['pack_ids'],
            weakest_index=reference['weakest_index'],modes=modes,
            active_assignment_cpu_gpu_equal=base['assignment']==reference['assignment'][:count],
            baseline_max_nmse_difference=float(np.max(np.abs(np.array(base['nmse'])-reference['nmse']))),
            baseline_max_si_difference_db=float(np.max(np.abs(np.array(base['si_sdr'])-reference['si_sdr'])))))
        if len(rows)%10==0 or len(rows)==630:
            write_json(output/'PARTIAL.json',dict(rows=rows,partial=True))
            write_json(output/'STATE.json',dict(status='PAIRED_CPU_VALIDATION',cases=len(rows),total=630,
                seconds=time.time()-start,pid=os.getpid(),time=time.time()))
    summary=[]
    for count in (1,2,3):
        sub=[r for r in rows if r['count']==count]
        if len(sub)!=210:raise ValueError('Unbalanced validation')
        for mode in protocol['modes']:
            summary.append(dict(count=count,cases=len(sub),mode=mode,
                mean_nmse=float(np.mean([v for r in sub for v in r['modes'][mode]['nmse']])),
                mean_si_sdr=float(np.mean([v for r in sub for v in r['modes'][mode]['si_sdr']])),
                weakest_nmse=float(np.mean([r['modes'][mode]['nmse'][r['weakest_index']] for r in sub]))))
    for rel,digest in source.items():
        if sha256(probe.ROOT/rel)!=digest or sha256(output/'source_snapshot'/rel)!=digest:raise ValueError('Source changed')
    passed=all(summary[2*(c-1)+1]['mean_nmse']<summary[2*(c-1)]['mean_nmse'] and
        summary[2*(c-1)+1]['mean_si_sdr']>summary[2*(c-1)]['mean_si_sdr'] for c in (2,3))
    result=dict(protocol,status='COMPLETE',protocol_sha256=sha256(output/'PROTOCOL.json'),rows=rows,summary=summary,
        two_three_four_metric_improvement=passed,seconds=time.time()-start,
        maximum_baseline_nmse_discrepancy=max(r['baseline_max_nmse_difference'] for r in rows),
        maximum_baseline_si_discrepancy_db=max(r['baseline_max_si_difference_db'] for r in rows),
        changed_active_assignment_cases=sum(not r['active_assignment_cpu_gpu_equal'] for r in rows))
    write_json(output/'COMPLETE.json',result);write_json(public.with_suffix('.json'),result)
    text=['# 주파수 배열 순서: 동일 CPU의 전체 개발검증 비교','',
        'U-Net e3/225업데이트의 고정 가중치, 기존630개 개발검증을 두 배열 모두 같은 CPU로 평가했다. '
        '원래 배열의 모든 점수를 저장된 GPU 결과와 허용 오차 안에서 검산했다. 새 학습이나 보류 평가가 아니다.','',
        '|성분 수|배열|NMSE↓|복소 SI-SDR↑ dB|약신호 NMSE↓|','|---:|---|---:|---:|---:|']
    for s in summary:text.append(f"|{s['count']}|{s['mode']}|{s['mean_nmse']:.6f}|{s['mean_si_sdr']:.3f}|{s['weakest_nmse']:.6f}|")
    text+=['',f'2/3성분의네파형지표동시개선: {passed}. 이는고정e3에대한추론 비교다. '
        '현재5epoch학습의선택규칙을변경하지않으며다른체크포인트에서같은효과를보장하지않는다.', '',
        f"CPU/GPU 원래 추론의 최대NMSE 차이 {result['maximum_baseline_nmse_discrepancy']:.3g}, "
        f"최대SI-SDR 차이 {result['maximum_baseline_si_discrepancy_db']:.3g}dB. "
        f"활성 출력 대응 차이는{result['changed_active_assignment_cases']}건이다.", '']
    public.with_suffix('.md').write_text('\n'.join(text))
    write_json(output/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',summary=summary,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('screen','prerequisite','output','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.screen.resolve(),a.prerequisite.resolve(),a.output.resolve(),a.public.resolve())
    except Exception:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
