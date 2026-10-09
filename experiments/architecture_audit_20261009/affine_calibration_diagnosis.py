"""Optimistic reference-assisted scalar/offset calibration diagnosis.

Derive the minimum NMSE under a constant complex gain PLUS a constant complex
offset from saved complex SI-SDR and reference DC fractions. No model is run.
This gives neither a deployable correction nor a bound on other separators.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import watch_epochs as watch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures,sha256,write_json
from drone_rf.waveform import complex_si_sdr


def optimum(si_db,variance_fraction):
    if not math.isfinite(si_db) or not 0<=variance_fraction<=1+1e-12:
        raise ValueError('Need finite SI-SDR and valid variance fraction')
    return variance_fraction/(1+10**(si_db/10))


def self_check():
    rng=np.random.default_rng(0)
    ref=rng.normal(size=10000)+1j*rng.normal(size=10000)+(.7+.3j)
    est=(1.2+.5j)*ref+.2*(rng.normal(size=10000)+1j*rng.normal(size=10000))+(1-2j)
    design=np.stack((est,np.ones_like(est)),axis=-1)
    weights=np.linalg.lstsq(design,ref,rcond=None)[0]
    direct=float(np.mean(abs(design@weights-ref)**2)/np.mean(abs(ref)**2))
    si=float(complex_si_sdr(torch.from_numpy(est)[None],torch.from_numpy(ref)[None])[0])
    derived=optimum(si,float(np.var(ref)/np.mean(abs(ref)**2)))
    np.testing.assert_allclose(derived,direct,rtol=1e-10,atol=1e-12)
    return dict(complex_affine_least_squares_nmse=direct,derived_nmse=derived)


def run(screen,output,public):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'PROTOCOL.json').exists():raise ValueError('Duplicate registration')
    plan=watch.read(screen/'PROTOCOL.json')
    paths=[screen/'unet_mean'/f'VALIDATION_{e:03d}.json' for e in (2,3)]
    _,identities=watch.validate(paths[0],None)
    watch.validate(paths[1],identities)
    rows_by_epoch={e:{r['index']:r for r in watch.read(p)['rows'] if r['count']>1} for e,p in zip((2,3),paths)}
    indices=sorted(rows_by_epoch[2])
    if indices!=sorted(rows_by_epoch[3]) or len(indices)!=420:raise ValueError('Expected420 matched cases')
    source=dict(plan['source_sha256'])
    for path in (Path(__file__),Path(watch.__file__)):source[str(path.relative_to(ROOT))]=sha256(path)
    protocol=dict(status='REGISTERED_REFERENCE_ASSISTED_AFFINE_DIAGNOSIS',source_sha256=source,
        validation_sha256={str(e):sha256(p) for e,p in zip((2,3),paths)},
        preparation_sha256=plan['preparation_sha256'],selected_indices=indices,
        model='unet_mean',epochs=[2,3],requires_reference_information=True,deployable=False,
        bound_on_other_separators=False,may_violate_mixture_consistency=True,
        correction_class='one complex gain and one complex DC offset per source per whole evaluated crop',
        formula='minimum raw NMSE = (reference variance / reference power) / (1 + 10^(complex SI-SDR / 10))',
        new_prediction=False,model_updates=0,validation_reference_iq_read=True,heldout_read=False,
        self_check=self_check())
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest:raise ValueError('Frozen source changed')
        dest=output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest)
    write_json(output/'PROTOCOL.json',protocol)
    data=NativeMixtures(plan['preparation'],'validation_pack',1)
    rows=[];started=time.time()
    for index in indices:
        item=data[index];count=int(item['construction_count'])
        ref=np.asarray(item['references'][:count],dtype=np.complex128)
        power=np.mean(abs(ref)**2,axis=-1)
        fraction=np.mean(abs(ref-ref.mean(-1,keepdims=True))**2,axis=-1)/power
        for epoch in (2,3):
            saved=rows_by_epoch[epoch][index]
            np.testing.assert_allclose(power,saved['reference_power'],rtol=1e-10,atol=1e-15)
            if count!=saved['count']:raise ValueError('Reference count changed')
            best=[optimum(si,float(fr)) for si,fr in zip(saved['si_sdr'],fraction)]
            if any(a>b+1e-9 for a,b in zip(best,saved['nmse'])):raise ValueError('Optimum exceeds uncorrected error')
            rows.append(dict(index=index,count=count,epoch=epoch,categories=saved['categories'],
                weakest_index=saved['weakest_index'],local_power_gap_db=10*math.log10(max(power)/min(power)),
                reference_variance_fraction=fraction.tolist(),actual_nmse=saved['nmse'],oracle_affine_nmse=best))
        if len(rows)%100==0:write_json(output/'STATE.json',dict(status='REFERENCE_DC_ANALYSIS',cases=len(rows)//2,total=420,pid=os.getpid(),time=time.time()))
    summary=[]
    for epoch in (2,3):
        for count in (2,3):
            for label,lo,hi in (('all',0,float('inf')),('[0,10)dB',0,10),('[10,20)dB',10,20),('[20,infinity)dB',20,float('inf'))):
                sub=[r for r in rows if r['epoch']==epoch and r['count']==count and lo<=r['local_power_gap_db']<hi]
                summary.append(dict(epoch=epoch,count=count,group=label,cases=len(sub),
                    actual_weak_nmse=float(np.mean([r['actual_nmse'][r['weakest_index']] for r in sub])),
                    oracle_affine_weak_nmse=float(np.mean([r['oracle_affine_nmse'][r['weakest_index']] for r in sub]))))
    for rel,digest in source.items():
        if sha256(ROOT/rel)!=digest or sha256(output/'source_snapshot'/rel)!=digest:raise ValueError('Diagnostic source changed')
    result=dict(protocol,status='COMPLETE',protocol_sha256=sha256(output/'PROTOCOL.json'),
        summary=summary,rows=rows,seconds=time.time()-started)
    write_json(output/'COMPLETE.json',result);write_json(public.with_suffix('.json'),result)
    text=['# 정답을 이용한 최적 복소 배율·상수 보정 진단','',
        'U-Net e2/e3의 기존 검증420혼합과 동일 정답을 사용했다. 각 출력에 정답을 보고 가장 유리한 '
        '복소 배율과 복소 상수 오프셋을 허용했을 때의 최소NMSE를 계산했다. 저장된 SI-SDR과 정답의 '
        '분산/전력 비로 유도했으며 독립 복소 최소제곱 계산으로 수식을 검산했다. 새 추론·학습은 없다.','',
        '이는 실제 적용할 보정기가 아니다. 정답 정보에 의존하고 혼합 합 일치를 깨뜨릴 수도 있다. '
        '다른 신경망·시간 가변 보정·추가 구조의 가능한 성능을 제한하는 하한도 아니다.','',
        '|epoch|성분 수|실제 창별 전력차|혼합 수|현재 약신호 NMSE↓|정답 이용 최적 배율+상수 NMSE↓|',
        '|---:|---:|---|---:|---:|---:|']
    for s in summary:text.append(f"|{s['epoch']}|{s['count']}|{s['group']}|{s['cases']}|{s['actual_weak_nmse']:.6f}|{s['oracle_affine_weak_nmse']:.6f}|")
    text+=['','한 구간 전체의 상수 배율·위상·DC 보정으로 제거 가능한 오차와 남는 파형 모양 차이를 구분한다. '
        '남는 차이를 특정 누출·변조·잡음 원인으로 확정하지 않는다. 보류 기록은 열지 않았다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(text))
    write_json(output/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(status='COMPLETE',summary=summary,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('screen','output','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    try:run(a.screen.resolve(),a.output.resolve(),a.public.resolve())
    except Exception:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
