"""Completion audit for the native count-gradient ablation and phase inference."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import sha256, write_json
from report import summarize


def read(path):
    return json.loads(path.read_text())


def identities(first, second, same_scores=False, same_count=False):
    if len(first)!=630 or len(second)!=630:
        raise ValueError('Expected all 630 validation rows')
    for a,b in zip(first,second):
        keys=['index','count','categories','pack_ids','nominal_levels_db']
        if same_count: keys+=['predicted_count']
        if any(a[k]!=b[k] for k in keys):
            raise ValueError('Mixture identities changed')
        np.testing.assert_allclose(a['reference_power'],b['reference_power'],rtol=1e-10,atol=1e-15)
        if b['sum_relative_error']>1e-9:
            raise ValueError('Mixture sum failed')
        if same_scores:
            for key in ('nmse','si_sdr'):
                np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-7)


def run(candidate, control, output):
    if not (candidate/'GPU_COMPLETE.json').exists() or not (candidate/'phase/COMPLETE.json').exists():
        raise ValueError('Training and phase inference must both complete')
    if (candidate/'GPU_FAILURE.json').exists() or (candidate/'phase/FAILURE.json').exists():
        raise ValueError('Failure receipt present')
    frozen=read(candidate/'GPU_PROTOCOL.json')
    if read(candidate/'GPU_COMPLETE.json')['protocol_sha256']!=sha256(candidate/'GPU_PROTOCOL.json'):
        raise ValueError('Training completion protocol mismatch')
    for rel,digest in frozen['source_sha256'].items():
        for p in (ROOT/rel,candidate/'source_snapshot'/rel):
            if sha256(p)!=digest: raise ValueError('Frozen training source changed')
    for path,digest in read(candidate/'PREP_PROTOCOL.json')['data_sha256'].items():
        if sha256(path)!=digest: raise ValueError('Original data metadata changed')
    if sha256(frozen['parent'])!=frozen['parent_sha256']:raise ValueError('Parent changed')
    if sha256(control/'GPU_PROTOCOL.json')!=frozen['control_protocol_sha256']:raise ValueError('Control changed')
    if candidate.joinpath('preparation').resolve()!=control.joinpath('preparation').resolve():raise ValueError('Not the same prepared data')
    initial=read(control/'gpu/VALIDATION_000.json')['rows']
    history=[]
    for epoch in range(6):
        path=candidate/'gpu'/f'VALIDATION_{epoch:03d}.json'
        identities(initial,read(path)['rows'],same_scores=epoch==0)
        history.append(dict(epoch=epoch,**summarize(path)))
        if epoch:
            selected=min(history,key=lambda x:x['selection_nmse'])
            receipt=read(candidate/'gpu'/f'EPOCH_{epoch:03d}.json')
            if receipt['updates']!=75*epoch or receipt['best']['epoch']!=selected['epoch']:
                raise ValueError('Update or selection mismatch')
            if receipt['protocol_sha256']!=sha256(candidate/'GPU_PROTOCOL.json'):
                raise ValueError('Epoch protocol mismatch')
    selected=min(history,key=lambda x:x['selection_nmse'])
    checkpoints=[]
    torch.set_num_threads(2)
    for name in ('SELECTED_005.pt','LAST.pt'):
        saved=torch.load(candidate/'gpu'/name,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=sha256(candidate/'GPU_PROTOCOL.json') or saved['best']['epoch']!=selected['epoch']:
            raise ValueError('Checkpoint identity mismatch')
        if name=='LAST.pt' and (saved['epoch']!=5 or saved['updates']!=375):raise ValueError('Training budget incomplete')
        if sum(t.numel() for t in saved['model'].values())!=32142859:raise ValueError('Capacity changed')
        if any(not torch.isfinite(t).all() for t in saved['model'].values()):raise ValueError('Nonfinite weights')
        checkpoints.append(dict(name=name,sha256=sha256(candidate/'gpu'/name),best=saved['best']))
        del saved;gc.collect()
    if sha256(candidate/'gpu/BEST.pt')!=checkpoints[0]['sha256']:raise ValueError('Selected copy mismatch')
    phase_root=candidate/'phase'
    phase_plan=read(phase_root/'PROTOCOL.json')
    for path,digest in phase_plan['data_sha256'].items():
        if sha256(path)!=digest: raise ValueError('Phase metadata changed')
    for rel,digest in phase_plan['source_sha256'].items():
        if sha256(ROOT/rel)!=digest or sha256(phase_root/'source_snapshot'/rel)!=digest:
            raise ValueError('Phase source mismatch')
    if read(phase_root/'COMPLETE.json')['protocol_sha256']!=sha256(phase_root/'PROTOCOL.json'):
        raise ValueError('Phase protocol mismatch')
    if read(phase_root/'CHECKPOINT.json')['sha256']!=checkpoints[0]['sha256']:raise ValueError('Phase selected another model')
    original=read(candidate/'gpu'/f"VALIDATION_{selected['epoch']:03d}.json")['rows']
    phase={}
    for mode in ('BASELINE','FOUR_PHASE'):
        path=phase_root/'native_selected'/f'{mode}.json'
        identities(original,read(path)['rows'],same_scores=mode=='BASELINE',same_count=True)
        phase[mode]=summarize(path)
    control_epoch=read(control/'gpu/EPOCH_005.json')['best']['epoch']
    reference=summarize(control/'gpu'/f'VALIDATION_{control_epoch:03d}.json')
    reference_phase=summarize(control/'phase/native_selected/FOUR_PHASE.json')
    def decision(a,b):
        return dict(both_counts_nmse_and_si_improve=all(y['mean_nmse']<x['mean_nmse'] and y['mean_si_sdr']>x['mean_si_sdr'] for x,y in zip(a['by_count'][1:],b['by_count'][1:])),
                    both_counts_weak_improve=all(y['weakest_nmse']<x['weakest_nmse'] for x,y in zip(a['by_count'][1:],b['by_count'][1:])))
    result=dict(status='PASS',source_sha256=sha256(Path(__file__)),epochs=5,updates_per_arm=375,
        history=history,selected_epoch=selected['epoch'],checkpoints=checkpoints,
        control_single=reference,candidate_single=selected,control_four_phase=reference_phase,candidate_four_phase=phase['FOUR_PHASE'],
        single_decision=decision(reference,selected),phase_decision=decision(reference_phase,phase['FOUR_PHASE']),
        heldout_read=False,independent_test=False,seed=0,physical_aircraft_count=False,
        source_checks=len(frozen['source_sha256']),phase_source_checks=len(phase_plan['source_sha256']))
    write_json(output.with_suffix('.json'),result)
    lines=['# 개수 보조손실의 공유 인코더 기울기 차단: 완료 결과','',
        'RFUAV 단일 데이터셋·같은 RF 대역·원 수신 중심 차이 유지. 동일 부모·자료·seed0·각5epoch/375업데이트. '
        '개수 head는 계속 학습하며 입력 detach만 변경했다. 파라미터32142859개와 추론 순전파는 그대로다. '
        '이전에 완료한 대조군을 재사용했다. 검증630혼합을 반복 사용한 개발 비교이며 독립 시험이 아니다.','',
        '| 방법 | 2성분 NMSE ↓ | 3성분 NMSE ↓ | 2성분 SI-SDR ↑ | 3성분 SI-SDR ↑ | 약성분 NMSE 2/3 ↓ |',
        '|---|---:|---:|---:|---:|---|']
    for label,key in [('기존·1회','control_single'),('기울기 차단·1회','candidate_single'),('기존·네 위상','control_four_phase'),('기울기 차단·네 위상','candidate_four_phase')]:
        a,b=result[key]['by_count'][1:]
        lines.append(f"| {label} | {a['mean_nmse']:.6f} | {b['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f} | {b['mean_si_sdr']:.3f} | {a['weakest_nmse']:.6f} / {b['weakest_nmse']:.6f} |")
    lines+=['',f"기존 선택epoch={control_epoch}, 차단군 선택epoch={selected['epoch']}. 선택은 사전 등록한2·3성분 평균 NMSE 최솟값이며 네 위상 평가로 다시 선택하지 않았다.",
        f"1회 추론의 두 조건·두 지표 동시 개선: {result['single_decision']['both_counts_nmse_and_si_improve']}. 동일 네 위상 추론끼리의 동시 개선: {result['phase_decision']['both_counts_nmse_and_si_improve']}.",
        '공유 기울기의 충돌 단서와 이 변경의 전체 성능 우월성은 다른 주장이다. 일부 조건 개선과 악화를 함께 보고한다. '
        '네 위상 평균은 정답 없는 출력 순열 정렬을 사용하며, true-count나 정답 배율을 추론에 넣지 않는다.','',
        '초기 상태와5개epoch의 모든 평가행, 선택 규칙, 원본 메타데이터·소스 해시, 유한한 전체 가중치·375업데이트, '
        '1회 추론 재현과 위상 평가630행을 검산했다. 보류 확인용 기종을 개봉하지 않았다.','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS',selected_epoch=selected['epoch'],single=result['single_decision'],phase=result['phase_decision'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.candidate,args.control,args.output)
