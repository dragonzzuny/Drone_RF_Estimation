"""Report saved native RF study evidence; never infer running from a plan."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from native_data import sha256,write_json


def summarize(path):
    value=json.loads(path.read_text());rows=value['rows'];groups=[]
    if len(rows)!=630 or sorted(r['index'] for r in rows)!=list(range(630)):
        raise ValueError('Missing/duplicate validation rows')
    for count in (1,2,3):
        items=[r for r in rows if r['count']==count]
        if len(items)!=210:raise ValueError('Wrong validation balance')
        def average(key):
            numbers=[v for r in items for v in r[key]]
            return float(np.mean(numbers)) if all(v is not None for v in numbers) else None
        group=dict(count=count,cases=210,mean_nmse=average('nmse'),mean_si_sdr=average('si_sdr'),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in items])),
            construction_count_accuracy=float(np.mean([r['predicted_count']==count for r in items])))
        saved=next(g for g in value['by_count'] if g['count']==count)
        for key in group:
            if key in saved and group[key] is not None and not np.isclose(group[key],saved[key],rtol=1e-9,atol=1e-10):
                raise ValueError('Saved aggregate mismatch')
        groups.append(group)
    return dict(path=str(path),sha256=sha256(path),by_count=groups,
                selection_nmse=float(np.mean([g['mean_nmse'] for g in groups if g['count']>1])))


def report(root,output):
    result=dict(report_time=time.time(),root=str(root),heldout_access=False,independent_test=False)
    for name in ('PREP_PROGRESS','PREP_COMPLETE','GPU_STATE','GPU_PROGRESS','GPU_PREFLIGHT','GPU_COMPLETE',
                 'SPECTRAL_PROGRESS','SPECTRAL_COMPLETE','FRAME_PROGRESS','FRAME_COMPLETE',
                 'PREP_FAILURE','GPU_FAILURE','SPECTRAL_FAILURE','FRAME_FAILURE'):
        path=root/f'{name}.json'
        if path.exists():result[name]=json.loads(path.read_text())
    diagnostic=json.loads((root/'diagnostic/RESULT.json').read_text())
    result['train_diagnostic']=dict(status=diagnostic['status'],cases=diagnostic['cases'],
        unique_contexts=diagnostic['unique_contexts'],sha256=sha256(root/'diagnostic/RESULT.json'))
    result['epochs']=[]
    for path in sorted((root/'gpu').glob('VALIDATION_*.json')):
        epoch=int(path.stem.split('_')[-1])
        if epoch and not (root/'gpu'/f'EPOCH_{epoch:03d}.json').exists():continue
        result['epochs'].append(dict(epoch=epoch,**summarize(path)))
    spectral=root/'spectral/VALIDATION.json'
    if spectral.exists():result['spectral']=summarize(spectral)
    frame=root/'spectral/FRAME_VALIDATION.json'
    if frame.exists():result['frame_spectral']=summarize(frame)
    def better(a,b):
        return all(x['mean_nmse']<y['mean_nmse'] and x['mean_si_sdr'] is not None
            and y['mean_si_sdr'] is not None and x['mean_si_sdr']>y['mean_si_sdr']
            for x,y in zip(a['by_count'][1:],b['by_count'][1:]))
    if 'frame_spectral' in result and 'spectral' in result:
        result['frame_joint_improvement_over_constant']=better(result['frame_spectral'],result['spectral'])
    if result['epochs']:
        selected=min(result['epochs'],key=lambda x:x['selection_nmse'])
        result['selected_epoch']=selected['epoch']
        initial=result['epochs'][0]
        result['joint_improvement_over_native_parent']=better(selected,initial)
        if 'spectral' in result:result['joint_improvement_over_spectral']=better(selected,result['spectral'])
        if 'frame_spectral' in result:result['joint_improvement_over_frame_spectral']=better(selected,result['frame_spectral'])
    manifest=root/'preparation/NATIVE_MANIFEST.json'
    if manifest.exists():
        clips=json.loads(manifest.read_text())['clips'];by_group=defaultdict(list)
        for c in clips:by_group[(c['role'],c['category'])].append(c['retained_power_fraction'])
        result['retained_power']=[dict(role=k[0],category=k[1],contexts=len(v),minimum=min(v),
            median=float(np.median(v)),maximum=max(v)) for k,v in sorted(by_group.items())]
    lines=['# 원 수신 주파수 배치 보존: 진행 및 복원 결과','',
        '수신 중심주파수 차이를 보존하고 공통 RF 관측 대역으로 제한한 별도 실험이다. 기존 중심 정렬 수치와 동일 조건의 모델 개선으로 비교하지 않는다.',
        '목표 정답은 공통 대역 안의 원기록 기여 파형이며 수신 잡음도 포함한다. 보류 기종은 열지 않았고 실제 드론 대수·독립 시험 성능으로 해석하지 않는다.','',
        f"보고 생성 UTC epoch: {result['report_time']:.3f}. 단계 표시는 저장된 상태이며 실행 여부는 PID를 별도 확인해야 한다.",
        f"학습 자료 사전 검사: {diagnostic['cases']}혼합 / {diagnostic['unique_contexts']}구간, {diagnostic['status']}.",'',
        '| 항목 | 저장 상태 |','|---|---|']
    for name in ('PREP_PROGRESS','PREP_COMPLETE','GPU_STATE','GPU_COMPLETE','SPECTRAL_PROGRESS','SPECTRAL_COMPLETE','FRAME_PROGRESS','FRAME_COMPLETE'):
        if name in result:
            v=result[name];lines.append(f"| {name} | {v.get('status',v.get('stage',''))} |")
    lines+=['','| 조건 | epoch | 1성분 NMSE | 2성분 NMSE | 3성분 NMSE | 2성분 복소 SI-SDR dB | 3성분 복소 SI-SDR dB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def row(name,e,groups):
        values=[groups[0]['mean_nmse'],groups[1]['mean_nmse'],groups[2]['mean_nmse'],groups[1]['mean_si_sdr'],groups[2]['mean_si_sdr']]
        return f'| {name} | {e} | '+' | '.join('미정의' if v is None else f'{v:.6f}' for v in values)+' |'
    for e in result['epochs']:lines.append(row('동일 U-Net',e['epoch'],e['by_count']))
    if 'spectral' in result:lines.append(row('TRAIN 스펙트럼 기준선','—',result['spectral']['by_count']))
    if 'frame_spectral' in result:lines.append(row('같은 스펙트럼·프레임별 분배','—',result['frame_spectral']['by_count']))
    if 'frame_joint_improvement_over_constant' in result:
        lines+=['',f"같은 스펙트럼에서 프레임별 분배의 두 지표 동시 개선: {result['frame_joint_improvement_over_constant']}."]
    if result['epochs']:
        lines+=['',f"평균 NMSE 사전 규칙 선택 epoch: {result['selected_epoch']}.",
            f"동일 관측의 초기 모델 대비 2/3성분 두 지표 동시 개선: {result['joint_improvement_over_native_parent']}."]
        if 'joint_improvement_over_spectral' in result:lines.append(f"스펙트럼 기준선 대비 두 지표 동시 개선: {result['joint_improvement_over_spectral']}.")
        if 'joint_improvement_over_frame_spectral' in result:lines.append(f"프레임별 스펙트럼 기준선 대비 두 지표 동시 개선: {result['joint_improvement_over_frame_spectral']}.")
        chosen=next(e for e in result['epochs'] if e['epoch']==result['selected_epoch'])
        lines+=['','| 선택 모델 조건 | 약신호 NMSE | 구성 성분 수 정확도 |','|---|---:|---:|']
        for g in chosen['by_count']:lines.append(f"| {g['count']}성분 | {g['weakest_nmse']:.6f} | {g['construction_count_accuracy']:.3%} |")
    lines+=['','필터·주파수 이동 수치 검사 통과는 분리 성능 개선을 뜻하지 않는다. 한 seed, 반복 개발 검증, 기록 묶음과 VTSBW 변화가 얽힌 분할이다.',
        '방법·범위: [실행 규약](../../experiments/rfuav_native_frequency_20261009/README.md). 모든 완료 epoch와 조건을 보존하며 유리한 사례로 대체하지 않는다.','']
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output.with_suffix('.json'),result)
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(epochs=[x['epoch'] for x in result['epochs']],selected=result.get('selected_epoch'),
        spectral='spectral' in result,complete='GPU_COMPLETE' in result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report(args.run,args.output)
