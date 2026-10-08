"""Audit every row of native phase inference and fixed-model fit-gap diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from native_data import ROOT, NativeMixtures, sha256, write_json
from report import summarize


def read(path):
    return json.loads(path.read_text())


def aggregate(rows, expected_per_count):
    groups = []
    for count in (1,2,3):
        subset = [r for r in rows if r['count'] == count]
        if len(subset) != expected_per_count:
            raise ValueError('Wrong diagnostic balance')
        g = dict(count=count, cases=len(subset),
            mean_nmse=float(np.mean([v for r in subset for v in r['nmse']])),
            mean_si_sdr=float(np.mean([v for r in subset for v in r['si_sdr']])),
            weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in subset])),
            construction_count_accuracy=float(np.mean([r['predicted_count']==count for r in subset])))
        if not all(np.isfinite(v) for v in g.values()):
            raise ValueError('Undefined diagnostic aggregate')
        groups.append(g)
    return groups


def same_summary(actual, saved):
    for a,b in zip(actual,saved):
        for key in a:
            np.testing.assert_allclose(a[key],b[key],rtol=1e-9,atol=1e-11)


def run(parent, output):
    audits = {}
    for name in ('phase','fit_gap'):
        root=parent/name
        if not (root/'COMPLETE.json').exists() or (root/'FAILURE.json').exists():
            raise ValueError(f'{name} incomplete/failed')
        plan=read(root/'PROTOCOL.json')
        for rel,digest in plan['source_sha256'].items():
            if sha256(ROOT/rel)!=digest or sha256(root/'source_snapshot'/rel)!=digest:
                raise ValueError('Inference source changed')
        for path,digest in plan['data_sha256'].items():
            if sha256(path)!=digest:
                raise ValueError('Inference metadata changed')
        complete=read(root/'COMPLETE.json')
        if complete['protocol_sha256']!=sha256(root/'PROTOCOL.json'):
            raise ValueError('Completion protocol mismatch')
        audits[name]=dict(source_files=len(plan['source_sha256']),data_files=len(plan['data_sha256']),
                         protocol_sha256=sha256(root/'PROTOCOL.json'),result_sha256=sha256(root/'COMPLETE.json'))
    identity=read(parent/'phase/CHECKPOINT.json')
    if sha256(parent/'gpu/SELECTED_005.pt')!=identity['sha256']:
        raise ValueError('Phase checkpoint changed')
    selected=identity['selected_epoch']
    original=read(parent/'gpu'/f'VALIDATION_{selected:03d}.json')
    phase={}
    for name in ('baseline','four_phase'):
        path=parent/'phase/native_selected'/f'{name.upper()}.json'
        value=read(path)
        phase[name]=summarize(path)
        same_summary(aggregate(value['rows'],210),value['by_count'])
        for row,old in zip(value['rows'],original['rows']):
            for key in ('index','count','categories','pack_ids','nominal_levels_db','predicted_count'):
                if row[key]!=old[key]:
                    raise ValueError('Native phase case identity/count changed')
            np.testing.assert_allclose(row['reference_power'],old['reference_power'],rtol=1e-10,atol=1e-15)
            if row['sum_relative_error']>1e-9:
                raise ValueError('Native phase mixture sum failed')
            if name=='baseline':
                for key in ('nmse','si_sdr'):
                    np.testing.assert_allclose(row[key],old[key],rtol=1e-5,atol=1e-7)
    directional=all(b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr']
                    for a,b in zip(phase['baseline']['by_count'][1:],phase['four_phase']['by_count'][1:]))
    weak=all(b['weakest_nmse']<a['weakest_nmse']
             for a,b in zip(phase['baseline']['by_count'][1:],phase['four_phase']['by_count'][1:]))
    phase_complete=read(parent/'phase/COMPLETE.json')['result']
    if directional!=phase_complete['directional_criterion']:
        raise ValueError('Phase decision mismatch')
    fit_plan=read(parent/'fit_gap/PROTOCOL.json')
    fit_complete=read(parent/'fit_gap/COMPLETE.json')
    if fit_complete['checkpoint_sha256']!=identity['sha256'] or fit_complete['selected_epoch']!=selected:
        raise ValueError('Fit-gap model differs')
    fit={}
    for epoch in (1,5):
        path=parent/'fit_gap'/f'TRAIN_EPOCH_{epoch:03d}.json'
        saved=read(path)
        rows=saved['rows']
        data=NativeMixtures(parent/'preparation','train_pack',epoch,use_features=False)
        if [r['index'] for r in rows]!=fit_plan['subsets'][str(epoch)]:
            raise ValueError('Fit-gap subset changed')
        for row in rows:
            source=data.rows[row['index']]
            clips=[data.library.clips[int(i)] for i in source['indices'][:int(source['count'])]]
            if (row['count']!=int(source['count']) or row['categories']!=[c['category'] for c in clips]
                or row['pack_ids']!=[c['pack_id'] for c in clips] or row['sum_relative_error']>1e-9):
                raise ValueError('Fit-gap case identity failed')
        groups=aggregate(rows,70)
        same_summary(groups,saved['by_count'])
        fit[str(epoch)]=dict(by_count=groups,sha256=sha256(path),
            native_adaptation_seen=saved['native_adaptation_seen'],
            exact_repeated_native_mixtures=saved['exact_repeated_native_mixtures'])
    value=dict(status='PASS',auditor_sha256=sha256(__file__),selected_epoch=selected,checkpoint_sha256=identity['sha256'],
        audits=audits,phase=phase,fit_gap=fit,phase_joint_improvement=directional,weakest_both_improved=weak,
        heldout_read=False,independent_test=False,physical_aircraft_count=False)
    write_json(output.with_suffix('.json'),value)
    lines=['# 고정 모델의 위상 평균 및 학습 적합도 진단','',
        f'원 주파수 배치 실험의 사전 NMSE 규칙으로 선택된 e{selected}를 고정했다. 추가 학습이나 위상 점수를 이용한 epoch 재선택은 없다.', '',
        '## 같은 모델: 1회 추론과 네 위상 평균','',
        '입력을 0/90/180/270도로 회전해 네 번 복원하고, 역회전한 출력끼리 순서를 맞춘 뒤 복소 평균했다. '
        '정답은 채점에만 사용했다. 원래630개발 검증의 1회 추론 수치가 재현되는지 전부 확인했다.','',
        '| 구성 수 | 추론 | NMSE ↓ | 복소 SI-SDR dB ↑ | 약신호 NMSE ↓ |','|---|---|---:|---:|---:|']
    for count in (1,2,3):
        for name in ('baseline','four_phase'):
            g=phase[name]['by_count'][count-1]
            lines.append(f"| {count} | {name} | {g['mean_nmse']:.6f} | {g['mean_si_sdr']:.3f} | {g['weakest_nmse']:.6f} |")
    lines += ['',f'두·세 성분 각각에서 NMSE 감소와 복소 SI-SDR 증가: **{directional}**. 양쪽 약신호 NMSE 감소: **{weak}**.',
        '이 판정은 방향상 변화이며 유의성 검정이 아니다. 연산량은 네 번의 순전파이고 개수 head는 바뀌지 않는다. '
        '공통 대역으로 제한한 합성 기록 성분의 결과이며, 원기록 전체 복원·새 기종·실제 동시 수신 성과를 뜻하지 않는다.','',
        '## 같은 선택 모델의 혼합 노출과 기록 변화','',
        'TRAIN epoch1/5에서 기종 조합 비율을 개발 검증과 같게 맞춘 각210혼합(구성 수별70개)을 추론 전에 고정했다. '
        '원기록을 공유하는 다른 혼합과 새 기록·대역폭의 개발 검증은 구분한다. 아래는 모두 1회 추론이다.','',
        '| 자료 | 모델의 native 학습에서 같은 혼합 노출 | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR dB | 3성분 SI-SDR dB |',
        '|---|---|---:|---:|---:|---:|']
    for epoch in (1,5):
        info=fit[str(epoch)];g=info['by_count']
        exposure=f"{info['exact_repeated_native_mixtures']}/210"
        lines.append(f"| TRAIN epoch{epoch}의 210혼합 | {exposure} | {g[1]['mean_nmse']:.6f} | {g[2]['mean_nmse']:.6f} | {g[1]['mean_si_sdr']:.3f} | {g[2]['mean_si_sdr']:.3f} |")
    g=phase['baseline']['by_count']
    lines.append(f"| 다른 기록의 개발 검증630혼합 | 학습 미사용·선택에 반복 사용 | {g[1]['mean_nmse']:.6f} | {g[2]['mean_nmse']:.6f} | {g[1]['mean_si_sdr']:.3f} | {g[2]['mean_si_sdr']:.3f} |")
    lines+=['','조합 비율은 같지만 국소 전력·활동 분포까지 같지는 않다. 개발 검증은 기록 묶음 변화와 VTSBW20 변화가 얽혀 있다. '
        '이 진단만으로 암기가 전혀 없다고 증명하거나, 특정 변화 하나를 원인으로 확정하지 않는다. '
        'native 적응 전 부모 가중치는 이전의 중심 정렬 개발 학습 이력이 있다.','',
        '검산: 43/44개 실행 소스 및 보존본, 자료 해시, 고정 가중치, 모든 예측 집계와 1회 추론 재현을 확인했다. '
        '진단 자료나 정답으로 새 가중치·배율·개수 임계값을 선택하지 않았다.','',
        '[전체 학습 결과](NATIVE_RF_FINAL.md) · [CPU 진단](NATIVE_RF_CPU_DIAGNOSTICS.md)','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS',selected_epoch=selected,joint_improvement=directional,weakest_both_improved=weak)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.run,args.output)
