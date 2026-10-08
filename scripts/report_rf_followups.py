"""Reaggregate followup phase inference and supervised power-loss results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from summarize_waveform_context import audit_validation,summarize,audit_completion


def read(path):
    return json.loads(path.read_text())


def phase(run):
    protocol=read(run/'PROTOCOL.json')
    digest=hashlib.sha256((run/'PROTOCOL.json').read_bytes()).hexdigest()
    models=[]
    lines=['# 네 위상 추론 평균: 개발 검증 결과','',
        '각 모델의 기존 선택 가중치를 고정하고 동일 630혼합에서 원 예측과 네 위상 평균을 비교했다.',
        '추가 학습은 없으며, 정답은 점수 계산에만 사용했다. 순전파 비용은 1회 대 4회다.',
        '네 위상 출력 순서는 정답 없이 예측끼리 전체 창에서 맞춘다. 정확한 집합 등변성을 증명한 구조가 아니다.','',
        '| 모델 | 성분 수 | 원 NMSE | 평균 NMSE | 원 SI-SDR dB | 평균 SI-SDR dB |',
        '|---|---:|---:|---:|---:|---:|']
    for name in protocol['models_fixed_before_evaluation']:
        folder=run/name
        if not (folder/'COMPLETE.json').exists():
            continue
        receipt=read(folder/'COMPLETE.json')
        if receipt['protocol_sha256']!=digest:
            raise ValueError('Phase receipt protocol differs')
        modes={}
        meta=None
        for mode in ('baseline','four_phase'):
            saved=read(folder/f'{mode.upper()}.json')
            audited=audit_validation(saved)
            if audited!=receipt[mode]['diagnostics'] or saved['by_count']!=receipt[mode]['by_count']:
                raise ValueError('Phase receipt disagrees with rows')
            identities=[{k:r[k] for k in ('index','count','categories','pack_ids','reference_power','input_si_sdr')} for r in saved['rows']]
            if meta is not None and meta!=identities:
                raise ValueError('Phase input identities changed')
            meta=identities;modes[mode]=saved
        a,b=({g['count']:g for g in modes[m]['by_count']} for m in ('baseline','four_phase'))
        criterion=all(b[k]['mean_nmse']<a[k]['mean_nmse'] and b[k]['mean_si_sdr']>a[k]['mean_si_sdr'] for k in (2,3))
        if receipt['directional_criterion']!=criterion:
            raise ValueError('Phase decision differs')
        changes=[]
        for k in (1,2,3):
            lines.append(f"| {name} | {k} | {a[k]['mean_nmse']:.6f} | {b[k]['mean_nmse']:.6f} | {a[k]['mean_si_sdr']:.3f} | {b[k]['mean_si_sdr']:.3f} |")
            changes.append(dict(count=k,nmse_percent=100*(b[k]['mean_nmse']/a[k]['mean_nmse']-1),
                si_sdr_db=b[k]['mean_si_sdr']-a[k]['mean_si_sdr']))
        models.append(dict(receipt,change=changes,all_rows_reaggregated=True))
    lines+=['','| 모델 | 2·3성분 모두 NMSE 감소 및 SI-SDR 증가 |','|---|---|']
    for m in models:
        lines.append(f"| {m['model']} | {'충족' if m['directional_criterion'] else '미충족'} |")
    lines+=['','조건별 진단: 실제 국소 최대/최소 정답 전력차. 아래 값으로 체크포인트나 사례를 다시 선택하지 않았다.','',
        '| 모델 | 성분 수 | 전력차 dB | 혼합 수 | 원 NMSE | 평균 NMSE | 원 SI-SDR | 평균 SI-SDR |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for m in models:
        for a,b in zip(m['baseline']['diagnostics'],m['four_phase']['diagnostics']):
            for x,y in zip(a['local_power_gap'],b['local_power_gap']):
                if not x['cases']:
                    continue
                label=f">{x['lower_db']}" if x['upper_db'] is None else f"{x['lower_db']}–{x['upper_db']}"
                lines.append(f"| {m['model']} | {a['count']} | {label} | {x['cases']} | {x['mean_nmse']:.6f} | {y['mean_nmse']:.6f} | {x['mean_si_sdr']:.3f} | {y['mean_si_sdr']:.3f} |")
    lines+=['','이 기준은 한 개발 집합에서의 변화 방향이며 통계적 유의성·독립 일반화의 증거가 아니다.',
        '모델 간 학습 이력이 다르다. 각 고정 가중치 내 추론 대조만 동일 조건 비교다.',
        'RFUAV 같은 원 RF 대역의 중심 정렬 합성, 100MS/s, 기록 그룹 분할이다. 원 주파수 간격은 미보존이다.',
        '정답에는 수신 잡음이 포함되며, 실제 동시 수신·물리 드론 수·기체 식별의 검증은 아니다.',
        '보류 Autel/예약 확인 파일은 열지 않았다. 위상 평균의 좋은 사례만 따로 골라 집계하지 않았다.',
        '[고정 규약](../../experiments/rfuav_phase_average_20261009/README.md).','']
    return dict(status='COMPLETE' if (run/'COMPLETE.json').exists() else 'INTERIM',
        protocol=protocol,protocol_sha256=digest,models=models), '\n'.join(lines)


def allocation(run,verify):
    result=summarize(run,('waveform_only','waveform_allocation'))
    result['allocation_meets_prespecified_acceptance']=result.pop('long_context_meets_prespecified_acceptance')
    if verify:
        result['completion_audit']=audit_completion(run,result)
    # Read the already selected checkpoints' rows. These diagnostics never
    # change selection, drop cases, or turn development cases into a new test.
    selected_rows={arm:read(run/arm/f"VALIDATION_{row['epoch']:03d}.json")['rows']
        for arm,row in result['selected_at_common_budget'].items()}
    diagnostics=[]
    for count in (1,2,3):
        by_arm={arm:[row for row in rows if row['count']==count]
            for arm,rows in selected_rows.items()}
        if any(len(rows)!=210 for rows in by_arm.values()):
            raise ValueError('Selected diagnostic case count changed')
        arms={arm:dict(cases=len(rows),
            weakest_nmse=statistics.mean(r['nmse'][r['weakest_index']] for r in rows),
            weakest_si_sdr=statistics.mean(r['si_sdr'][r['weakest_index']] for r in rows))
            for arm,rows in by_arm.items()}
        control,candidate=(by_arm[a] for a in ('waveform_only','waveform_allocation'))
        for x,y in zip(control,candidate):
            if any(x[k]!=y[k] for k in ('index','count','reference_power','weakest_index')):
                raise ValueError('Selected diagnostic input identities differ')
        gaps=[]
        if count>1:
            for lo,hi in ((0,10),(10,20),(20,math.inf)):
                indices=[i for i,r in enumerate(control)
                    if (lambda g:(g>=lo if lo==0 else g>lo) and g<=hi)(
                        10*math.log10(max(r['reference_power'])/min(r['reference_power'])))]
                groups={arm:dict(cases=len(indices),
                    mean_nmse=statistics.mean(v for i in indices for v in rows[i]['nmse']) if indices else None,
                    mean_si_sdr=statistics.mean(v for i in indices for v in rows[i]['si_sdr']) if indices else None)
                    for arm,rows in by_arm.items()}
                gaps.append(dict(lower_db=lo,upper_db=None if math.isinf(hi) else hi,arms=groups))
            if sum(g['arms']['waveform_only']['cases'] for g in gaps)!=210:
                raise ValueError('Power-gap diagnostics lost cases')
        diagnostics.append(dict(count=count,arms=arms,local_power_gap=gaps,
            candidate_case_mean_nmse_improved=sum(statistics.mean(y['nmse'])<statistics.mean(x['nmse'])
                for x,y in zip(control,candidate)),
            candidate_both_case_metrics_improved=sum(
                statistics.mean(y['nmse'])<statistics.mean(x['nmse']) and
                statistics.mean(y['si_sdr'])>statistics.mean(x['si_sdr']) for x,y in zip(control,candidate))))
    result['selected_diagnostics']=diagnostics
    lines=['# 지도학습 전력 배분 보조 손실: 동일 예산 대조','',
        f"상태: {result['status']} · 공통 {result['common_completed_epoch']} epoch, 각 {result['updates_per_arm']}업데이트.",
        '동일 STFT U-Net 선택 가중치, 양쪽 32,142,859파라미터. 후보만 원래 파형 손실에 0.1×전력 배분 KL을 추가했다.',
        '동일 RFUAV 목록, 5 epoch/375업데이트씩, seed0, AdamW1e-5, batch32/micro2. 개수 손실·파형 손실·추론 구조는 같다.',
        '배분 감독은 정답 전력의 상대 비율이며, 혼합 전력을 성분 전력 합이라고 가정하지 않는다. 추론에는 정답이 들어가지 않는다.','',
        '| 군 | epoch | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR dB | 3성분 SI-SDR dB |','|---|---:|---:|---:|---:|---:|']
    for arm,info in result['arms'].items():
        for row in info['epochs']:
            g={v['count']:v for v in row['metrics']['by_count']}
            lines.append(f"| {arm} | {row['epoch']} | {g[2]['mean_nmse']:.6f} | {g[3]['mean_nmse']:.6f} | {g[2]['mean_si_sdr']:.3f} | {g[3]['mean_si_sdr']:.3f} |")
    lines+=['','초기 epoch0을 포함하여 2·3성분 평균 NMSE로 선택했다. 아래는 공통 예산의 선택 결과다.','',
        '| 군 | 선택 epoch | 2성분 NMSE | 3성분 NMSE | 2성분 SI-SDR | 3성분 SI-SDR |','|---|---:|---:|---:|---:|---:|']
    for arm,row in result['selected_at_common_budget'].items():
        g={v['count']:v for v in row['by_count']}
        lines.append(f"| {arm} | {row['epoch']} | {g[2]['mean_nmse']:.6f} | {g[3]['mean_nmse']:.6f} | {g[2]['mean_si_sdr']:.3f} | {g[3]['mean_si_sdr']:.3f} |")
    lines+=['',f"두 지표·두 신호 수의 사전 방향 기준: {'충족' if result['allocation_meets_prespecified_acceptance'] else '미충족'}.",
        '', '선택된 모델에서 실제 국소 전력이 가장 작은 성분의 결과:', '',
        '| 성분 수 | 기존 손실 약신호 NMSE | 보조 손실 약신호 NMSE | 기존 손실 약신호 SI-SDR | 보조 손실 약신호 SI-SDR |',
        '|---|---:|---:|---:|---:|']
    for row in diagnostics:
        if row['count']==1:
            continue
        a,b=(row['arms'][k] for k in ('waveform_only','waveform_allocation'))
        lines.append(f"| {row['count']} | {a['weakest_nmse']:.6f} | {b['weakest_nmse']:.6f} | {a['weakest_si_sdr']:.3f} | {b['weakest_si_sdr']:.3f} |")
    lines+=['', '전력차별 사후 진단이며 사례 제외·재선택에 사용하지 않는다. 단위는 dB다.', '',
        '| 성분 수 | 최대/최소 국소 전력차 | 혼합 수 | 기존 NMSE | 보조 NMSE | 기존 SI-SDR | 보조 SI-SDR |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in diagnostics:
        for gap in row['local_power_gap']:
            a,b=(gap['arms'][k] for k in ('waveform_only','waveform_allocation'))
            if not a['cases']:
                continue
            label=f">{gap['lower_db']}" if gap['upper_db'] is None else f"{gap['lower_db']}–{gap['upper_db']}"
            lines.append(f"| {row['count']} | {label} | {a['cases']} | {a['mean_nmse']:.6f} | {b['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f} | {b['mean_si_sdr']:.3f} |")
    lines+=['', '모든 epoch와 검산은 함께 저장된 JSON에 있다.',
        '반복 사용한 개발 검증 630혼합이며 독립 시험이 아니다. 좋은 조건만 제외/선택하지 않는다.',
        '중심 정렬 합성은 원 RF 주파수 간격을 보존하지 않으며, 수신 잡음도 정답에 포함한다.',
        '[고정 규약](../../experiments/rfuav_power_allocation_20261009/README.md).','']
    return result,'\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=('phase','allocation'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-completion',action='store_true');args=parser.parse_args()
    result,md=phase(args.run) if args.kind=='phase' else allocation(args.run,args.verify_completion)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    args.output.with_suffix('.md').write_text(md)
    print(json.dumps(dict(status=result['status'],kind=args.kind)))
