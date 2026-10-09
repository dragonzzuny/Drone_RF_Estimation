"""Independently reaggregate completed TRAIN4 probes; no waveform/model reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def check(root,expected_steps,arms):
    p=read(root/'PROTOCOL.json');c=read(root/'COMPLETE.json')
    assert c['status']=='COMPLETE' and c['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert c['initial_predictions_exactly_equal'] and c['weights_discarded'] and not c['validation_read'] and not c['heldout_read']
    for rel,sha in p['source_sha256'].items():
        assert digest(ROOT/rel)==sha and digest(root/'source_snapshot'/rel)==sha
    assert [a['arm'] for a in c['results']]==arms
    for a in c['results']:
        assert a==read(root/(a['arm']+'_COMPLETE.json'))
        assert a['updates']==expected_steps
        assert a['optimizer_groups']==[dict(lr=1e-5,parameters=32142859),dict(lr=(1e-5 if a['arm']=='head_lr_1e5' else 1e-3),parameters=37888)]
        h=a['history'];assert [x['step'] for x in h]==p['observations']
        assert h==read(root/(a['arm']+'_HISTORY.json'))['history']
        for point in h:
            rows=point['rows'];assert [r['case'] for r in rows]==list(range(4))
            assert [r['count'] for r in rows]==[2,2,3,3]
            for r in rows:
                assert len(r['nmse'])==r['count'] and len(r['si_sdr'])==r['count']
                assert all(math.isfinite(v) and v>=0 for v in r['nmse'])
                assert all(math.isfinite(v) for v in r['si_sdr'])
            for g in point['by_count']:
                for key,out in [('nmse','mean_nmse'),('si_sdr','mean_si_sdr')]:
                    actual=stats.mean(v for r in rows if r['count']==g['count'] for v in r[key])
                    assert close(actual,g[out])
        assert len(a['direct_head_effect'])==4
        assert all(len(x['raw_slot_delta_energy_over_mixture'])==4 and all(math.isfinite(v) and v>=0 for v in x['raw_slot_delta_energy_over_mixture']) for x in a['direct_head_effect'])
    assert c['results'][0]['history'][0]==c['results'][1]['history'][0]
    return p,c


def run(lr_root,balance_root,output):
    studies={}
    for label,root,steps,arms in [('head_lr',lr_root,32,['head_lr_1e5','head_lr_1e3']),('input_balance',balance_root,64,['unbalanced','balanced'])]:
        p,c=check(root,steps,arms)
        comparisons=[]
        for n in (2,3):
            a,b=[next(g for g in x['history'][-1]['by_count'] if g['count']==n) for x in c['results']]
            comparisons.append(dict(count=n,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],
                both_improved=b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr']))
        studies[label]=dict(protocol_sha256=digest(root/'PROTOCOL.json'),complete_sha256=digest(root/'COMPLETE.json'),
            train_indices=p['train_indices'],parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
            steps_per_arm=steps,results=c['results'],comparison=comparisons,
            all_four_mean_directions_improved=all(x['both_improved'] for x in comparisons),
            all_saved_metric_rows_reaggregated=True,all_frozen_sources_checked=True,
            reported_optimizer_groups_checked=True,discarded_optimizer_state_not_independently_read=True)
    assert studies['head_lr']['train_indices']==studies['input_balance']['train_indices']==[4,5,2,11]
    assert studies['head_lr']['parent_checkpoint_sha256']==studies['input_balance']['parent_checkpoint_sha256']
    old=studies['head_lr']['results'][1]['history'][-1]['rows']
    repeat=next(h for h in studies['input_balance']['results'][0]['history'] if h['step']==32)['rows']
    deltas={key:max(abs(a-b) for x,y in zip(old,repeat) for a,b in zip(x[key],y[key])) for key in ('nmse','si_sdr')}
    reproduces=deltas['nmse']<1e-4 and deltas['si_sdr']<.02
    result=dict(status='PASS',source_sha256=digest(Path(__file__)),studies=studies,
        repeated_1e3_control_at32_max_metric_difference=deltas,repeated_control_within_tolerance=reproduces,
        tolerance=dict(nmse=1e-4,si_sdr_db=.02),model_updates=0,recorded_iq_reads=0,
        independent_test=False,heldout_read=False,interpretation='Previously seen TRAIN4 only; no retained models or generalization claim')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 추가층의 학습률과 입력 규모: TRAIN4 후속 검사','','두 검사는 원래 32,142,859개 파라미터의 학습률을 1e-5로 유지하고 전체 모델을 학습했다. '
        '같은 보존 부모와 TRAIN4(두 성분 2혼합·세 성분 2혼합)를 사용했다. 이전 검사 가중치는 재사용하지 않았고 완료 후 모두 폐기했다. '
        '개발검증·보류 기록은 읽지 않았다.','','| 검사 | 군 | 마지막 업데이트 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB |','|---|---|---:|---|---|']
    for label,s in studies.items():
        for a in s['results']:
            x,y=a['history'][-1]['by_count']
            lines.append(f"| {label} | {a['arm']} | {a['updates']} | {x['mean_nmse']:.6f}/{y['mean_nmse']:.6f} | {x['mean_si_sdr']:.3f}/{y['mean_si_sdr']:.3f} |")
        lines += []
    lines += ['', '학습률 검사는 새 37,888개 파라미터만 1e-5 대 1e-3으로 바꿨다. 입력 규모 검사는 두 군의 새 층 학습률을 1e-3으로 고정하고, 추가층에 들어가는 공통/성분 특징의 규모만 맞췄다. 모두 같은 32,180,747개 파라미터다.', '',
        '## 사전 고정한 마지막 단계 비교', '']
    for label,s in studies.items():
        lines.append(f"- {label}: 2/3성분 NMSE·SI-SDR 네 평균 방향 모두 개선 = **{s['all_four_mean_directions_improved']}**.")
    lines += ['', f'겹치는 대조(새 층 lr1e-3)의 32단계 재현은 NMSE 최대 차이 {deltas["nmse"]:.3g}, SI-SDR 최대 차이 {deltas["si_sdr"]:.3g} dB였다. 허용 오차 내 재현: {reproduces}.', '',
        '## 단계별 전체 결과','','| 검사 | 군 | 단계 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB |','|---|---|---:|---|---|']
    for label,s in studies.items():
        for a in s['results']:
            for h in a['history']:
                x,y=h['by_count']
                lines.append(f"| {label} | {a['arm']} | {h['step']} | {x['mean_nmse']:.6f}/{y['mean_nmse']:.6f} | {x['mean_si_sdr']:.3f}/{y['mean_si_sdr']:.3f} |")
    lines += ['', '각 사례·성분 점수와 추가층을 켜고 끈 출력 차이는 JSON에 보존했다. 모든 단계의 평균을 성분별 행에서 다시 계산했고 동결 소스·저장된 완료 기록·파라미터 그룹을 확인했다. 가중치와 optimizer는 폐기했으므로 해당 tensor를 독립적으로 재검사한 것은 아니다.', '',
        '이 결과는 반복해서 본 학습 사례의 적합도다. 구조의 일반적 우월성·약한 미학습 기체의 복원 성공·논문 게재 가능성으로 확대하지 않는다. 후속 개발 대조는 같은 자료·시작점·학습량·단일 추론에서 별도로 수행해야 한다.', '']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(dict(status='PASS',directions={k:v['all_four_mean_directions_improved'] for k,v in studies.items()},repeated_control=reproduces))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('lr-run','balance-run','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.lr_run.resolve(),a.balance_run.resolve(),a.output.resolve())
