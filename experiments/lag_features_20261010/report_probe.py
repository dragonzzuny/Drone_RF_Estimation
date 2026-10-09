"""Independent reaggregation of the completed training-only lag probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root,output):
    p=read(root/'PROTOCOL.json');c=read(root/'COMPLETE.json')
    assert c['status']=='COMPLETE' and c['protocol_sha256']==sha(root/'PROTOCOL.json')
    assert c['initial_predictions_exactly_equal'] and c['weights_discarded']
    assert not c['validation_read'] and not c['heldout_read']
    for rel,digest in p['source_sha256'].items():
        assert sha(ROOT/rel)==digest and sha(root/'source_snapshot'/rel)==digest
    assert [a['arm'] for a in c['results']]==['lag_power','lag_power_phase']
    for a in c['results']:
        assert a==read(root/(a['arm']+'_COMPLETE.json'))
        assert a['history']==read(root/(a['arm']+'_HISTORY.json'))['history']
        assert a['updates']==64 and a['parameters']==p['parameters'][a['arm']]
        assert a['optimizer_groups']==[dict(lr=1e-5,parameters=32142859),dict(lr=1e-3,parameters=a['parameters']-32142859)]
        assert [h['step'] for h in a['history']]==[0,1,8,16,32,64]
        for h in a['history']:
            rows=h['rows']
            assert [r['case'] for r in rows]==[0,1,2,3]
            assert [r['count'] for r in rows]==[2,2,3,3]
            for r in rows:
                assert len(r['nmse'])==len(r['si_sdr'])==r['count']
                assert all(math.isfinite(v) and v>=0 for v in r['nmse'])
                assert all(math.isfinite(v) for v in r['si_sdr'])
            for g in h['by_count']:
                for key,field in [('nmse','mean_nmse'),('si_sdr','mean_si_sdr')]:
                    actual=statistics.mean(v for r in rows if r['count']==g['count'] for v in r[key])
                    assert math.isclose(actual,g[field],rel_tol=1e-10,abs_tol=1e-10)
    assert c['results'][0]['history'][0]==c['results'][1]['history'][0]
    comparisons=[]
    for count in (2,3):
        a,b=[next(g for g in r['history'][-1]['by_count'] if g['count']==count) for r in c['results']]
        comparisons.append(dict(count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],both_improved=b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr']))
    result=dict(status='PASS',protocol_sha256=sha(root/'PROTOCOL.json'),complete_sha256=sha(root/'COMPLETE.json'),auditor_sha256=sha(Path(__file__)),
        results=c['results'],comparison=comparisons,all_four_mean_directions_improved=all(v['both_improved'] for v in comparisons),
        all_saved_component_metrics_reaggregated=True,frozen_sources_checked=True,
        reported_optimizer_groups_checked=True,discarded_optimizer_tensors_not_independently_read=True,
        train_only=True,recorded_iq_reads=0,heldout_read=False)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 지연 전력과 지연 복소 관계: TRAIN4 완료 대조','','같은 보존 부모·TRAIN4·전체 U-Net·원 손실·본체lr1e-5/추가층lr1e-3·seed0·각64업데이트다. 대조도 동일한 지연 관측의 전력 특징을 받는다. 후보만 복소 관계6채널과3456파라미터를 추가한다. 총 파라미터는32,146,315 대32,149,771이다.','','| 군 | 단계 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB |','|---|---:|---|---|']
    for a in c['results']:
        for h in a['history']:
            x,y=h['by_count'];lines.append(f"| {a['arm']} | {h['step']} | {x['mean_nmse']:.6f}/{y['mean_nmse']:.6f} | {x['mean_si_sdr']:.3f}/{y['mean_si_sdr']:.3f} |")
    lines+=['',f"마지막64에서 후보의 네 평균 방향 동시 개선: **{result['all_four_mean_directions_improved']}**.",'',
        '이는 반복해서 본 학습4혼합의 적합도이며 개발·미학습 기체 결과가 아니다. 모든 단계와10개 성분 지표는JSON에 보존했다. 폐기된 optimizer tensor를 독립 재검사한 것은 아니다. 정답은 학습 손실과 채점에만 쓰고 모델 입력에는 넣지 않았다. 지연 곱에는 혼합 성분 사이 교차항이 남으며, 관측 지연이 실제 통신 주기 정답이라는 주장도 하지 않는다.','',
        '[설계와 수식](LAG_FEATURE_PLAN_KO.md) · [원 규모 CPU 검사](LAG_FEATURE_CPU_CHECK.json)','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(dict(status='PASS',directions=result['all_four_mean_directions_improved'],comparison=comparisons))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve())
