"""Reaggregate the adaptive intermediate-LR diagnostic without I/Q reads."""
import argparse
import json
import math
from pathlib import Path
import statistics
import report_head_probes as common


def run(root, previous, output):
    p = common.read(root/'PROTOCOL.json')
    c = common.read(root/'COMPLETE.json')
    old = common.read(previous)
    assert c['status'] == 'COMPLETE' and c['protocol_sha256'] == common.digest(root/'PROTOCOL.json')
    assert c['initial_predictions_exactly_equal'] and c['weights_discarded']
    assert not c['validation_read'] and not c['heldout_read']
    assert p['reviewed_evidence_sha256'] == common.digest(previous)
    assert p['new_lr'] == 1e-4 and p['old_lr'] == 1e-5
    assert p['train_indices'] == [4,5,2,11] and p['updates_per_arm'] == 64
    for rel, sha in p['source_sha256'].items():
        assert common.digest(common.ROOT/rel) == sha and common.digest(root/'source_snapshot'/rel) == sha
    assert [a['arm'] for a in c['results']] == ['unbalanced','balanced']
    for a in c['results']:
        assert a == common.read(root/(a['arm']+'_COMPLETE.json'))
        assert a['history'] == common.read(root/(a['arm']+'_HISTORY.json'))['history']
        assert a['updates'] == 64
        assert a['optimizer_groups'] == [dict(lr=1e-5,parameters=32142859),dict(lr=1e-4,parameters=37888)]
        assert [h['step'] for h in a['history']] == [0,1,8,16,32,64]
        for h in a['history']:
            rows = h['rows']
            assert [r['count'] for r in rows] == [2,2,3,3]
            assert [r['case'] for r in rows] == [0,1,2,3]
            for r in rows:
                assert len(r['nmse']) == len(r['si_sdr']) == r['count']
                assert all(math.isfinite(v) and v >= 0 for v in r['nmse'])
                assert all(math.isfinite(v) for v in r['si_sdr'])
            for group in h['by_count']:
                for key, metric in [('nmse','mean_nmse'),('si_sdr','mean_si_sdr')]:
                    value = statistics.mean(v for r in rows if r['count'] == group['count'] for v in r[key])
                    assert common.close(value,group[metric])
    assert c['results'][0]['history'][0] == c['results'][1]['history'][0]
    original = old['studies']['input_balance']
    assert original['parent_checkpoint_sha256'] == p['parent_checkpoint_sha256']
    for a, b in zip(c['results'][0]['history'][0]['rows'], original['results'][0]['history'][0]['rows']):
        assert all(abs(x-y) < 1e-8 for key in ('nmse','si_sdr') for x,y in zip(a[key],b[key]))
    comparisons=[]
    for count in (2,3):
        a,b=[next(g for g in arm['history'][-1]['by_count'] if g['count']==count) for arm in c['results']]
        comparisons.append(dict(count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],
            si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],
            both_improved=b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr']))
    result=dict(status='PASS',auditor_sha256=common.digest(Path(__file__)),
        protocol_sha256=common.digest(root/'PROTOCOL.json'),complete_sha256=common.digest(root/'COMPLETE.json'),
        comparisons=comparisons,all_four_mean_directions_improved=all(v['both_improved'] for v in comparisons),
        results=c['results'],new_lr=p['new_lr'],all_metric_rows_reaggregated=True,
        initial_parent_reproduced=True,frozen_sources_checked=True,
        reported_optimizer_groups_checked=True,discarded_optimizer_not_independently_read=True,
        adaptive_train_only=True,recorded_iq_reads=0,heldout_read=False)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 중간 학습률 TRAIN4 대조: 완료 결과','','앞선 결과를 보고 설계한 적응적 학습 진단이다. 독립 검증 결과가 아니다. 같은 부모·전체32,180,747파라미터·원 손실·본체lr1e-5·각64업데이트이며, 새 층만lr1e-4다. 이전 가중치를 재사용하지 않았다.','','| 군 | 단계 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB |','|---|---:|---|---|']
    for a in c['results']:
        for h in a['history']:
            x,y=h['by_count']
            lines.append(f"| {a['arm']} | {h['step']} | {x['mean_nmse']:.6f}/{y['mean_nmse']:.6f} | {x['mean_si_sdr']:.3f}/{y['mean_si_sdr']:.3f} |")
    lines+=['',f"정해 둔 마지막64의 네 평균 방향 동시 개선: **{result['all_four_mean_directions_improved']}**.",'',
        '모든 단계와 10성분 점수는 JSON에 보존했다. 평균·저장 기록·실행 소스·보고된 optimizer 그룹을 다시 확인했다. 폐기한 optimizer tensor를 독립 재검사한 것은 아니다. DEV·미학습 기체·예약 기록은 읽지 않았다. 이 결과만으로 전체 자료 성능이나 범용 분리 가능성을 주장하지 않는다.','',
        '[앞선 lr1e-3의 전체 결과](HEAD_PROBES_FINAL.md) · [적응적 설계와 사전 판단 기준](INTERMEDIATE_LR_PROBE_PLAN_KO.md)','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(dict(status='PASS',directions=result['all_four_mean_directions_improved'],comparisons=comparisons))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','previous','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();run(a.run.resolve(),a.previous.resolve(),a.output.resolve())
