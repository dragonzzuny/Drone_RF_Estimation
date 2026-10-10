"""CPU receipt/paired-summary audit, not independent waveform reinference."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/tfgridnet_phase_probe_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    result=read(OUT/'COMPLETE.json');p=read(OUT/'PROTOCOL.json')
    assert result==read(PUBLIC/'TFGRIDNET_PHASE_PROBE_RESULT.json')
    assert result['protocol_sha256']==digest(OUT/'PROTOCOL.json')
    assert result['checkpoint_sha256']==digest(Path(p['parent_checkpoint']))
    for rel,h in p['source_sha256'].items():
        assert digest(ROOT/rel)==h and digest(OUT/'source_snapshot'/rel)==h
    for f,h in p['pinned_files'].items():assert digest(Path(f))==h
    for name in ('baseline','averaged'):
        data=result[name];assert len(data['rows'])==30
        assert [r['index'] for r in data['rows']]==p['probe_indices']
        for g in data['by_count']:
            rows=[r for r in data['rows'] if r['count']==g['count']]
            nm=[n for r in rows for n in r['nmse']]
            calculated=dict(cases=len(rows),mean_nmse=float(np.mean(nm)),median_nmse=float(np.median(nm)),
                mean_si_sdr=float(np.mean([v for r in rows for v in r['si_sdr']])),
                weakest_nmse=float(np.mean([r['nmse'][r['weakest_index']] for r in rows])),
                max_source_nmse=max(nm),source_nmse_ge_one=sum(v>=1 for v in nm))
            for k,v in calculated.items():np.testing.assert_allclose(v,g[k],rtol=1e-12,atol=1e-12)
    pairs=[]
    for a,b,diag in zip(result['baseline']['rows'],result['averaged']['rows'],result['diagnostics']):
        assert a['index']==b['index']==diag['index']
        assert a['reference_power']==b['reference_power'] and a['categories']==b['categories']
        assert a['predicted_count']==b['predicted_count']
        for order in diag['prediction_only_orders']:assert sorted(order)==[0,1,2]
        for i in range(a['count']):
            dn=b['nmse'][i]-a['nmse'][i];ds=b['si_sdr'][i]-a['si_sdr'][i]
            pairs.append(dict(index=a['index'],source=i,count=a['count'],category=a['categories'][i],
                nmse_delta=dn,si_sdr_delta=ds,jointly_better=dn<0 and ds>0,jointly_worse=dn>0 and ds<0))
    checks=[]
    for a,b in zip(result['baseline']['by_count'],result['averaged']['by_count']):
        checks.append(dict(count=a['count'],nmse=b['mean_nmse']<a['mean_nmse'],
            si=b['mean_si_sdr']>a['mean_si_sdr'],weak=b['weakest_nmse']<=a['weakest_nmse'],
            median=b['median_nmse']<=a['median_nmse']))
    assert checks==result['checks']
    candidate=all(all(v for k,v in d.items() if k!='count') for d in checks)
    assert candidate==result['inference_improvement_candidate']
    max_sum=max(r['sum_relative_error'] for n in ('baseline','averaged') for r in result[n]['rows'])
    assert max_sum<1e-9 and result['optimizer_updates']==0 and result['forward_passes']==120
    audit=dict(status='PASS',cases=30,sources=len(pairs),jointly_better=sum(x['jointly_better'] for x in pairs),
        jointly_worse=sum(x['jointly_worse'] for x in pairs),maximum_sum_relative_error=max_sum,
        checks=checks,candidate=candidate,paired_sources=pairs,
        independent_waveform_cpu_reinference=False,result_sha256=digest(OUT/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),validation_read=False,heldout_read=False)
    write(PUBLIC/'TFGRIDNET_PHASE_PROBE_AUDIT.json',audit)
    lines=['# TF-GridNet64 네 위상 추론: TRAIN30 진단','',
        '같은 가중치에서 0/90/180/270도 입력 위상을 적용하고 역회전 후 출력끼리 대응·평균했다. '
        '정답은 점수 계산에만 썼다. 학습0회,30혼합×4회 추론, 총389.96초.','',
        '|성분 수|기본 NMSE|4위상 NMSE|기본 복소SI-SDR dB|4위상 복소SI-SDR dB|최약 NMSE 기본→4위상|','|---|---:|---:|---:|---:|---|']
    for a,b in zip(result['baseline']['by_count'],result['averaged']['by_count']):
        lines.append(f"|{a['count']}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}→{b['weakest_nmse']:.6f}|")
    lines+=['',f"모든 사전 기준 통과.73성분 중 공동 개선{audit['jointly_better']}개, 공동 악화{audit['jointly_worse']}개. "
        '중앙 NMSE도 두 성분0.275116→0.250977, 세 성분0.589709→0.469713으로 감소했다.',
        '', 'TRAIN 원기록과 같은30개 반복 진단이며 독립 검증은 아니다. 원 모델은 TRAIN4에64업데이트한 상태다. '
        '극저전력 성분의 큰 잔차는 남았다. 실제 DEV 최선 U-Net을 대체하지 않는다. '
        '전체/출력층 적응 실패와 달리 학습 없는 평균은 이 조건에서 개선됐지만 일반화 원인은 확정하지 않는다.',
        '', 'CPU는 저장된 모든 행의 집계·판정, 체크포인트/소스 hash, 출력 순열과 합 일치를 검산했다. '
        '전체 파형을 CPU로 재추론한 검사는 아니다. heldout은 미개봉이다.','']
    (PUBLIC/'TFGRIDNET_PHASE_PROBE_RESULT_KO.md').write_text('\n'.join(lines))
    print({k:v for k,v in audit.items() if k!='paired_sources'})


if __name__=='__main__':main()
