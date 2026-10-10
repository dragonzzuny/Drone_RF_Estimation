"""Independent CPU aggregation and inference-count/provenance receipts."""
import numpy as np
import torch
import run as experiment

ROOT,OUT,PUBLIC,w,stats=experiment.ROOT,experiment.OUT,experiment.PUBLIC,experiment.w,experiment.stats


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    p=w.read(OUT/'PROTOCOL.json');r=w.read(OUT/'COMPLETE.json');experiment.verify(p)
    assert r==w.read(PUBLIC/'PHASE_EIGHT_RESULT.json')
    assert r['protocol_sha256']==w.digest(OUT/'PROTOCOL.json')
    baseline=w.read(experiment.Path(p['baseline']))
    for name in ('four_phase','eight_phase'):assert stats.summarize(r[name]['rows'],epoch=0)==r[name]
    changes=[]
    for old,a,b in zip(baseline['rows'],r['four_phase']['rows'],r['eight_phase']['rows']):
        for key in ('index','count','categories','pack_ids','reference_power','predicted_count','input_si_sdr'):
            assert old[key]==a[key]==b[key]
        np.testing.assert_allclose(a['nmse'],old['nmse'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(a['si_sdr'],old['si_sdr'],rtol=1e-5,atol=1e-4)
        assert a['phase_details']['forward_passes']==4 and b['phase_details']['forward_passes']==8
        for order in b['phase_details']['additional_prediction_only_orders']:assert sorted(order)==[0,1,2]
        for i in range(a['count']):
            dn=b['nmse'][i]-a['nmse'][i];ds=b['si_sdr'][i]-a['si_sdr'][i]
            changes.append(dict(index=a['index'],count=a['count'],source=i,category=a['categories'][i],
                nmse_delta=dn,si_sdr_delta=ds,jointly_better=dn<0 and ds>0,jointly_worse=dn>0 and ds<0))
    checks=stats.compare(r['eight_phase'],r['four_phase']);assert checks==r['checks']
    candidate=all(all(v for k,v in row.items() if k!='count') for row in checks)
    assert candidate==r['candidate'] and r['optimizer_updates']==0 and r['forward_passes']==5040
    audit=dict(status='PASS',cases=630,sources=1260,checks=checks,candidate=candidate,
        jointly_better=sum(v['jointly_better'] for v in changes),jointly_worse=sum(v['jointly_worse'] for v in changes),
        rows=changes,independent_cpu_waveform_reinference=False,
        maximum_sum_relative_error=max(v['sum_relative_error'] for name in ('four_phase','eight_phase') for v in r[name]['rows']),
        result_sha256=w.digest(OUT/'COMPLETE.json'),auditor_sha256=w.digest(experiment.Path(__file__)),heldout_read=False)
    w.write(PUBLIC/'PHASE_EIGHT_AUDIT.json',audit)
    lines=['# 동일 최선 가중치의4위상 대8위상 추론','',
        f"DEV630,학습0회,5040forward,{r['seconds']:.1f}초. 기준4위상630행을 재현하고 CPU 집계·선택 감사를 통과했다.",
        '', '|성분 수|4위상 NMSE|8위상 NMSE|4위상 복소SI-SDR dB|8위상 복소SI-SDR dB|최약 NMSE4→8|',
        '|---|---:|---:|---:|---:|---|']
    for a,b in zip(r['four_phase']['by_count'],r['eight_phase']['by_count']):
        lines.append(f"|{a['count']}|{a['mean_nmse']:.6f}|{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}|{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}→{b['weakest_nmse']:.6f}|")
    lines+=['', '사전 공동 개선 기준 '+('통과:8위상 개선 후보.' if candidate else '실패:4위상 유지.'),
        f"1260성분 중 공동 개선{audit['jointly_better']}개,공동 악화{audit['jointly_worse']}개. 모든행을 JSON에 남겼다.",
        '', '같은5개DEV기록에서 반복 개발 평가한 한 seed다. 독립 확인·통계적 유의성이나 실제 드론 대수 검증은 아니다. '
        '4위상보다 추론 횟수가2배이며 가중치는 변하지 않았다. 정답은 채점에만 썼다. '
        '최약 성분 잔차와 미개봉 heldout을 그대로 유지한다. CPU는 기록 집계를 검산했으며 전체 파형을 재추론하지 않았다.','']
    (PUBLIC/'PHASE_EIGHT_RESULT_KO.md').write_text('\n'.join(lines))
    print(dict(status='PASS',candidate=candidate,jointly_better=audit['jointly_better'],jointly_worse=audit['jointly_worse']),flush=True)


if __name__=='__main__':main()
