"""Audit all overlap-alignment cases, separating oracle and usable matching."""
import argparse
import statistics
from pathlib import Path
from report import read,digest,write,ROOT


def run(root,folder):
    p=read(root/'PROTOCOL.json');r=read(root/'COMPLETE.json')
    assert r['status']=='COMPLETE' and r['protocol_sha256']==digest(root/'PROTOCOL.json')
    assert len(r['rows'])==42 and len(r['details'])==54 and len(r['by_count'])==21
    assert r['updates']==0 and r['inferences']==18 and not r['heldout_read']
    for rel,sha in p['source_sha256'].items():assert digest(ROOT/rel)==digest(root/'source_snapshot'/rel)==sha
    for summary in r['by_count']:
        rows=[q for q in r['rows'] if all(q[k]==summary[k] for k in ('alignment','averaging','count'))]
        assert len(rows)==summary['cases']==2
        expected=dict(nmse=statistics.mean(statistics.mean(q['nmse']) for q in rows),
            si_sdr=statistics.mean(statistics.mean(q['si_sdr']) for q in rows),
            weakest_nmse=statistics.mean(q['nmse'][q['weakest_index']] for q in rows))
        for key,value in expected.items():assert abs(value-summary[key])<1e-12
    for q in r['rows']+r['details']:
        assert q['sum_relative_error']<1e-10
        for error,power,nmse in zip(q['absolute_error_power'],q['reference_power'],q['nmse']):assert abs(error/power-nmse)<1e-12
    independent=[]
    for summary in r['by_count']:
        if summary['alignment']=='central':continue
        rows=[q for q in r['rows'] if all(q[k]==summary[k] for k in ('alignment','averaging','count'))]
        independent.append(dict(alignment=summary['alignment'],averaging=summary['averaging'],count=summary['count'],
            mean_nmse=statistics.mean(statistics.mean(q['independently_scored_nmse']) for q in rows),
            mean_si_sdr=statistics.mean(statistics.mean(q['independently_scored_si_sdr']) for q in rows)))
    audit=dict(status='PASS',protocol_sha256=r['protocol_sha256'],complete_sha256=digest(root/'COMPLETE.json'),
        auditor_sha256=digest(Path(__file__)),all_42_averaged_and_54_window_rows_checked=True,
        all_21_summaries_reaggregated=True,independent_global_pit_scores=independent,
        source_hashes_checked=True,original_mse_average_reproduced=r['baseline_reproduced'],
        waveforms_independently_reinferred=False,heldout_read=False,
        oracle_results_are_diagnostic_only=True)
    write(folder/'OVERLAP_ALIGNMENT_AUDIT.json',audit)
    def get(name,average,n):return next(s for s in r['by_count'] if s['alignment']==name and s['averaging']==average and s['count']==n)
    lines=['# 창 간 출력 정렬과 분리 오차의 구분','',
        '기존 TRAIN6의 모든 사례를 다시 사용한 사후 진단이다. 같은 부모 모델을 CPU에서18회 추론했다. '
        '정답 없이 MSE로 정렬, 정답 없이 복소 coherence로 정렬, 정답 PIT 대응을 사용한 진단용 정렬을 비교했다. '
        '마지막 조건은 실제 적용 가능한 방법이 아니다.','',
        '| 정렬·병합 | 두 성분 NMSE ↓ | 두 성분 복소 SI-SDR ↑ dB | 세 성분 NMSE ↓ | 세 성분 복소 SI-SDR ↑ dB |',
        '|---|---:|---:|---:|---:|']
    for name,average in [('central','none')]+[(a,b) for a in p['comparison'] for b in ('uniform','center_weighted')]:
        a,b=get(name,average,2),get(name,average,3)
        lines.append(f"| {name} / {average} | {a['nmse']:.6f} | {a['si_sdr']:.3f} | {b['nmse']:.6f} | {b['si_sdr']:.3f} |")
    lines+=['',
        '복소 상관으로 바꿔도 두·세 성분의 공동 개선은 없었다. 정답으로 순서를 알려준 단순 평균에서는 '
        '두·세 NMSE가 중앙 단일 창보다 낮아졌지만 두 성분 SI-SDR은 악화했다. 따라서 정렬 오류가 일부 영향을 주지만 '
        '순서만 완벽히 알면 정밀 분리가 해결된다고 해석할 수 없다.',
        '', '42개 중앙/평균 행,54개 개별 창 행,21개 집계를 검산했다. 이전 MSE 평균은 같은 값으로 재현됐다. '
        '주 표는 중앙 창의 정답 대응을 고정하며, 평균 결과의 전역 출력 순서를 다시 PIT로 평가한 수치는 검산 JSON에 함께 보존했다. '
        '검산에서 원파형 추론을 다시 수행한 것은 아니다.',
        '', '이번 판단은 고정 TRAIN6의311.04μs 공통 구간에 한정한다. 부모 DEV 성능이나 새로운 독립 평가를 대체하지 않는다. '
        '상관 기반 정렬을 새 최선으로 채택하지 않는다. 겹치는 두 창의 지도학습과 추가 일관성을 구분하는 후속 가설을 유지한다.',
        '', '[전체 결과](OVERLAP_ALIGNMENT_RESULT.json) · [검산 및 독립 PIT 점수](OVERLAP_ALIGNMENT_AUDIT.json) · '
        '[기존 창 비교](WINDOW_OVERLAP_KO.md) · [후속 학습 규약](PAIRED_WINDOW_PLAN_KO.md)']
    (folder/'OVERLAP_ALIGNMENT_KO.md').write_text('\n'.join(lines)+'\n')
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
    print(run(a.run.resolve(),a.folder.resolve()))
