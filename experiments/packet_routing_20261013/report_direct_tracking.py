"""All fixed-final direct routing results and every epoch; no selection on C24."""
from pathlib import Path
import numpy as np
from evaluate_parent_full_context import ROOT,read,write,sha


def main():
    run=ROOT/'local/direct_tracking_20261013_v1'
    audit=ROOT/'local/direct_tracking_audit_20261013_v1'
    base=ROOT/'local/incumbent_tracking_c24_20261013_v1'
    old=ROOT/'local/native_tracking_20261013_v1'
    for p in (run,audit,base,old):
        assert read(p/'STATE.json')['status']=='COMPLETE_AUDITED'
        assert sha(p/'RESULT.json')==read(p/'AUDIT.json')['result_sha256']
    result=read(run/'RESULT.json');baseline=read(base/'RESULT.json')
    affinity=read(old/'RESULT.json');psd={r['index']:r['metrics']['psd'] for r in baseline['cases']}
    floor={r['index']:r for r in read(ROOT/'reports/2026-10-13/NATIVE_TRACKING_PERMUTATION_FLOOR.json')['groups']['C24']['cases']}
    comparisons=[]
    for r in result['C']['rows']:
        assert np.mean(r['nmse'])>=floor[r['index']]['reference_assisted_minimum_nmse']-1e-7
        delta=float(np.mean(r['nmse'])-np.mean(psd[r['index']]['nmse']))
        comparisons.append(dict(index=r['index'],count=r['count'],categories=r['categories'],
            nominal_levels_db=r['nominal_levels_db'],change_vs_psd=delta,improved=delta < -1e-10,
            worsened=delta > 1e-10,psd=psd[r['index']],direct50=r))
    report=ROOT/'reports/2026-10-13'
    lines=['# 파형 오차 기반 직접 배정 추적기 결과', '',
        '고정 native e2 출력의 같은 FIT48,동일 인코더 초기화,seed0,50epoch/600업데이트.',
        '인코더875521개에384개 출력층을 추가했다. 전체33블록에서 하나의 정답 대응을 적용하는',
        '전역 PIT의 기대 NMSE+비활성 출력 전력으로 학습했고,추론은6순열 중 최대 점수를 선택했다.',
        '입력은 관측 특징만이며 참/예측 개수로 라우팅하지 않는다. 배경과 복소 값은 보존한다.', '',
        'C24는 이미 반복 사용한20.8896ms 개발 평가다. 최종50을 고정하고 CPU 실제 I/Q 검산을 통과했다.', '',
        '|성분 수|방법|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|','|---:|---|---:|---:|---:|']
    for k in (1,2,3):
        for label,m in [('고정 PSD',baseline['by_method']['psd'][str(k)]),
                        ('native affinity50',affinity['C']['by_count'][str(k)]),
                        ('직접 배정50',result['C']['by_count'][str(k)])]:
            lines.append(f"|{k}|{label}|{m['mean_nmse']:.6f}|{m['mean_si_sdr']:.6f}|{m['weakest_nmse']:.6f}|")
    lines+=['','|성분 수|PSD 대비 개선/악화/동일 조건|','|---:|---:|']
    for k in (1,2,3):
        rows=[r for r in comparisons if r['count']==k];good=sum(r['improved'] for r in rows);bad=sum(r['worsened'] for r in rows)
        lines.append(f'|{k}|{good}/{bad}/{len(rows)-good-bad}|')
    lines+=['','단일 성분은 개선했지만 두·세 성분은 PSD보다 NMSE·SI-SDR·최약NMSE가 모두 나쁘다.',
        '따라서 다중 신호 분리용으로 채택하지 않는다. 단일의 이득을 전체 분리 개선으로 확대하지 않는다.',
        '출력층·손실·추론 배정을 함께 변경했으므로 단일 요소의 효과를 분리한 실험은 아니다.',
        '전역 파형 오차를 넣는 이 구현과 예산에서의 결과이며,원리 자체가 불가능하다는 증거는 아니다.', '',
        f"학습 손실은 epoch1 {read(run/'EPOCH_001.json')['training_loss']:.6f}에서 epoch50 {read(run/'EPOCH_050.json')['training_loss']:.6f}로 감소했다.",
        '확률적 순열의 기대 손실과 추론의 최대확률 순열은 다르며,학습 손실 감소를 직접 복원 성공으로 부르지 않는다.',
        'CPU에서 예상 손실과 실제 I/Q 오차 일치(1/2/3성분),전역 정답 순서 불변성,입력 슬롯 순열 등변성,',
        '공통 인코더 초기화 일치와 비영 기울기를 검증했다. 최종24개 GPU/CPU 순서·파형 점수도 일치했다.', '',
        '[고정 계획](DIRECT_TRACKING_PLAN_KO.md) · [CPU 수학 검사](DIRECT_TRACKING_CPU_CHECK.json)',
        '· [최종 파형 감사](DIRECT_TRACKING_DIRECT_AUDIT.json) · [조건별 비교](DIRECT_TRACKING_CASE_COMPARISON.json)',
        '· [50epoch 전체 기록](DIRECT_TRACKING_EPOCH_REPORTS_KO.md)']
    (report/'DIRECT_TRACKING_RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    write(report/'DIRECT_TRACKING_CASE_COMPARISON.json',dict(cases=comparisons,
        result_sha256=sha(run/'RESULT.json'),audit_sha256=sha(audit/'RESULT.json'),
        all_cases_respect_fixed_permutation_lower_bound=True,multisource_adopted=False))
    lines=['# 직접 배정 추적기 전 epoch', '', '**FIT48 학습 진단이며 검증이 아니다.** 각 칸은 두/세 성분.최종50 고정.', '',
        '|Epoch|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|직전 대비NMSE|','|---:|---:|---:|---:|---:|']
    prev=read(run/'FIT_000.json')['by_count']
    for e in range(1,51):
        p=run/f'EPOCH_{e:03d}.json';d=read(p);assert sha(p)==read(run/f'AUDIT_{e:03d}.json')['event_sha256']
        x,y=d['by_count']['2'],d['by_count']['3']
        lines.append(f"|{e}|{x['mean_nmse']:.6f}/{y['mean_nmse']:.6f}|{x['mean_si_sdr']:.4f}/{y['mean_si_sdr']:.4f}|{x['weakest_nmse']:.6f}/{y['weakest_nmse']:.6f}|{x['mean_nmse']-prev['2']['mean_nmse']:+.6f}/{y['mean_nmse']-prev['3']['mean_nmse']:+.6f}|")
        prev=d['by_count']
    (report/'DIRECT_TRACKING_EPOCH_REPORTS_KO.md').write_text('\n'.join(lines)+'\n')
    print('Direct routing result, all cases, all50epochs verified and reported.')


if __name__=='__main__':main()
