"""Report fixed-final native tracking, with all cases and every training epoch."""
import json
from pathlib import Path
import numpy as np
from evaluate_parent_full_context import ROOT, read, write, sha

REPORT = ROOT/'reports/2026-10-13'
RUN = ROOT/'local/native_tracking_20261013_v1'
AUDIT = ROOT/'local/native_tracking_direct_audit_20261013_v1'
BASE = ROOT/'local/incumbent_tracking_c24_20261013_v1'


def main():
    for folder in (RUN, AUDIT, BASE):
        assert read(folder/'STATE.json')['status'] == 'COMPLETE_AUDITED'
        assert sha(folder/'RESULT.json') == read(folder/'AUDIT.json')['result_sha256']
    result, baseline = read(RUN/'RESULT.json'), read(BASE/'RESULT.json')
    assert result['epoch'] == 50 and result['optimizer_updates'] == 600
    expected = {c['index']: c for c in baseline['cases']}
    comparisons = []
    for row in result['C']['rows']:
        prior = expected[row['index']]
        assert row['count'] == prior['count']
        assert row['predicted_count'] == prior['metrics']['parent']['predicted_count']
        delta = float(np.mean(row['nmse'])-np.mean(prior['metrics']['psd']['nmse']))
        comparisons.append(dict(index=row['index'], count=row['count'], categories=row['categories'],
            nominal_levels_db=row['nominal_levels_db'], change_vs_psd=delta,
            improved=delta < -1e-10, worsened=delta > 1e-10, native_learned50=row,
            previous=prior['metrics']))
    lines = ['# native e2 맞춤 추적기: 고정 final50 결과', '',
        'native e2 U-Net은 동결하고, 같은 FIT48 문맥에서 새 출력으로 추적기만 학습했다.',
        '875,521파라미터,seed0,50epoch/600업데이트. 이전 B99 추적기와 같은 구조·초기화·예산이다.',
        'native e2는 이전 가중치의 native 주파수 조건 적응 단계2이며 처음부터 총2epoch가 아니다.', '',
        '평가는 기존에 반복 사용한 별도 녹음 C24, 전체20.8896ms. 단일5/두성분12/세성분7조건.',
        'VTSBW 조건과 원녹음이 함께 바뀐다. 독립 최종시험 또는 짧은 C630과 동일한 평가는 아니다.', '',
        '|성분 수|방법|평균NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|', '|---:|---|---:|---:|---:|']
    groups = baseline['by_method'] if 'by_method' in baseline else baseline['groups']
    for k in (1, 2, 3):
        for key, label in [('parent','native e2+C8'), ('psd','native e2+고정 PSD'),
                           ('learned50','B99 학습 추적기 전이'), ('new','native 맞춤 추적기50')]:
            m = result['C']['by_count'][str(k)] if key == 'new' else groups[key][str(k)]
            lines.append(f"|{k}|{label}|{m['mean_nmse']:.6f}|{m['mean_si_sdr']:.6f}|{m['weakest_nmse']:.6f}|")
    lines.extend(['', '|성분 수|PSD 대비 개선/악화/동일 조건|NMSE 상대 변화|', '|---:|---:|---:|'])
    verdict = {}
    for k in (1, 2, 3):
        rows = [r for r in comparisons if r['count'] == k]
        good, bad = sum(r['improved'] for r in rows), sum(r['worsened'] for r in rows)
        before, after = groups['psd'][str(k)], result['C']['by_count'][str(k)]
        change = (after['mean_nmse']/before['mean_nmse']-1)*100
        lines.append(f'|{k}|{good}/{bad}/{len(rows)-good-bad}|{change:+.2f}%|')
        verdict[str(k)] = dict(nmse_improved=after['mean_nmse'] < before['mean_nmse'],
            si_sdr_improved=after['mean_si_sdr'] > before['mean_si_sdr'],
            weakest_improved=after['weakest_nmse'] < before['weakest_nmse'], improved_cases=good, worsened_cases=bad)
    joint = all(all(verdict[str(k)][key] for key in ('nmse_improved','si_sdr_improved','weakest_improved')) for k in (2,3))
    lines.extend(['', '두·세 성분 세 지표의 공동 개선: **'+('충족' if joint else '미충족')+'**.',
        '이는 위 고정 C24의 결과이며 모든 조건의 개선이나 새로운 녹음 일반화 완료를 뜻하지 않는다.',
        '조건별 악화 및 단일 성분 결과를 보존한다. 짧은 C630 CURRENT_BEST는 유지한다.', '',
        '추적기는 복원 조각의 출력 순서만 바꾼다. 개수 예측·배경·진폭·위상은 바꾸지 않는다.',
        '정답은 FIT 지도학습과 점수 계산에만 쓰며 추론은 출력 PSD·상대전력·시간 위치를 사용한다.',
        '독립 CPU에서24개 실제 I/Q를 재배열하여 GPU 순서와 파형 점수를 검산했다.', '',
        '매epoch FIT48 결과는 학습 진단이다. native 부모가 시간 분할 전 녹음으로 학습됐으므로',
        'B를 미학습 검증으로 사용하지 않았다. 사전 지정한 최종50만 C에서 평가했다.',
        '50epoch는 추적기600업데이트이며, U-Net100epoch/7500업데이트와 다른 학습량이다.', '',
        '[실행 계획](NATIVE_TRACKING_PLAN_KO.md) · [직접 파형 감사](NATIVE_TRACKING_DIRECT_AUDIT.json)',
        '· [모든 epoch](NATIVE_TRACKING_EPOCH_REPORTS_KO.md) · [조건별 비교](NATIVE_TRACKING_CASE_COMPARISON.json)'])
    (REPORT/'NATIVE_TRACKING_RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    write(REPORT/'NATIVE_TRACKING_CASE_COMPARISON.json', dict(cases=comparisons, verdict=verdict,
        all_multisource_metrics_improved=joint, result_sha256=sha(RUN/'RESULT.json'),
        direct_audit_sha256=sha(AUDIT/'RESULT.json')))
    epochs = []
    lines = ['# native 추적기 전 epoch 기록', '',
        '**FIT48 학습 진단이며 검증 성능이 아니다.** 각 칸은 두/세 성분. 최종50 고정.', '',
        '|Epoch|NMSE↓|복소SI-SDR↑ dB|최약NMSE↓|직전 대비NMSE|', '|---:|---:|---:|---:|---:|']
    prev = read(RUN/'FIT_000.json')['by_count']
    for epoch in range(1, 51):
        d = read(RUN/f'EPOCH_{epoch:03d}.json')
        assert sha(RUN/f'EPOCH_{epoch:03d}.json') == read(RUN/f'AUDIT_{epoch:03d}.json')['event_sha256']
        a,b = d['by_count']['2'],d['by_count']['3']
        lines.append(f"|{epoch}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|{a['mean_nmse']-prev['2']['mean_nmse']:+.6f}/{b['mean_nmse']-prev['3']['mean_nmse']:+.6f}|")
        prev = d['by_count']
        epochs.append(d)
    (REPORT/'NATIVE_TRACKING_EPOCH_REPORTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(joint_multisource_improvement=joint, verdict=verdict), indent=2))


if __name__ == '__main__':
    main()
