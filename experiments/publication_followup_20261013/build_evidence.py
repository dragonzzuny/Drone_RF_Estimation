"""Recompute manuscript tables from every retained case, without reading I/Q."""
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/2026-10-13'
SOURCES = {}


def read(name):
    path = REPORT / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def summarize(rows, count):
    group = [r for r in rows if r['count'] == count]
    assert len(group) == {1: 5, 2: 12, 3: 7}[count]
    values = dict(cases=len(group), mean_nmse=mean(mean(r['nmse']) for r in group),
                  mean_si_sdr=mean(mean(r['si_sdr']) for r in group),
                  weakest_nmse=mean(r['nmse'][r['weakest_index']] for r in group),
                  count_accuracy=mean(r['predicted_count'] == count for r in group))
    assert all(math.isfinite(v) for v in values.values())
    return values


def main():
    result = read('LEARNED_TRACKING_RESULT.json')
    positionless = read('POSITIONLESS_TRACKING_RESULT.json')
    native = read('INCUMBENT_TRACKING_RESULT.json')
    matched = read('NATIVE_TRACKING_RESULT.json')
    direct = read('DIRECT_TRACKING_RESULT.json')
    comparisons = read('LEARNED_TRACKING_CASE_COMPARISON.json')['cases']
    for name in ('LEARNED_TRACKING_DIRECT_AUDIT.json', 'POSITIONLESS_TRACKING_DIRECT_AUDIT.json',
                 'NATIVE_TRACKING_DIRECT_AUDIT.json', 'DIRECT_TRACKING_DIRECT_AUDIT.json'):
        assert read(name)['status'] == 'PASS'
    datasets = []
    for group, key in [('B24', 'B'), ('C24', 'C')]:
        cases = [r for r in comparisons if r['group'] == group]
        for method in ('parent', 'psd', 'learned50'):
            datasets.append((group, 'B99', method, [r[method] for r in cases],
                             result[key]['by_count'] if method == 'learned50' else None))
        datasets.append((group, 'B99', 'positionless50', positionless[key]['rows'],
                         positionless[key]['by_count']))
    for method in ('parent', 'psd', 'learned50'):
        datasets.append(('C24', 'native_e2', method,
                         [r['metrics'][method] for r in native['cases']], native['by_method'][method]))
    datasets.extend([('C24', 'native_e2', 'matched50', matched['C']['rows'], matched['C']['by_count']),
                     ('C24', 'native_e2', 'direct50', direct['C']['rows'], direct['C']['by_count'])])
    table = []
    for group, parent, method, cases, published in datasets:
        assert len(cases) == 24
        for count in (1, 2, 3):
            values = summarize(cases, count)
            if published:
                for key in ('mean_nmse', 'mean_si_sdr', 'weakest_nmse'):
                    assert math.isclose(values[key], published[str(count)][key], abs_tol=1e-10, rel_tol=1e-10)
            table.append(dict(group=group, parent=parent, method=method, count=count, **values))
    path = REPORT / 'PAPER_EVIDENCE_TABLE.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 논문용 결과표: 같은 부모·같은 평가 안에서 비교', '',
             'CPU에서 전 사례를 재집계했다. B24와 C24 모두 반복 개발 자료이며 독립 최종시험이 아니다.',
             'B: 동일 기록의 시간 분리. C: 기록과 VTSBW 조건이 함께 달라진 평가.',
             '각 표의 관측 길이는 20.8896 ms. 짧은 C630과 평균을 섞지 않는다.',
             'native_e2는 B 시간 분리 이전 가중치여서 B의 일반화 비교에 넣지 않는다.', '']
    for group, parent in [('B24', 'B99'), ('C24', 'B99'), ('C24', 'native_e2')]:
        lines.extend([f'## {group} / {parent}', '',
                      '|방법|성분|조건 수|NMSE↓|복소 SI-SDR↑ dB|최약 NMSE↓|개수 정확도|',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for row in table:
            if (row['group'], row['parent']) == (group, parent):
                lines.append('|{method}|{count}|{cases}|{mean_nmse:.6f}|{mean_si_sdr:.6f}|'
                             '{weakest_nmse:.6f}|{count_accuracy:.4f}|'.format(**row))
        lines.append('')
    gains = {}
    for count in (2, 3):
        subset = {r['method']: r for r in table if (r['group'], r['parent'], r['count']) == ('B24', 'B99', count)}
        gains[str(count)] = 100 * (1-subset['learned50']['mean_nmse']/subset['psd']['mean_nmse'])
    lines.extend(['두/세 성분의 B99 학습 추적기 NMSE 상대 감소율 대 PSD: '
                  f"{gains['2']:.3f}% / {gains['3']:.3f}%.", '',
                  '평균 NMSE는 정확도 백분율이 아니다. 모든 단일/두/세 성분 행을 포함했다.',
                  '고정 PSD는 전력 스펙트럼 특징 군집, learned50는 B99로 학습한 최종50 추적기,',
                  'matched50는 native 맞춤 학습, direct50는 파형 오차 기반 배정이다.',
                  '논문에서 조건별 악화와 한 seed의 한계를 함께 보고한다.'])
    (REPORT / 'PAPER_EVIDENCE_TABLE_KO.md').write_text('\n'.join(lines)+'\n')
    receipt = dict(status='PASS', rows=len(table), unique_development_cases_per_group=24,
                   independent_final_test=False, raw_iq_read=False, new_training_updates=0,
                   relative_nmse_reduction_B_vs_psd_percent=gains, input_sha256=SOURCES,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   human_manuscript_verification='PENDING')
    (REPORT / 'PAPER_EVIDENCE_AUDIT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'input_sha256'}))


if __name__ == '__main__':
    main()
