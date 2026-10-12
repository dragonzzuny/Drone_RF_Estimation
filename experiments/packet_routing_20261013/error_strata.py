"""Descriptive reconstruction diagnostics from already exposed DEV rows only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def describe(rows):
    n=np.array([v['nmse'] for v in rows]);si=np.array([v['si_sdr'] for v in rows])
    return dict(sources=len(rows),mean_nmse=float(n.mean()),median_nmse=float(np.median(n)),
        p90_nmse=float(np.quantile(n,.9)),mean_si_sdr=float(si.mean()),
        nmse_below_point1=int((n<.1).sum()),nmse_above_1=int((n>1).sum()))


def main(epoch):
    p=ROOT/'local/timeblock_extend100_20261012_v1';out=ROOT/'reports/2026-10-13'
    event=json.loads((p/f'EPOCH_{epoch:03d}.json').read_text());audit=json.loads((p/f'AUDIT_{epoch:03d}.json').read_text())
    source=p/f'VALIDATION_{epoch:03d}.json'
    assert audit['status']=='PASS' and audit['event_sha256']==sha(p/f'EPOCH_{epoch:03d}.json')
    assert sha(source)==event['validation_sha256']
    source_rows=json.loads(source.read_text())['rows'];rows=[]
    for row in source_rows:
        if row['count']==1:continue
        powers=np.array(row['reference_power']);total=powers.sum()
        for j,cat in enumerate(row['categories']):
            sir=10*np.log10(powers[j]/(total-powers[j]))
            rows.append(dict(case=row['index'],count=row['count'],category=cat,
                actual_sir_db=float(sir),nominal_level_db=row['nominal_levels_db'][j],
                weakest=j==row['weakest_index'],nmse=row['nmse'][j],si_sdr=row['si_sdr'][j]))
    bins=[-float('inf'),-20,-10,-3,3,10,20,float('inf')]
    strata=[]
    for k in (2,3):
        for lo,hi in zip(bins[:-1],bins[1:]):
            r=[v for v in rows if v['count']==k and lo<=v['actual_sir_db']<hi]
            if r:strata.append(dict(count=k,sir_interval_db=f'[{lo},{hi})',**describe(r)))
    by_cat=[]
    for k in (2,3):
        for cat in sorted(set(r['category'] for r in rows if r['count']==k)):
            r=[v for v in rows if v['count']==k and v['category']==cat]
            weak=[v for v in r if v['weakest']]
            by_cat.append(dict(count=k,category=cat,**describe(r),weakest_subset=describe(weak) if weak else None))
    result=dict(epoch=epoch,status='VERIFIED_SAVED_ROWS',validation_sha256=sha(source),
        by_actual_sir=strata,by_category=by_cat,
        definitions=dict(actual_sir='Target reference power / sum of other source reference powers within evaluated0.63872ms',
            noise='Noise in each original recording is included in its reference',
            thresholds='Descriptive fixed NMSE0.1 and1; not device acceptance thresholds'),
        limitation='Same-record B development; no independence or causal conclusions; no new I/Q access')
    (out/f'B_ERROR_STRATA_E{epoch:03d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=[f'# Epoch{epoch}: 잔차가 남는 조건', '',
        '이미 평가한630개 B 개발 사례의 저장 수치만 분석했다. 실제 SIR은 '
        '평가한0.63872ms 안에서 해당 정답의 전력을 나머지 정답 전력 합으로 나눈 값이다. '
        '원기록의 수신 잡음도 정답에 포함되므로 송신 신호만의 SIR과는 구분한다.', '',
        '|성분 수|실제 SIR(dB)|성분 수|평균 NMSE↓|중앙값|90백분위|평균 복소SI-SDR↑|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in strata:lines.append(f"|{r['count']}|{r['sir_interval_db']}|{r['sources']}|{r['mean_nmse']:.4f}|{r['median_nmse']:.4f}|{r['p90_nmse']:.4f}|{r['mean_si_sdr']:.2f}dB|")
    lines += ['', '|성분 수|기종|평균 NMSE|해당 기종이 최약일 때 NMSE|NMSE<0.1 성분|NMSE>1 성분|', '|---|---|---:|---:|---:|---:|']
    for r in by_cat:
        w=r['weakest_subset'];w=f"{w['mean_nmse']:.4f} (n={w['sources']})" if w else '해당 없음'
        lines.append(f"|{r['count']}|{r['category']}|{r['mean_nmse']:.4f}|{w}|{r['nmse_below_point1']}/{r['sources']}|{r['nmse_above_1']}/{r['sources']}|")
    lines += ['', 'SIR별 차이는 관찰이며 전력 차이만이 실패 원인이라는 증거가 아니다. '
        '기종·송신 활동·주파수 중첩·원녹음 조건이 함께 달라질 수 있다. '
        '창 수를 독립 녹음 수로 취급하지 않았다.0.1과1은 결과 설명용 오차 기준이며 '
        '분리 성공률이나 상용 디텍터의 허용 기준으로 부르지 않는다.', '']
    (out/f'B_ERROR_STRATA_E{epoch:03d}_KO.md').write_text('\n'.join(lines))
    print(json.dumps(strata,indent=2));print(json.dumps(by_cat,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);main(p.parse_args().epoch)
