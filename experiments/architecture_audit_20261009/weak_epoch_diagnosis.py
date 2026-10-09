"""Describe weak-source changes using saved validation rows only.

Post-hoc development analysis, not an independent test or causal attribution.
No prediction, I/Q access, threshold selection or training takes place here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import watch_epochs as watch


def summary(pairs):
    old = [a['nmse'][a['weakest_index']] for a,b in pairs]
    new = [b['nmse'][b['weakest_index']] for a,b in pairs]
    si_old = [a['si_sdr'][a['weakest_index']] for a,b in pairs]
    si_new = [b['si_sdr'][b['weakest_index']] for a,b in pairs]
    return dict(cases=len(pairs), weak_nmse_before=mean(old), weak_nmse_after=mean(new),
        weak_nmse_delta=mean(b-a for a,b in zip(old,new)),
        nmse_improved_cases=sum(b<a for a,b in zip(old,new)),
        nmse_worsened_cases=sum(b>a for a,b in zip(old,new)),
        weak_si_before=mean(si_old) if all(v is not None for v in si_old) else None,
        weak_si_after=mean(si_new) if all(v is not None for v in si_new) else None)


def run(root,arm,before,after,output):
    paths=[root/arm/f'VALIDATION_{epoch:03d}.json' for epoch in (before,after)]
    _,identities=watch.validate(paths[0],None)
    watch.validate(paths[1],identities)
    values=[watch.read(p) for p in paths]
    pairs=list(zip(values[0]['rows'],values[1]['rows']))
    for a,b in pairs:
        if any(a[k]!=b[k] for k in ('index','count','categories','pack_ids','reference_power','weakest_index')):
            raise ValueError('Paired source identity or local power changed')
    rows=[]
    for count in (2,3):
        selected=[(a,b) for a,b in pairs if a['count']==count]
        rows.append(dict(count=count,group_kind='all',group='all',**summary(selected)))
        categories=sorted({a['categories'][a['weakest_index']] for a,b in selected})
        for category in categories:
            sub=[(a,b) for a,b in selected if a['categories'][a['weakest_index']]==category]
            rows.append(dict(count=count,group_kind='weak_category',group=category,**summary(sub)))
        for lo,hi,label in ((0,10,'[0,10)dB'),(10,20,'[10,20)dB'),(20,float('inf'),'[20,infinity)dB')):
            sub=[(a,b) for a,b in selected if lo<=10*math.log10(max(a['reference_power'])/min(a['reference_power']))<hi]
            if sub:rows.append(dict(count=count,group_kind='local_strong_weak_power_gap',group=label,**summary(sub)))
    result=dict(status='COMPLETE_SAVED_VALIDATION_PAIRED_DIAGNOSIS',arm=arm,
        before=before,after=after,rows=rows,source_metrics_sha256={str(e):hashlib.sha256(p.read_bytes()).hexdigest() for e,p in zip((before,after),paths)},
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset='RFUAV native same-band development validation,210mixtures per count',
        post_hoc=True,independent_test=False,causal_attribution=False,new_iq_read=False,training=False)
    watch.write(output.with_suffix('.json'),result)
    text=['# 약신호 지표의 epoch 간 변화', '',
        f'{arm}, e{before}→e{after}. 같은 검증 행·정답 순서·원기록·실제 전력을 확인한 뒤 집계했다. '
        '새 I/Q를 읽거나 모델을 학습하지 않은 사후 기술 분석이다. 혼합 창을 독립 원기록으로 세지 않는다.', '',
        '|성분 수|집계|조건|혼합 수|약신호 NMSE 이전→이후↓|개선/악화 혼합 수|약신호 SI-SDR 이전→이후↑ dB|',
        '|---:|---|---|---:|---|---|---|']
    for r in rows:
        fmt=lambda v:'미정의' if v is None else f'{v:.4f}'
        text.append(f"|{r['count']}|{r['group_kind']}|{r['group']}|{r['cases']}|"
            f"{fmt(r['weak_nmse_before'])}→{fmt(r['weak_nmse_after'])}|"
            f"{r['nmse_improved_cases']}/{r['nmse_worsened_cases']}|"
            f"{fmt(r['weak_si_before'])}→{fmt(r['weak_si_after'])}|")
    text+=['','SI-SDR은 일정한 복소 배율에 불변이므로, SI-SDR 개선과 NMSE 악화가 함께 나타날 수 있다. '
        '이 지표만으로 진폭 부족·위상 오차·누출 중 원인을 확정하지 않는다. 실제 전력차는 가장 강한 성분과 '
        '가장 약한 성분의 창별 전력비다. 평균을 개선하기 위해 실패 사례를 제외하거나 선택 규약을 변경하지 않았다.','']
    watch.write(output.with_suffix('.md'),'\n'.join(text))
    print(json.dumps([r for r in rows if r['group_kind']=='all']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--before',type=int,required=True)
    p.add_argument('--after',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.arm,a.before,a.after,a.output)
