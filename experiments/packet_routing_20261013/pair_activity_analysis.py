"""Saved-score/FIT-template diagnostic; no waveform inference or tuning."""
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x):return statistics.mean(x) if x else None


def main():
    parent=ROOT/'local/timeblock_extend100_20261012_v1'
    spectral=ROOT/'local/b_timeblock_spectral_20261013_v1'
    p=parent/'VALIDATION_099.json';s=spectral/'RESULT.json';t=spectral/'TEMPLATES.json'
    e=read(parent/'EPOCH_099.json');a=read(parent/'AUDIT_099.json');sa=read(spectral/'AUDIT.json')
    assert a['status']=='PASS' and a['event_sha256']==sha(parent/'EPOCH_099.json')
    assert sha(p)==e['validation_sha256']
    assert sa['status']=='PASS' and sha(s)==sa['result_sha256'] and sha(t)==sa['templates_sha256']
    base,control,templates=read(p),read(s),read(t)
    assert base['rows_sha256']==control['fixed_dev_schedule_sha256']
    assert not templates['dev_contexts_used']
    overlaps=[]
    for center,bank in templates['banks'].items():
        for i,j in itertools.combinations(range(len(bank['categories'])),2):
            x,y=bank['psd'][i],bank['psd'][j]
            x=[v/sum(x) for v in x];y=[v/sum(y) for v in y]
            overlaps.append(dict(common_receiver_center_hz=int(center),
                categories=[bank['categories'][i],bank['categories'][j]],
                overlap_mass=sum(min(v,w) for v,w in zip(x,y)),
                cosine=sum(v*w for v,w in zip(x,y))/math.sqrt(sum(v*v for v in x)*sum(w*w for w in y))))
    groups=defaultdict(list);sources=defaultdict(list)
    assert len(base['rows'])==len(control['rows'])==630
    for left,right in zip(base['rows'],control['rows']):
        for key in ('index','count','categories','weakest_index'):assert left[key]==right[key]
        assert all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-14)
                   for x,y in zip(left['reference_power'],right['reference_power']))
        if left['count']==1:continue
        groups[tuple(left['categories'])].append((left,right))
        weights=[10**(v/10) for v in left['nominal_levels_db']]
        for j,cat in enumerate(left['categories']):
            activity=left['reference_power'][j]/(weights[j]/sum(weights))
            sources[(left['count'],cat,activity<.1)].append(dict(
                relative_power=activity,nmse=left['nmse'][j],spectral_nmse=right['nmse'][j],
                si=left['si_sdr'][j]))
    pair_rows=[]
    for cats,rows in sorted(groups.items()):
        def stats(which):
            values=[x for pair in rows for x in pair[which]['nmse']]
            si=[x for pair in rows for x in pair[which]['si_sdr']]
            return dict(mean_nmse=mean(values),median_nmse=statistics.median(values),
                mean_si_sdr=mean(si) if all(v is not None for v in si) else None,
                undefined_si_sources=sum(v is None for v in si),
                weakest_nmse=mean([pair[which]['nmse'][pair[which]['weakest_index']] for pair in rows]))
        pair_rows.append(dict(categories=list(cats),cases=len(rows),parent99=stats(0),spectral=stats(1)))
    activity_rows=[dict(count=k,category=cat,relative_power_below_point1=low,sources=len(rows),
        mean_nmse=mean([v['nmse'] for v in rows]),spectral_mean_nmse=mean([v['spectral_nmse'] for v in rows]),
        mean_relative_power=mean([v['relative_power'] for v in rows]))
        for (k,cat,low),rows in sorted(sources.items())]
    result=dict(status='VERIFIED_SAVED_ROWS',parent_epoch=99,pairs=pair_rows,activity=activity_rows,
        fit_psd_overlap=overlaps,input_sha256={str(q.relative_to(ROOT)):sha(q) for q in (p,s,t)},
        new_inference=False,threshold_source='Existing descriptive activity ratio0.1, not model threshold tuning',
        definitions=dict(overlap_mass='Sum of minimum of two unit-sum FIT average PSDs, not per-window RF overlap',
            activity='Local reference power / full20.8896ms nominal normalized source power; not noise-ground-truth'),
        limitation='One recording group per category; mixed nominal levels; posthoc descriptive, not causal or independent validation')
    out=ROOT/'reports/2026-10-13'
    (out/'PAIR_ACTIVITY_E099.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 조합·국소 활동별 부모99의 복원 오차','',
        '기존630개 평가 수치와 학습 전용 PSD 템플릿을 재분석했다. 추가 추론·학습·임계값 탐색은 없다.','',
        '|조합|혼합 수|U-Net99 NMSE↓|기존 스펙트럼 기준 NMSE↓|U-Net 최약 NMSE↓|',
        '|---|---:|---:|---:|---:|']
    for row in pair_rows:
        lines.append(f"|{' + '.join(row['categories'])}|{row['cases']}|{row['parent99']['mean_nmse']:.6f}|{row['spectral']['mean_nmse']:.6f}|{row['parent99']['weakest_nmse']:.6f}|")
    lines+=['','|성분 수|기체|국소 상대 전력<0.1|성분 수|U-Net NMSE|스펙트럼 기준 NMSE|',
        '|---|---|---|---:|---:|---:|']
    for row in activity_rows:
        lines.append(f"|{row['count']}|{row['category']}|{row['relative_power_below_point1']}|{row['sources']}|{row['mean_nmse']:.6f}|{row['spectral_mean_nmse']:.6f}|")
    lines+=['','|조합|학습 평균PSD 겹침 질량|','|---|---:|']
    for row in overlaps:lines.append(f"|{' + '.join(row['categories'])}|{row['overlap_mass']:.6f}|")
    lines+=['','겹침 질량은 두 정규화된 학습 평균PSD의 bin별 최솟값을 합한 기술통계다.',
        '평균PSD가 떨어져 있어도 순간 패킷의 중첩 여부와 저활동 구간 잡음 분리의 난도는 별도다.',
        '수신 중심 주파수는 송신 신호의 실제 중심 또는 점유 대역폭과 같다고 가정하지 않았다.',
        '국소 상대 전력0.1은 기존 사후 설명 기준이다. 신호/잡음 정답이나 실패의 원인을 확정하지 않는다.',
        '기체·기록·전력·활동 조건이 함께 달라지며, 표의 창 수는 독립 녹음 수가 아니다.','']
    (out/'PAIR_ACTIVITY_E099_KO.md').write_text('\n'.join(lines))
    print(json.dumps(dict(pairs=pair_rows,activity=activity_rows),ensure_ascii=False))


if __name__=='__main__':main()
