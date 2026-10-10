"""Compare saved, identity-matched DEV rows; never load recorded I/Q."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/architecture_audit_20261009'))
import watch_epochs as w


def report(include_fresh=False):
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    paths={'parent':Path(old['baseline']),
        'schedule1_reset_adam':Path(old['study'])/'retained_unet/VALIDATION_001.json',
        'schedule3_continued_adam':ROOT/'local/native_frequency_20261009_v1/gpu/VALIDATION_003.json'}
    if include_fresh:
        assert w.read(ROOT/'reports/2026-10-10/FRESH_SCHEDULE_AUDIT.json')['status']=='PASS'
        paths['schedule3_reset_adam']=ROOT/'local/fresh_schedule_20261010_v1/VALIDATION_001.json'
    parent,ids=w.validate(paths['parent'],None)
    baseline={r['index']:r for r in w.read(paths['parent'])['rows']}
    summaries={}; strata=[]
    for label,path in paths.items():
        v,_=w.validate(path,ids); summaries[label]=v
        groups=defaultdict(list)
        for r in w.read(path)['rows']:
            ref=baseline[r['index']]; n=r['count']
            order=sorted(range(n),key=lambda j:(-r['reference_power'][j],j))
            for j in range(n):
                nmse=r['nmse'][j]; si=r['si_sdr'][j]
                assert math.isfinite(nmse) and math.isfinite(si)
                entry=dict(nmse=nmse,si=si,dn=nmse-ref['nmse'][j],ds=si-ref['si_sdr'][j])
                for kind,key in [('category',r['categories'][j]),('power_rank',order.index(j)+1),('recording_group',r['pack_ids'][j])]:
                    groups[(n,kind,str(key))].append(entry)
        for (count,kind,key),items in sorted(groups.items()):
            strata.append(dict(method=label,count=count,stratum=kind,label=key,source_cases=len(items),
                nmse=statistics.mean(r['nmse'] for r in items),si_sdr=statistics.mean(r['si'] for r in items),
                delta_nmse=statistics.mean(r['dn'] for r in items),delta_si_sdr=statistics.mean(r['ds'] for r in items),
                both_improved=sum(r['dn']<0 and r['ds']>0 for r in items),
                both_worsened=sum(r['dn']>0 and r['ds']<0 for r in items)))
    out=dict(status='COMPLETE_CHECKED' if include_fresh else 'HISTORICAL_CONTEXT_CHECKED',
        summaries=summaries,strata=strata,validation_sha256={k:w.digest(v) for k,v in paths.items()},
        reporter_sha256=w.digest(Path(__file__)),all_630_identities_checked=True,iq_reads=0,new_inference=0,
        heldout_read=False,independent_test=False,
        interpretation='Native e3 starts from native e2 weights, continuing AdamW state with microbatch2. '
        'Reset trials start from the same selected e2 weights with fresh AdamW, microbatch1. '
        'Thus historical native e3 is supporting context, not a fully matched reset-only experiment.')
    public=ROOT/'reports/2026-10-10';w.write(public/'FRESH_SCHEDULE_HISTORY.json',out)
    lines=['# 학습 이력을 포함한 같은 DEV 비교','',
        '모든630혼합의 정답·기종·기록·전력 지문을 대조하고 저장된 행을 재집계했다. 추가추론·I/Q읽기0회다. '
        '기존native e3는 선택부모e2에서 schedule3으로75업데이트하면서 AdamW상태를 이어간 결과다. '
        '이번새 비교는 같은e2에서 optimizer를 새로 시작한다. 과거e3는microbatch2였고이번은1이므로 optimizer만 바꾼 완전대조라고 부르지 않는다.','',
        '| 조건 | NMSE2/3 ↓ | 복소SI-SDR2/3 ↑ dB | 최약NMSE2/3 ↓ |','|---|---|---|---|']
    for label,v in summaries.items():
        a,b=v['by_count'][1:]
        lines.append(f"|{label}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|")
    lines+=['','기존schedule3연속학습은 세 성분의세지표를개선했지만 두 성분은 악화했다. '
        '새 혼합 자체로 문제가 해결된다고 미리 결론내리지 않는다. '
        '이번reset비교 결과와함께 다음실험을 선택하며 새구조를자동실행하지 않는다.',
        '', '전체개수·모든기종·전력순위·기록묶음 집계는JSON에보존했다. '
        '기종과DEV원기록은각하나씩묶여 있으므로 두효과를인과적으로분리하지 않는다. '
        '동일DEV를반복사용한사후설명이며 독립검증·유의성주장·유리한하위집합선택에사용하지않는다.']
    (public/'FRESH_SCHEDULE_HISTORY_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['by_count'][1:] for k,v in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-fresh',action='store_true');report(p.parse_args().include_fresh)
