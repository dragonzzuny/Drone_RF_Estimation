"""Posthoc category/power diagnostics of the already-scored ordered branch."""
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'local/ordered_branch_20261011_v1'
PUBLIC=ROOT/'reports/2026-10-11'


def read(p):return json.loads(p.read_text())


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    p=read(OUT/'PROTOCOL.json');r=read(OUT/'COMPLETE.json')
    assert read(PUBLIC/'ORDERED_BRANCH_AUDIT.json')['status']=='PASS'
    groups=collections.defaultdict(list)
    for name,path in [('single',p['baseline']),('four_phase',p['four_phase_baseline'])]:
        for a,b in zip(read(Path(path))['rows'],r[name]['rows']):
            for i in range(a['count']):
                # Relative power to SUM OF SEPARATE SOURCE POWERS, not SNR or coherent mixture power.
                fraction=a['reference_power'][i]/sum(a['reference_power'])
                relative_db=10*math.log10(fraction)
                power_bin='<-25dB' if relative_db<-25 else ('-25to-15dB' if relative_db<-15 else '>=-15dB')
                value=dict(before_nmse=a['nmse'][i],after_nmse=b['nmse'][i],before_si=a['si_sdr'][i],after_si=b['si_sdr'][i])
                for scope,label in [('category',a['categories'][i]),('relative_power_bin',power_bin)]:
                    groups[(name,a['count'],scope,label)].append(value)
    table=[]
    for key,rows in sorted(groups.items()):
        means={k:float(np.mean([v[k] for v in rows])) for k in rows[0]}
        table.append(dict(mode=key[0],count=key[1],scope=key[2],label=key[3],sources=len(rows),**means,
            jointly_better=sum(v['after_nmse']<v['before_nmse'] and v['after_si']>v['before_si'] for v in rows),
            jointly_worse=sum(v['after_nmse']>v['before_nmse'] and v['after_si']<v['before_si'] for v in rows)))
    initial=torch.load(OUT/'INITIAL.pt',map_location='cpu',weights_only=False,mmap=True)['model']
    final=torch.load(OUT/'ACTUAL_001.pt',map_location='cpu',weights_only=False,mmap=True)['model']
    weights={}
    for prefix in ('ordered_encoder.','ordered_projection.'):
        error=sum(float((v.double()-initial[k].double()).square().sum()) for k,v in final.items() if k.startswith(prefix))
        energy=sum(float(v.double().square().sum()) for k,v in initial.items() if k.startswith(prefix))
        weights[prefix]=dict(change_l2=error**.5,relative_change_l2=(error/energy)**.5)
    gate=final['gate'].tanh().double().numpy()
    result=dict(status='COMPLETE',posthoc=True,selection_changed=False,table=table,branch_weight_changes=weights,
        tanh_gate=dict(min=float(gate.min()),max=float(gate.max()),mean_abs=float(np.abs(gate).mean()),
            norm=float(np.linalg.norm(gate))),
        limitations='Repeated DEV sources share recordings; descriptive counts, not independent replicates or causal power effects. '
        'Power bins were reused from earlier TRAIN diagnostics. Sum separate source powers is not coherent mixture power or noise SNR.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),heldout_read=False)
    (PUBLIC/'ORDERED_BRANCH_CONDITIONAL_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 시간 변화 경로: 성분별 사후 분석','',
        '원래 선택 규칙은 바꾸지 않는다. 같은 기록에서 나온 반복 창이므로 성분 수를 독립 표본 수로 해석하지 않는다.',
        '', '|추론|성분 수|범주|표본|NMSE 기존→새|복소SI-SDR 기존→새 dB|공동 개선/악화|',
        '|---|---|---|---:|---|---|---|']
    for x in table:
        if x['scope']=='category':
            lines.append(f"|{x['mode']}|{x['count']}|{x['label']}|{x['sources']}|{x['before_nmse']:.6f}→{x['after_nmse']:.6f}|{x['before_si']:.3f}→{x['after_si']:.3f}|{x['jointly_better']}/{x['jointly_worse']}|")
    lines+=['',f"tanh(gate)의 평균 절댓값{np.abs(gate).mean():.6f}, 범위[{gate.min():.6f}, {gate.max():.6f}]. "
        '가중치가 변했다는 사실은 새 기록에서 유용해졌다는 증거와 구분한다.',
        '', '전력 구간별 전체 숫자는 JSON에 보존했다. 분모는 개별 정답 전력의 합이며 수신 SNR이 아니다. '
        '기종·개수·수집 조건이 얽혀 있어 원인을 확정하지 않는다.','']
    (PUBLIC/'ORDERED_BRANCH_CONDITIONAL_ANALYSIS_KO.md').write_text('\n'.join(lines))
    print(dict(status='COMPLETE',groups=len(table),gate=result['tanh_gate']))


if __name__=='__main__':main()
