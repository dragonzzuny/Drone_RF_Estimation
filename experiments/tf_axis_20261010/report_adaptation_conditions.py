"""Post-hoc complete strata of saved adaptation results; no new inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(public):
    original=read(ROOT/'local/tf_axis_adaptation_20261010_v1/PROTOCOL.json')['original_protocol']
    paths={'parent':Path(original['baseline']),
        'original_loss_e1':Path(original['study'])/'retained_unet/VALIDATION_001.json',
        'joint_adapter_lr':ROOT/'local/tf_axis_adaptation_20261010_v1/joint_adapter_lr/VALIDATION_001.json',
        'frozen_backbone':ROOT/'local/tf_axis_adaptation_20261010_v1/frozen_backbone/VALIDATION_001.json'}
    baseline={r['index']:r for r in read(paths['parent'])['rows']};assert len(baseline)==630
    summaries=[]
    for method,path in paths.items():
        cases=read(path)['rows'];assert len(cases)==630 and {q['index'] for q in cases}==set(baseline)
        rows=[]
        for case in cases:
            parent=baseline[case['index']]
            for key in ('count','categories','pack_ids','nominal_levels_db','reference_power','weakest_index'):
                assert case[key]==parent[key],(method,key,case['index'])
            n=case['count']
            if n==1:continue
            power=case['reference_power'];order=sorted(range(n),key=lambda i:(-power[i],i))
            for j in range(n):
                sir=10*math.log10(power[j]/sum(v for k,v in enumerate(power) if k!=j))
                bucket=next(name for lower,upper,name in [(-math.inf,-20,'<-20dB'),(-20,-10,'[-20,-10)dB'),
                    (-10,0,'[-10,0)dB'),(0,10,'[0,10)dB'),(10,20,'[10,20)dB'),(20,math.inf,'>=20dB')] if lower<=sir<upper)
                nmse=case['nmse'][j];si=case['si_sdr'][j]
                assert math.isfinite(nmse) and math.isfinite(si)
                rows.append(dict(index=case['index'],count=n,category=case['categories'][j],
                    rank=order.index(j)+1,sir_bucket=bucket,si=si,nmse=nmse,
                    delta_nmse=nmse-parent['nmse'][j],delta_si=si-parent['si_sdr'][j]))
        assert len(rows)==1050
        for n in (2,3):
            for field in ('rank','category','sir_bucket'):
                for value in sorted({r[field] for r in rows if r['count']==n}):
                    selected=[r for r in rows if r['count']==n and r[field]==value]
                    summaries.append(dict(method=method,count=n,stratum=field,label=value,source_cases=len(selected),
                        mixture_cases=len({r['index'] for r in selected}),
                        mean_nmse=statistics.mean(r['nmse'] for r in selected),
                        mean_si_sdr=statistics.mean(r['si'] for r in selected),
                        nmse_delta=statistics.mean(r['delta_nmse'] for r in selected),
                        si_sdr_delta=statistics.mean(r['delta_si'] for r in selected),
                        improved_both_sources=sum(r['delta_nmse']<0 and r['delta_si']>0 for r in selected)))
    result=dict(status='COMPLETE_CHECKED',summaries=summaries,all_630_identities_checked=True,
        validation_sha256={k:sha(v) for k,v in paths.items()},reporter_sha256=sha(Path(__file__)),
        iq_reads=0,new_inference=0,heldout_read=False,
        sir_definition='Reference component power / sum of OTHER reference powers; not coherent interference power',
        limitation='All strata retained; descriptive reused DEV from5 recording groups, no subgroup selection or independent significance test')
    (public/'TF_AXIS_ADAPTATION_CONDITIONS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 본체 고정 후 변화: 기종·전력 순위·국소 SIR별','',
        '저장된 부모·원 손실 대조·두 축 공동/고정군의630개 개발 행을 다시 집계했다. '
        '정답 기종·원기록·전력·혼합ID가 모두 같음을 검사했으며 새 파형 읽기나 추론은 없다. '
        '순위1이 가장 강하다. SIR은 각 정답 전력/다른 정답 전력의 합이다. 복소 교차항이 있는 실제 간섭 파형 전력과는 구별한다.','',
        '| 본체 고정군 조건 | 성분 수 | 사례 수 | NMSE ↓ | 복소SI-SDR ↑ dB | 부모 대비 NMSE | 부모 대비 SI-SDR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['method']=='frozen_backbone':
            label=f"{s['stratum']}={s['label']}"
            lines.append(f"|{label}|{s['count']}|{s['source_cases']}|{s['mean_nmse']:.6f}|{s['mean_si_sdr']:.3f}|{s['nmse_delta']:+.6f}|{s['si_sdr_delta']:+.3f}|")
    lines+=['','모든 군·조건의 사례 수와 두 지표가 모두 좋아진 성분 수는JSON에 보존했다. '
        '관측된 하위 조건 차이를 기종 특성의 인과효과로 확정하거나 유리한 조건만 골라 채택 기준을 바꾸지 않는다. '
        '세 성분 최약 NMSE가 감소해도 전체 SI-SDR이 악화할 수 있어 두 지표를 함께 판단한다. '
        '630창은 독립 기록630개가 아니고 개발 원기록은5묶음이다.',
        '', '[모든 군과 조건](TF_AXIS_ADAPTATION_CONDITIONS.json) · [원 결과와최종 검산](TF_AXIS_ADAPTATION_KO.md)']
    (public/'TF_AXIS_ADAPTATION_CONDITIONS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([s for s in summaries if s['method']=='frozen_backbone' and s['count']==3 and s['stratum']=='rank'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);run(p.parse_args().public.resolve())
