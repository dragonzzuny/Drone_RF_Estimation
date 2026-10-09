"""Paired descriptive DEV strata; saved measurements only, no model selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(public):
    p=read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    paths=[('parent',Path(p['baseline'])),('original_loss_e1',Path(p['study'])/'retained_unet/VALIDATION_001.json')]
    for method,folder,prefix in [('pcgrad','count_pcgrad_20261010_v1','COUNT_PCGRAD'),
        ('cagrad','count_cagrad_20261010_v1','COUNT_CAGRAD'),('wave_guard','wave_update_guard_20261010_v1','WAVE_UPDATE_GUARD'),
        ('tf_axis','tf_axis_20261010_v1','TF_AXIS')]:
        audit=public/f'{prefix}_FINAL_AUDIT.json'
        if not audit.exists():continue
        a=read(audit);assert a['status']=='PASS'
        root=ROOT/'local'/folder;assert a['complete_sha256']==digest(root/'COMPLETE.json')
        paths.append((method,root/'VALIDATION_001.json'))
    parent=read(paths[0][1])['rows'];assert len(parent)==630
    ids={r['index']:r for r in parent};assert len(ids)==630
    summaries=[];hashes={}
    for method,path in paths:
        values=read(path)['rows'];assert len(values)==630 and {r['index'] for r in values}==set(ids)
        hashes[method]=digest(path);groups={}
        for row in values:
            original=ids[row['index']]
            for key in ('count','categories','pack_ids','nominal_levels_db','reference_power'):
                assert row[key]==original[key],(method,row['index'],key)
            n=row['count'];assert len(row['nmse'])==len(row['si_sdr'])==n
            if n==1:continue
            power=row['reference_power'];gap=10*math.log10(max(power)/min(power))
            level='0-10dB' if gap<10 else '10-20dB' if gap<=20 else '>20dB'
            for j,category in enumerate(row['categories']):
                assert math.isfinite(row['nmse'][j]) and math.isfinite(row['si_sdr'][j])
                record=dict(nmse=row['nmse'][j],si_sdr=row['si_sdr'][j],
                    nmse_delta=row['nmse'][j]-original['nmse'][j],si_sdr_delta=row['si_sdr'][j]-original['si_sdr'][j],
                    weak=j==row['weakest_index'],index=row['index'])
                for typ,label in [('category',category),('power_gap',level)]:groups.setdefault((n,typ,label),[]).append(record)
        for (count,typ,label),group in sorted(groups.items()):
            weak=[r for r in group if r['weak']]
            summaries.append(dict(method=method,count=count,stratum=typ,label=label,source_cases=len(group),
                mixture_cases=len({r['index'] for r in group}),weak_cases=len(weak),
                mean_nmse=statistics.mean(r['nmse'] for r in group),mean_si_sdr=statistics.mean(r['si_sdr'] for r in group),
                mean_nmse_delta=statistics.mean(r['nmse_delta'] for r in group),mean_si_sdr_delta=statistics.mean(r['si_sdr_delta'] for r in group),
                weakest_nmse=statistics.mean(r['nmse'] for r in weak) if weak else None))
    result=dict(status='COMPLETE',all_630_case_identities_checked=True,validation_sha256=hashes,
        reporter_sha256=digest(Path(__file__)),summaries=summaries,recorded_iq_reads=0,heldout_read=False,
        limitation='Post-hoc descriptive strata on repeatedly used DEV; no independent test, confidence interval, or subgroup-based checkpoint selection')
    (public/'COUNT_CONDITION_REPORT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 기종·국소 전력차에 따른 실제 e1 결과','',
        '동일한630개 DEV 혼합의 실제 e1끼리 비교했다. 기종은 평가 정답의 라벨이며 추론 입력이 아니다. '
        '전력차는 창 안 정답 성분의 최대/최소 전력비다. 630창은 독립 기록630개가 아니며 개발 기록은5묶음이다. '
        '사후 기술 집계로, 유리한 하위 조건을 골라 모델을 선택하거나 논문 주장으로 승격하지 않는다.']
    for typ,title in [('category','정답 기종별'),('power_gap','국소 최대/최소 전력차별')]:
        lines+=['',title,'',
            '| 방법 | 성분 수 | 조건 | 성분 사례 | NMSE ↓ | 복소 SI-SDR ↑ dB | 부모 대비 NMSE 차이 | 부모 대비 SI-SDR 차이 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['stratum']!=typ:continue
            lines.append(f"|{r['method']}|{r['count']}|{r['label']}|{r['source_cases']}|{r['mean_nmse']:.6f}|"
                f"{r['mean_si_sdr']:.3f}|{r['mean_nmse_delta']:+.6f}|{r['mean_si_sdr_delta']:+.3f}|")
    lines+=['','모든 집계의 성분 수·약신호 사례 수와 약신호 평균은 [JSON](COUNT_CONDITION_REPORT.json)에 보존했다. '
        '약신호 사례가 없는 조건은 평균을0으로 대체하지 않았다. [전체 평균](COUNT_OPTIMIZATION_REPORT_KO.md).']
    (public/'COUNT_CONDITION_REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    return dict(status='COMPLETE',methods=[name for name,_ in paths],strata=len(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);a=p.parse_args();print(run(a.public))
