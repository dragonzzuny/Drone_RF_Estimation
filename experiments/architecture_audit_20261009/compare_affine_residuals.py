"""Compare retained models using saved scores and cached reference DC fractions.

Reference-assisted affine optima are diagnostic only: no I/Q, weights, new
inference, fitting, or deployable gain selection. Keep all 420 mixtures/model.
"""
import math
from pathlib import Path
import statistics
import watch_epochs as watch


def main():
    root=watch.ROOT
    cache_path=root/'reports/2026-10-09/AFFINE_CALIBRATION_DIAGNOSIS.json'
    cache=watch.read(cache_path)
    source=root/'local/architecture_native_screen_20261009_v1/unet_mean/VALIDATION_002.json'
    if watch.digest(source)!=cache['validation_sha256']['2']:
        raise ValueError('Cached reference fractions are for a different validation')
    _,identities=watch.validate(source,None)
    fractions={r['index']:r for r in cache['rows'] if r['epoch']==2}
    repeated={r['index']:r for r in cache['rows'] if r['epoch']==3}
    expected=[r['index'] for r in identities if r['count']>1]
    if sorted(fractions)!=expected or sorted(repeated)!=expected or len(expected)!=420:
        raise ValueError('Missing diagnostic reference rows')
    for index,row in fractions.items():
        template=identities[index]
        if any(row[k]!=template[k] for k in ('count','categories')):
            raise ValueError('Cached reference identities differ')
        if row['reference_variance_fraction']!=repeated[index]['reference_variance_fraction']:
            raise ValueError('Repeated cached reference fractions differ')
        p=template['reference_power']; gap=10*math.log10(max(p)/min(p))
        if not math.isclose(gap,row['local_power_gap_db'],abs_tol=1e-9):raise ValueError('Reference powers differ')
    paths={
        'retained_unet_e2_once':root/'local/native_frequency_20261009_v1/phase/native_selected/BASELINE.json',
        'retained_unet_e2_four_phase':root/'local/native_frequency_20261009_v1/phase/native_selected/FOUR_PHASE.json',
        'matched_parent_unet_e4':root/'local/architecture_native_screen_20261009_v1/unet_mean/VALIDATION_004.json',
        'low_lr_original_e1':root/'local/low_lr_unet_comparison_20261009_v1/original/VALIDATION_001.json',
        'low_lr_log1p_e1':root/'local/low_lr_unet_comparison_20261009_v1/log1p_nmse/VALIDATION_001.json'}
    summaries=[];all_rows=[]
    for label,path in paths.items():
        value=watch.read(path);rows=sorted(value['rows'],key=lambda r:r['index'])
        if len(rows)!=630:raise ValueError('Missing model rows')
        actual=[{k:r[k] for k in identities[0]} for r in rows]
        if actual!=identities:raise ValueError('Models did not use identical reference mixtures')
        measured=[]
        for count in (1,2,3):
            group=[r for r in rows if r['count']==count]
            stored=next(g for g in value['by_count'] if g['count']==count)
            for key,field in [('mean_nmse','nmse'),('mean_si_sdr','si_sdr')]:
                vals=[x for r in group for x in r[field]]
                if any(x is None for x in vals):raise ValueError('Undefined source score needs explicit handling')
                if not watch.close(statistics.mean(vals),stored[key]):raise ValueError('Saved model aggregate differs')
        for row in rows:
            if row['count']==1:continue
            for j,(nmse,si,fraction) in enumerate(zip(row['nmse'],row['si_sdr'],fractions[row['index']]['reference_variance_fraction'])):
                if not math.isfinite(si) or not 0<=fraction<=1+1e-12:raise ValueError('Invalid affine inputs')
                optimum=fraction/(1+10**(si/10))
                if optimum>nmse+1e-7:raise ValueError('Affine optimum exceeds raw error')
                measured.append(dict(model=label,index=row['index'],count=row['count'],component=j,
                    category=row['categories'][j],weakest=j==row['weakest_index'],
                    raw_nmse=nmse,oracle_affine_nmse=optimum,si_sdr=si))
        if len(measured)!=1050:raise ValueError('Missing source components')
        for count in (2,3):
            for group in ('all','weakest'):
                subset=[r for r in measured if r['count']==count and (group=='all' or r['weakest'])]
                raw=statistics.mean(r['raw_nmse'] for r in subset)
                best=statistics.mean(r['oracle_affine_nmse'] for r in subset)
                summaries.append(dict(model=label,count=count,group=group,components=len(subset),
                    raw_nmse=raw,oracle_affine_nmse=best,removable_nmse=raw-best,
                    removable_fraction_of_aggregate_error=(raw-best)/raw,
                    mean_si_sdr=statistics.mean(r['si_sdr'] for r in subset)))
        all_rows.extend(measured)
    result=dict(status='COMPLETE',source_sha256=watch.digest(Path(__file__)),
        cached_reference_fraction_sha256=watch.digest(cache_path),reference_identity_sha256=watch.digest(source),
        validation_sha256={label:watch.digest(path) for label,path in paths.items()},
        summary=summaries,components_per_model=1050,mixtures_per_model=420,
        waveform_reads=0,new_predictions=0,model_updates=0,requires_reference_information=True,
        deployable=False,bound_on_other_separators=False,heldout_read=False,
        note='Different training histories; model comparison is descriptive, not matched architecture evidence. Affine gains and DC are constant per entire crop, not time-varying.')
    public=root/'reports/2026-10-09/CURRENT_AFFINE_RESIDUALS'
    watch.write(public.with_suffix('.json'),result)
    watch.write(root/'local/current_affine_residuals_20261009_v1/COMPONENTS.json',dict(result,rows=all_rows))
    lines=['# 보존 기준과 현재 모델: 배율·상수 보정으로 남는 파형 오차','',
        '모델별 같은 개발420혼합·1050성분의 저장 수치와 이미 계산한 정답 분산/전력 비를 사용했다. '
        '원시I/Q·체크포인트를 읽거나 새 추론·학습을 하지 않았다. 모든 혼합을 포함했다.','',
        '정답을 보고 한 구간 전체에 가장 유리한 복소 배율과 복소 상수를 적용했을 때 최소NMSE는 '
        '`(정답 분산/정답 전력)/(1+10^(복소 SI-SDR/10))`이다. '
        '원래 캐시의 독립 최소제곱 검산을 재사용하며, 모델 간 모든 정답 식별자/전력과 캐시 대응을 검사했다.','',
        '**정답 이용 진단이며 실제 보정기가 아니다.** 원인별 분해나 다른 분리기의 성능 한계도 아니다. '
        '상수 크기만 아니라 상수 위상·DC도 허용한다. 학습 이력이 다른 모델의 구조 효과를 비교하는 표가 아니다.','',
        '| 모델 | 혼합 내 성분 수 | 집계 | 평가 성분 수 | 현재 NMSE ↓ | 정답 이용 최적 배율+상수 NMSE ↓ | 제거 가능 오차 비중 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['model']}|{r['count']}|{r['group']}|{r['components']}|{r['raw_nmse']:.6f}|"
            f"{r['oracle_affine_nmse']:.6f}|{r['removable_fraction_of_aggregate_error']:.2%}|")
    lines+=['','제거 가능 비중은 평균오차의 차이/평균원오차이며 사례별 비율의 평균이 아니다. '
        '남는 오차를 특정 누출·변조·잡음 원인으로 단정하지 않는다. 새 기종 확인 기록은 읽지 않았다.','']
    watch.write(public.with_suffix('.md'),'\n'.join(lines))
    print({'status':'COMPLETE','summary':summaries})


if __name__=='__main__':main()
