"""Evidence-linked progress/final report; no waveform reads or model selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
from update_projection import solve as project_actual_update

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/2026-10-10'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def summarize(rows):
    out=[]
    for n in (1,2,3):
        r=[v for v in rows if v['count']==n];assert len(r)==210
        out.append(dict(count=n,cases=len(r),mean_nmse=statistics.mean(x for v in r for x in v['nmse']),
            mean_si_sdr=statistics.mean(x for v in r for x in v['si_sdr']),
            weakest_nmse=statistics.mean(v['nmse'][v['weakest_index']] for v in r),
            count_accuracy=statistics.mean(v['predicted_count']==n for v in r)))
    return out


def gradient_diagnostic(root):
    saved=read(root/'GRADIENT_UPDATES.json');rows=saved['updates'];gram=np.array([r['gram'] for r in rows])
    norm=np.sqrt(np.diagonal(gram,axis1=1,axis2=2));den=norm[:,:,None]*norm[:,None,:]
    cosine=np.divide(gram,den,out=np.zeros_like(gram),where=den>0)
    dots=np.array([r['original_gradient_dot_actual_delta'] for r in rows]);shadow=[]
    for row in rows:
        a=np.array(row['gram']);q=np.array(row['original_gradient_dot_actual_delta'])
        result=project_actual_update(a[1:,1:],q[1:])
        ratio=math.sqrt(result['correction_squared_norm'])/max(row['actual_adamw_delta_norm'],1e-30)
        shadow.append(dict(update=row['update'],**result,relative_correction_norm=ratio))
    result=dict(batches=len(rows),negative_gram_batches=(gram<0).sum(0).tolist(),mean_cosine=cosine.mean(0).tolist(),
        positive_gradient_dot_actual_delta=(dots>0).sum(0).tolist(),
        relative_gradient_change_median=statistics.median(r['change_norm']/max(r['ordinary_norm'],1e-30) for r in rows),
        shadow_actual_update_projection=shadow,
        shadow_batches_requiring_correction=sum(bool(r['active_constraints']) for r in shadow),
        shadow_relative_correction_max=max(r['relative_correction_norm'] for r in shadow),
        shadow_note='Counterfactual first-order geometry at recorded iterates only; not applied, no future trajectory or validation prediction')
    if 'waveform_guard' in rows[0]:
        result['waveform_guard_applied']=True
        result['waveform_guard_corrected_batches']=sum(bool(r['waveform_guard']['projection']['active_constraints']) for r in rows)
        result['waveform_guard_constraint_violations']=sum(any(q>t for q,t in zip(r['waveform_guard']['protected_dot_after_fp32'],r['waveform_guard']['fp32_constraint_tolerance'])) for r in rows)
    return result


def run():
    baseline=ROOT/'local/native_frequency_20261009_v1/phase/native_selected/BASELINE.json'
    control=ROOT/'local/source_interaction_comparison_20261010_v1/retained_unet/VALIDATION_001.json'
    refs=[('parent',baseline,0),('original_loss_e1',control,75)]
    arms=[];status=[];dependencies={str(p.relative_to(ROOT)):digest(p) for _,p,_ in refs}
    for name,path,updates in refs:
        arms.append(dict(arm=name,updates=updates,by_count=summarize(read(path)['rows'])))
    studies=[('pcgrad','count_pcgrad','COUNT_PCGRAD'),('cagrad','count_cagrad','COUNT_CAGRAD'),
             ('wave_guard','wave_update_guard','WAVE_UPDATE_GUARD'),('tf_axis','tf_axis','TF_AXIS')]
    for key,folder,prefix in studies:
        root=ROOT/f'local/{folder}_20261010_v1'
        if not (root/'STATE.json').exists():continue
        state=read(root/'STATE.json')
        current=dict(arm=key,status=state['status'],updates=state.get('updates',0),state_time=state['time'])
        if (root/'COMPLETE.json').exists():
            complete=read(root/'COMPLETE.json');validation=root/'VALIDATION_001.json'
            dependencies[str(validation.relative_to(ROOT))]=digest(validation)
            audit_path=PUBLIC/f'{prefix}_FINAL_AUDIT.json'
            audit=read(audit_path) if audit_path.exists() else None
            if audit:
                assert audit['status']=='PASS' and audit['complete_sha256']==digest(root/'COMPLETE.json')
                dependencies[str(audit_path.relative_to(ROOT))]=digest(audit_path)
            arm=dict(arm=key,updates=75,by_count=summarize(read(validation)['rows']),
                selected_epoch=complete['selected']['epoch'],criterion_met=complete['criterion_met'],
                audited=audit is not None,train_seconds=complete['train_seconds'],gradient=gradient_diagnostic(root))
            for a,b in zip(arm['by_count'],complete['actual']['by_count']):
                for k in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert math.isclose(a[k],b[k],abs_tol=1e-12)
            arms.append(arm);current.update(selected_epoch=arm['selected_epoch'],audited=arm['audited'])
        status.append(current)
    partition=PUBLIC/'COUNT_LOSS_PARTITION.json';partition_summary=None
    if partition.exists():
        data=read(partition);assert data['status']=='COMPLETE'
        dependencies[str(partition.relative_to(ROOT))]=digest(partition)
        parts=data['blocks'];total=sum(np.array(g['gram']) for g in parts if g['group']!='all')
        assert np.allclose(total,next(g['gram'] for g in parts if g['group']=='all'))
        for block,derived in zip(parts,data['derived']):
            a=np.array(block['gram']);assert np.allclose(a,a.T) and np.linalg.eigvalsh(a).min()>-1e-7
            assert np.allclose(a[:3,:3]+a[3:,3:]+a[:3,3:]+a[3:,:3],derived['main']['gram'])
        partition_summary=dict(indices=[r['index'] for r in data['rows']],derived=data['derived'],
            additive_gram_identity_checked=True,gradients_independently_recomputed=False)
    done=all((x['status']=='COMPLETE' and x.get('audited')) or x['status'].startswith('SKIPPED_') for x in status)
    result=dict(status='COMPLETE_AUDITED' if done else 'IN_PROGRESS',arms=arms,run_status=status,
        partition=partition_summary,dependencies=dependencies,heldout_read=False,independent_test=False,
        time=time.time(),reporter_sha256=digest(Path(__file__)))
    write(PUBLIC/'COUNT_OPTIMIZATION_REPORT.json',result)
    lines=['# 개수별 기울기 조정: 실제 실행 및 결과','',
        'RFUAV 동일 native RF 자료·같은 부모·원 규모 U-Net·원 손실·seed 0. 기본 32,142,859개, TF축 후보 37,406,475개 파라미터다. '
        '각 후보 2,400예제/75업데이트, 원 손실 대조 e1을 재사용한다. 부모의 이전 학습 이력은 별도다. '
        'DEV는 다섯 기록 묶음의 반복 630창이며 독립 시험이 아니다.','',
        '| 방법 | 추가 업데이트 | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ |','|---|---:|---|---|---|']
    for arm in arms:
        a,b=arm['by_count'][1:]
        lines.append(f"|{arm['arm']}|{arm['updates']}|{a['mean_nmse']:.6f}/{b['mean_nmse']:.6f}|"
                     f"{a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f}|{a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f}|")
    lines+=['','단일 성분과 개수 예측값은 연결된 JSON에 모두 보존한다. 표는 실제 e1이며 e0 선택으로 실패를 숨기지 않는다.','']
    for state in status:
        lines.append(f"- {state['arm']}: {state['status']}, {state['updates']}/75업데이트, 감사 {state.get('audited',False)}.")
    for arm in arms[2:]:
        g=arm['gradient'];negative=g['negative_gram_batches']
        lines+=['',f"{arm['arm']}는 선택 e{arm['selected_epoch']}, 진전 기준 {arm['criterion_met']}, 학습 {arm['train_seconds']/60:.2f}분이다. "
            f"원 기울기 충돌은 1–2 {negative[0][1]}/75, 1–3 {negative[0][2]}/75, 2–3 {negative[1][2]}/75배치였다. "
            f"실제 AdamW 변위와 각 집단 기울기 내적이 양수인 횟수는 {g['positive_gradient_dot_actual_delta']}다. "
            '이는 현재 배치의 일차 근사이며 실제 유한 이동의 손실 증가나 DEV 악화 원인 확정이 아니다.',
            f"기록된 변위에 두·세 성분 비증가 반공간 투영을 가정하면 {g['shadow_batches_requiring_correction']}/75배치에서 수정이 필요했고 "
            f"수정 norm/원 이동 norm의 최댓값은 {g['shadow_relative_correction_max']:.4f}다. 이 계산은 실제 학습에 적용하지 않았다."]
    for arm in arms[2:]:
        g=arm['gradient']
        if g.get('waveform_guard_applied'):
            lines+=['',f"실제 파형 보호 보정은 {g['waveform_guard_corrected_batches']}/75배치에서 적용됐고 FP32 허용오차를 넘는 제약 위반은 {g['waveform_guard_constraint_violations']}회다. 위 가정 계산은 전체 원 손실 기준이며 실제 보호는 분류 CE를 제외한 파형 손실 기준이다."]
    if partition_summary:
        lines+=['','## 파형/개수 분류 손실 분해','',
            '이전에 고정한 TRAIN6의 개수별 두 사례 평균을 사용했다. GPU 배치와 표본이 다르며 전체 자료의 대표 추정이 아니다.','',
            '| 파라미터 부분 | 파형 손실 cosine 1–2 / 1–3 / 2–3 | 원 전체 손실 cosine 1–2 / 1–3 / 2–3 |',
            '|---|---|---|']
        for group in partition_summary['derived']:
            vals=[]
            for name in ('waveform','main'):
                c=group[name]['cosine'];zero=group[name]['zero_norm']
                vals.append('/'.join('정의 안 됨' if zero[i] or zero[j] else f'{c[i][j]:.3f}' for i,j in ((0,1),(0,2),(1,2))))
            lines.append(f"|{group['group']}|{vals[0]}|{vals[1]}|")
    lines+=['','[PCGrad 사전 계획](COUNT_PCGRAD_PLAN_KO.md) · [CAGrad 사전 계획](COUNT_CAGRAD_PLAN_KO.md) · [수치·개수별 전체 결과](COUNT_OPTIMIZATION_REPORT.json)',
            '','기존 최선은 공동 개선 기준을 충족한 새 후보가 검산되기 전까지 유지한다. Autel과 예약 확인 파일은 열지 않았다.','']
    (PUBLIC/'COUNT_OPTIMIZATION_REPORT_KO.md').write_text('\n'.join(lines))
    print(dict(status=result['status'],runs=status),flush=True)


if __name__=='__main__':run()
