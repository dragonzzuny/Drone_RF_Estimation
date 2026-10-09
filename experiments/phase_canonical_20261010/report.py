"""Complete audited one-epoch comparison and reproducible vector figure."""
import argparse
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(study,control,audit_path,output):
    audit=read(audit_path);result=read(study/'COMPLETE.json')
    assert audit['status']=='PASS' and audit['complete_sha256']==sha(study/'COMPLETE.json')
    assert audit['study_protocol_sha256']==sha(study/'PROTOCOL.json')
    retained=[read(control/'retained_unet'/f'VALIDATION_{i:03d}.json') for i in (0,1)]
    canonical=[result['initial'],result['actual']]
    public=dict(status='COMPLETE_AUDITED',protocol_sha256=sha(study/'PROTOCOL.json'),
        audit_sha256=sha(audit_path),criterion_met=result['criterion_met'],selected_epoch=result['selected']['epoch'],
        canonical=canonical,retained=[{k:r[k] for k in ('epoch','by_count','selection_nmse')} for r in retained],
        comparison=result['comparison'],updates_per_arm=75,seed=0,parameters_per_arm=32142859,
        validation_cases=630,validation_recording_groups=5,independent_test=False,heldout_read=False,
        source_sha256=sha(Path(__file__)))
    output.with_suffix('.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    lines=['# 기준 위상 정렬: 학습 전후 완료 비교','','RFUAV 원 대역·수신 중심 간격·기록 분할을 유지했다. 같은 부모 가중치·전체32,142,859파라미터·원 손실·새AdamW1e-5·seed0·2400혼합·75업데이트이며, 개발630혼합을 한 번씩 추론했다. 기존 U-Net 대조는 완료·검산한 같은 예산 e1을 재사용했다. 위상 처리로 후보의 시작 출력은 대조와 다르다.','','| 모델 | 추가 epoch | NMSE 2/3 ↓ | 복소 SI-SDR 2/3 ↑ dB | 약신호 NMSE 2/3 ↓ |','|---|---:|---|---|---|']
    for name,history in [('기존 U-Net',retained),('기준 위상 U-Net',canonical)]:
        for epoch,value in enumerate(history):
            a,b=value['by_count'][1:]
            lines.append(f"| {name} | {epoch} | {a['mean_nmse']:.6f}/{b['mean_nmse']:.6f} | {a['mean_si_sdr']:.3f}/{b['mean_si_sdr']:.3f} | {a['weakest_nmse']:.6f}/{b['weakest_nmse']:.6f} |")
    lines+=['',f"후보의 e0 포함 평균 NMSE 규칙 선택: 추가e{public['selected_epoch']}. 부모 및 같은 예산 대조에 대한 네 평균 지표와 약신호 기준 충족: **{result['criterion_met']}**.",'','## 1성분과 개수 추정','','| 모델/시점 | 1성분 NMSE | 1성분 SI-SDR dB | 개수 정확도 1/2/3 |','|---|---:|---:|---|']
    for label,value in [('기존e0',retained[0]),('기존e1',retained[1]),('위상e0',canonical[0]),('위상e1',canonical[1])]:
        one=value['by_count'][0]
        counts='/'.join(f"{g['construction_count_accuracy']:.3f}" for g in value['by_count'])
        lines.append(f"| {label} | {one['mean_nmse']:.6f} | {one['mean_si_sdr']:.3f} | {counts} |")
    lines+=['','원기록8학습/5개발 묶음, 한 세 성분 기종 조합, 반복 개발 선택, 한 seed의 초기 대조다.630창은630개 독립 기록이 아니다. 한 epoch로 충분한 수렴이나 기체 일반화를 주장하지 않는다. 위상 동작의 정확성과 복원 성능 개선은 구분한다. 실패한 epoch도 포함했다.','',
        '[수학·CPU 검사](CANONICAL_PHASE_PLAN_KO.md) · [선행과 적용 범위](CANONICAL_PHASE_RELATED_WORK_KO.md) · [학습 전 정한 규약](CANONICAL_PHASE_TRAIN_PLAN_KO.md) · [최종 검산](CANONICAL_PHASE_TRAIN_AUDIT.json)','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7.2))
    for col,count in enumerate((2,3)):
        for row,(key,label) in enumerate([('mean_nmse','NMSE (lower is better)'),('mean_si_sdr','Complex SI-SDR (dB)\n(higher is better)')]):
            ax=axes[row,col]
            for name,hist,color,marker in [('Retained U-Net',retained,'#4477AA','o'),('Canonical-phase U-Net',canonical,'#CC6677','s')]:
                values=[next(g[key] for g in value['by_count'] if g['count']==count) for value in hist]
                ax.plot([0,1],values,label=name,color=color,marker=marker,linewidth=1.8,markersize=6)
            ax.set_xticks([0,1],['Before adaptation','After 75 updates'])
            ax.set_title(f'{count} recorded contributions');ax.set_ylabel(label)
            ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False)
            if key=='mean_nmse':ax.set_ylim(0,max(.8,ax.get_ylim()[1]))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
    fig.suptitle('Global phase canonicalization: matched one-epoch adaptation',fontsize=14,y=.98)
    fig.text(.5,.035,'630 development mixtures / 5 recording groups / seed 0 / one inference per mixture\nSame parent weights; phase processing changes the initial predictions. No independent test.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.95])
    fig.savefig(output.with_suffix('.pdf'));fig.savefig(output.with_suffix('.png'),dpi=220);plt.close(fig)
    print(dict(status='COMPLETE_AUDITED',criterion_met=result['criterion_met'],selected_epoch=public['selected_epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('study','control','audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.study.resolve(),a.control.resolve(),a.audit.resolve(),a.output.resolve())
