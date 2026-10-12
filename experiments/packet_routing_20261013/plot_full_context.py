"""Provisional diagnostic figure from all five fixed full-context cases.

No waveform data are changed. Local reference-based reassignment is a scoring
diagnostic, not a deployable output or an NMSE lower bound (PIT uses its own
combined objective). All actual per-window values are supplied as CSV.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(parent,candidate,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    base=read(parent);assert base['status']=='VERIFIED'
    arms=[('Parent99',base['cases'],'#0072B2','o')]
    if candidate is not None:
        new=read(candidate);assert new['status']=='VERIFIED'
        arms.append((new['architecture']+' e3',new['full_context'],'#D55E00','s'))
    assert all([v['index'] for v in cases]==[0,7,14,15,20] for _,cases,_,_ in arms)
    outputs=[prefix.with_suffix(ext) for ext in ('.pdf','.svg','.png','.csv','.json','.md')]
    assert not any(p.exists() for p in outputs),'Use a new output prefix; original figures are preserved'
    prefix.parent.mkdir(parents=True,exist_ok=True);table=[];errors=[];activity=[]
    for label,cases,_,_ in arms:
        for case in cases:
            cats=case['categories'];global_power=case['metrics']['reference_power']
            for window in case['local_scoring']:
                assert window['zero_power_sources']==0,'Zero-power scoring needs explicit display treatment'
                assert len(window['fixed_global_nmse'])==len(cats)
                for i,cat in enumerate(cats):
                    values=dict(model=label,case_index=case['index'],category=cat,
                        start_ms=window['start']/1e5,stop_ms=window['stop']/1e5,
                        fixed_global_nmse=window['fixed_global_nmse'][i],local_pit_nmse=window['local_pit_nmse'][i],
                        local_reference_power=window['reference_power'][i],whole_reference_power=global_power[i],
                        relative_activity=window['reference_power'][i]/global_power[i])
                    assert all(np.isfinite(values[k]) for k in ('fixed_global_nmse','local_pit_nmse','relative_activity'))
                    assert min(values['fixed_global_nmse'],values['local_pit_nmse'],values['relative_activity'])>=0
                    table.append(values);activity.append(values['relative_activity'])
                errors.extend([statistics.mean(window['fixed_global_nmse']),statistics.mean(window['local_pit_nmse'])])
    with outputs[3].open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    style={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(5,2,figsize=(12,11.5),layout='constrained',sharex=True,sharey='col')
        for row,base_case in enumerate(base['cases']):
            left,right=axes[row];cats=base_case['categories'];title=' + '.join(c.replace('DJI ','') for c in cats)
            left.set_title(f"{chr(65+row)}  {title}",loc='left',fontweight='bold')
            for label,cases,color,marker in arms:
                case=cases[row];assert case['index']==base_case['index'] and case['categories']==cats
                assert case['metrics']['reference_power']==base_case['metrics']['reference_power']
                windows=case['local_scoring'];times=[(v['start']+v['stop'])/2e5 for v in windows]
                g=[statistics.mean(v['fixed_global_nmse']) for v in windows]
                local=[statistics.mean(v['local_pit_nmse']) for v in windows]
                left.plot(times,g,color=color,marker=marker,markersize=2.8,lw=1.3,label=label)
                left.plot(times,local,color=color,linestyle='--',lw=1.1,alpha=.95)
            left.axhline(1,color='#555555',ls=':',lw=.8)
            left.set_yscale('symlog',linthresh=.01,base=10);left.set_ylim(0,max(1.,max(errors))*1.08)
            left.set_ylabel('Mean source NMSE')
            windows=base_case['local_scoring'];times=[(v['start']+v['stop'])/2e5 for v in windows]
            for j,(cat,color,marker,linestyle) in enumerate(zip(cats,['#0072B2','#D55E00','#009E73'],['o','s','^'],['-','--','-.'])):
                y=[v['reference_power'][j]/base_case['metrics']['reference_power'][j] for v in windows]
                right.plot(times,y,color=color,marker=marker,markersize=2.6,lw=1,ls=linestyle,label=cat.replace('DJI ',''))
            right.set_yscale('symlog',linthresh=.01,base=10);right.set_ylim(0,max(activity)*1.08)
            right.set_ylabel('Local / whole reference power');right.legend(loc='upper right',fontsize=7,framealpha=.92)
            for ax in (left,right):
                ax.grid(True,axis='y',which='major',alpha=.2);ax.set_xlim(0,20.8896)
                ax.spines[['top','right']].set_visible(False)
        for ax in axes[-1]:ax.set_xlabel('Time in the original observation (ms)')
        legend=[Line2D([0],[0],color=color,marker=marker,lw=1.3,label=label) for label,_,color,marker in arms]
        legend += [Line2D([0],[0],color='#333333',ls='-',label='Whole-record source mapping'),
                   Line2D([0],[0],color='#333333',ls='--',label='Local reference mapping: diagnostic only')]
        fig.legend(handles=legend,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
        for ext in ('.pdf','.svg','.png'):fig.savefig(prefix.with_suffix(ext),dpi=300,facecolor='white',transparent=False)
        plt.close(fig)
    caption='''# 전체 구간 출력 대응 진단 그림

미리 고정한 동일 명목 전력의 다섯 조합을 모두 표시한다. 왼쪽은 각 짧은 평가 구간에서
성분별 NMSE의 산술평균이다. 실선은 전체20.8896ms에 하나의 정답 대응을 적용한 점수이고,
점선은 같은 출력에 짧은 구간별 정답 대응을 다시 적용한 **진단 점수**다. 점선을 실제 추론
결과로 제시하지 않는다. 대응은 기존 NMSE·coherence 등의 PIT 목적을 쓰므로 점선이
NMSE의 수학적 하한이라고 가정하지 않는다. 점 사이 연결은 경향을 보기 위한 것으로 보간·평활화하지 않았다.

오른쪽은 각 원성분의 국소 전력을 그 성분의 전체 구간 전력으로 나눈 값이다. 수신 잡음도
포함된 기록 전력이며 신호 활동의 대리 지표다. 0.01 이하는 선형, 그 위는 밑10인 symlog 축으로
모든 값을 표시한다. 열별 모든 패널의 축 범위는 같다. 왼쪽 회색 수평선의1은 오차 전력이
정답 전력과 같은 기준이며 분리 성공/실패 임계값이 아니다.

각 구간의 전력으로 NMSE를 정규화하므로 왼쪽 시간평균은 전체 구간 NMSE와 다를 수 있다.
신뢰구간을 그리지 않았다. 다섯 조합은 같은 원녹음 시간분할 개발 사례이며 독립 반복이 아니다.
짧은 구간의 대응 변경에는 저활동 구간의 모호함도 포함된다. 원 수치와 모든 정규화 값은
같은 이름의CSV에 있다. 파형을 변경하거나 좋은 구간만 선택하지 않았다.
그림의 기체명은 평가용 정답 표지이며 분리 모델의 기종 식별 출력이 아니다.

일반 연구 검토용 크기의 임시 그림이다. 특정 학술지의 제출 규격 준수를 주장하지 않는다.

그림 작성 절차에 Scientific Visualization 스킬을 사용했다. 도구 출처:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
서지정보는2026-10-13 arXiv 최신 기록(v2)에서 확인했다.
'''
    outputs[5].write_text(caption)
    manifest=dict(status='EXPORTED_PENDING_VISUAL_INSPECTION',input_sha256={str(p):sha(p) for p in (parent,candidate) if p is not None},
        cases=5,selection='Every predeclared equal-power full context',size_inches=[12,11.5],png_dpi=300,
        yscale='symlog base10 linear threshold0.01; common limits per column; zeros retained',
        transformations='Mean source NMSE per existing nonoverlapping window; activity divided by whole reference power; no smoothing',
        uncertainty='None; same-record dependent development cases',matplotlib=matplotlib.__version__,
        source_sha256=sha(Path(__file__)),exports={p.name:sha(p) for p in outputs if p.suffix!='.json'},
        alt_text='Five-row figure of source reconstruction error and local reference power over20.89ms; whole-record versus local reference-based source matching.')
    outputs[4].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--candidate',type=Path)
    p.add_argument('--output-prefix',required=True,type=Path);a=p.parse_args();main(a.parent,a.candidate,a.output_prefix)
