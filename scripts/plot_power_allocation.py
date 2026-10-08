"""Draw audited paired epoch means, without IQ access or reselection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main(source,output,complete):
    result=json.loads(source.read_text())
    if not result['all_rows_reaggregated'] or not result['same_validation_metadata_verified']:
        raise ValueError('Figure requires an audited report')
    if complete and result['status']!='COMPLETED':
        raise ValueError('Final figure requested before completion')
    common=result['common_completed_epoch']
    if common<1:
        raise ValueError('No paired epoch yet')
    rows=[]
    for arm,info in result['arms'].items():
        for metric in [info['initial']]+[r['metrics'] for r in info['epochs']]:
            for group in metric['by_count']:
                rows.append(dict(arm=arm,epoch=metric['epoch'],count=group['count'],cases=group['cases'],
                    mean_nmse=group['mean_nmse'],mean_complex_si_sdr_db=group['mean_si_sdr'],
                    weakest_nmse=group['weakest_nmse'],paired_budget=metric['epoch']<=common))
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf','.csv','.json','.md'):
        if output.with_suffix(suffix).exists():
            raise FileExistsError(output.with_suffix(suffix))
    with output.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    font=Path(subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True).strip())
    family=font_manager.FontProperties(fname=font).get_name()
    styles=dict(waveform_only=dict(color='#0072B2',marker='o',linestyle='-',label='기존 파형 손실'),
        waveform_allocation=dict(color='#D55E00',marker='s',linestyle='--',label='파형 + 전력 배분 손실'))
    shown=[r for r in rows if r['paired_budget'] and r['count']>1]
    with plt.rc_context({'font.family':family,'font.size':10,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(9,6.8),sharex=True,sharey='col')
        fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.78,hspace=.32,wspace=.25)
        for row_index,count in enumerate((2,3)):
            for column,(key,label) in enumerate((('mean_nmse','I/Q NMSE  ↓'),
                                                 ('mean_complex_si_sdr_db','복소 SI-SDR (dB)  ↑'))):
                ax=axes[row_index,column]
                for arm,style in styles.items():
                    values=[r for r in shown if r['arm']==arm and r['count']==count]
                    ax.plot([r['epoch'] for r in values],[r[key] for r in values],
                        linewidth=1.7,markersize=5,**style)
                ax.set_title(f'{count}개 기록 성분 · {label}',fontsize=11,loc='left')
                ax.set_xticks(range(common+1));ax.grid(axis='y',alpha=.18)
                ax.tick_params(axis='both',labelsize=9)
                if row_index==1:ax.set_xlabel('추가 학습 epoch (0 = 공통 초기 가중치)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.87),ncol=2,frameon=False)
        fig.suptitle('전력 배분 감독이 복원을 개선하는가?',y=.975,fontsize=17,fontweight='bold')
        fig.text(.5,.915,f"{'최종' if complete else '중간'} 비교 · 두 군 각 {common} epoch · 같은 검증 혼합·학습 예산",ha='center',fontsize=11)
        fig.text(.09,.09,'신호 수마다 210혼합의 성분 평균. 같은 열은 동일 축. 선은 측정 epoch를 연결하며 평활화하지 않음.',fontsize=8.5)
        fig.text(.09,.055,'seed 0의 반복 개발 검증이며 독립 반복 실험이 아님. 개수는 합성 기록 성분 수이며 물리 드론 대수 검증이 아님.',fontsize=8.5)
        fig.savefig(output.with_suffix('.png'),dpi=300,facecolor='white')
        fig.savefig(output.with_suffix('.pdf'),facecolor='white')
        limits=[[list(ax.get_ylim()) for ax in line] for line in axes]
        plt.close(fig)
    provenance=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_all_rows_audited=True,common_epochs=common,all_completed_aggregates_in_csv=True,
        plot_exclusion='not-yet-paired later epochs omitted only from paired trajectory; retained in CSV',
        transformation='unchanged linear-scale means; no smoothing; same limits within metric columns',
        uncertainty='not estimated: single seed, repeated dependent development mixtures',
        axis_limits=limits,font_path=str(font),matplotlib_version=matplotlib.__version__,
        size_inches=[9,6.8],png_dpi=300,publisher_requirements='unspecified; provisional research report figure',
        procedure_reference='Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065; v2 checked 2026-10-09')
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    text=['# 전력 배분 학습의 검증 궤적','',
        f'공통 완료 {common} epoch까지 두 손실의 2·3성분 NMSE와 복소 SI-SDR 평균을 그렸다.',
        '모든 완료 수치는 CSV에 있으며 그림에는 양쪽 모두 완료한 예산만 표시한다. 선택된 epoch만 보여주는 그림이 아니다.',
        '신호 수별 210혼합은 원기록을 반복 사용하므로 210개 독립 실험으로 해석하지 않는다. 오차막대·유의성은 추정하지 않았다.',
        'NMSE는 복소 배율 보정 없는 I/Q 오차다. 복소 SI-SDR은 상수 복소 배율을 허용하는 별도 분리 지표다.',
        '출판사·학술지 규격은 미지정이며 일반 연구 보고용 PDF/PNG다.',
        '대체 설명: 왼쪽 열은 두·세 신호의 NMSE, 오른쪽은 복소 SI-SDR이다. 파란 실선·원은 기존 손실, 주황 점선·네모는 보조 손실이다. 시점마다 같은 검증 자료를 평가했다.',
        '', '그림 점검 절차 출처: Kassis 외 (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065). 2026-10-09에 v2 서지 확인. 분리 성능 근거 인용이 아니다.','']
    output.with_suffix('.md').write_text('\n'.join(text))
    print(json.dumps(dict(output=str(output),paired_epochs=common,rows=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.source,a.output,a.require_complete)
