"""All multi-source B24/C24 conditions, with both mean and individual cases."""
import csv
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'reports/2026-10-13/CONSTRAINED_ROUTING_RESULT.json'
PREFIX=ROOT/'reports/2026-10-13/figures/routing_b_c'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data=json.loads(INPUT.read_text());assert data['status']=='VERIFIED'
    fields=['group','index','count','categories','nominal_levels_db','method','mean_nmse','mean_si_sdr','weakest_nmse']
    table=[]
    for group,values in data['groups'].items():
        for case in values['cases']:
            for method in ('parent','previous','candidate'):
                m=case[method];assert all(v is not None for v in m['si_sdr'])
                table.append(dict(group=group,index=case['index'],count=m['count'],
                    categories=' + '.join(case['categories']),nominal_levels_db=json.dumps(case['nominal_levels_db']),
                    method=method,mean_nmse=float(np.mean(m['nmse'])),
                    mean_si_sdr=float(np.mean(m['si_sdr'])),weakest_nmse=m['nmse'][m['weakest_index']]))
    files=[PREFIX.with_suffix(x) for x in ('.pdf','.svg','.png','.csv','.json','.md')]
    assert not any(p.exists() for p in files)
    with files[3].open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(table)
    metrics=[('mean_nmse','Source NMSE (log scale; lower is better)'),
             ('mean_si_sdr','Complex SI-SDR (dB; higher is better)'),
             ('weakest_nmse','Weakest-source NMSE (log; lower is better)')]
    colors={2:'#0072B2',3:'#D55E00'};markers={2:'o',3:'s'}
    styles={2:'-',3:'--'};x=np.arange(3);groups=['B24','C24']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for column,(metric,label) in enumerate(metrics):
            vals=[r[metric] for r in table if r['count']>1]
            if metric!='mean_si_sdr':
                assert min(vals)>0;limits=(min(vals)/1.4,max(vals)*1.4)
            else:
                width=max(vals)-min(vals);limits=(min(vals)-.07*width,max(vals)+.07*width)
            for row,group in enumerate(groups):
                ax=axes[row,column]
                for count in (2,3):
                    cases=[v for v in data['groups'][group]['cases'] if v['parent']['count']==count]
                    assert len(cases)==(12 if count==2 else 7)
                    offset=-.04 if count==2 else .04
                    for number,case in enumerate(cases):
                        jitter=(number/(len(cases)-1)-.5)*.035
                        y=[next(r[metric] for r in table if r['group']==group and r['index']==case['index']
                                and r['method']==key) for key in ('parent','previous','candidate')]
                        ax.plot(x+offset+jitter,y,color=colors[count],marker=markers[count],markersize=2.8,
                                alpha=.35,lw=.65,ls=styles[count])
                    means=[data['groups'][group]['summaries'][key]['by_count'][count-1][metric]
                           for key in ('parent','previous','candidate')]
                    ax.plot(x+offset,means,color=colors[count],marker=markers[count],markersize=7,
                            markeredgecolor='white',markeredgewidth=.6,lw=2.3,ls=styles[count])
                ax.set_xticks(x,['U-Net + C8','PSD routing','Constrained\nrouting'])
                ax.set_ylim(*limits);ax.set_xlim(-.25,2.25)
                if metric!='mean_si_sdr':ax.set_yscale('log')
                ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
                ax.grid(axis='y',which='major',alpha=.18)
                title='B: same recording, separate time blocks' if row==0 else 'C: separate recording and VTSBW condition'
                ax.set_title(f'{chr(97+row*3+column)}) {title}',loc='left',fontsize=9)
        legend=[Line2D([0],[0],color=colors[k],marker=markers[k],ls=styles[k],lw=2.3,
                       label=f'{k} sources: mean over {12 if k==2 else 7} cases') for k in (2,3)]
        legend.append(Line2D([0],[0],color='#555555',marker='.',lw=.65,alpha=.5,label='Thin lines: every individual case'))
        fig.legend(handles=legend,loc='outside upper center',ncol=3,frameon=False)
        for path in files[:3]:fig.savefig(path,dpi=300,facecolor='white')
        plt.close(fig)
    files[5].write_text('''# 긴 구간 배정 비교: B와 C

각 행의19개 다중 성분 조건을 모두 표시한다. 두 성분12개와 세 성분7개다.
굵은 선과 큰 표식은 성분 수별 산술평균, 가는 선은 개별 조건이다. 같은 성분 수의
모든 조건은 동일 성분 수를 가지므로 조건 평균의 평균과 성분 평균이 일치한다.
오른쪽은 각 혼합에서 원정답의 전력이 가장 작은 성분의 NMSE를 평균한 값이다.
명목 동일 전력 조건에서도 실제 저장 파형의 미세한 전력 차이로 최약 성분을 정한다.

모든 비교는 동일 부모99 전체20.8896ms 복소 출력을 사용한다. PSD routing은
자유 군집 후 일대일 배정(v2), Constrained는 군집 갱신부터 일대일 배정이다.
정답은 점수 계산용으로만 사용하며 배정에 기체명이나 원파형을 제공하지 않는다.
개수는 부모 모델의 예측값이다. 추가 학습은 없다. 이 그림의 새 신경망 학습 효과는0이다.

B는 VTSBW=10 원녹음 안의 시간 분리, C는 별도 VTSBW=20 녹음이다.
VTSBW는 영상전송 신호 대역폭이며 표본률은 양쪽 모두100MS/s다.
두 자료 모두 노출된 개발 자료이며 독립적인 반복 실험이나 새 최종시험이 아니다.
신뢰구간을 그리지 않았다. 얇은 선은 불확실성 구간이 아니라 실제 각 조건이다.

NMSE 두 열은 밑10 로그축이고 SI-SDR는 선형 dB축이다. 열마다 B/C 축 범위가
같으며 모든 값이 표시된다. x축의 작은 결정적 이동은 겹침을 줄이는 표현일 뿐이다.
수치 평활화나 보간은 하지 않았다. 단일 성분은 다중 분리 그래프의 대상에서 제외했지만
양쪽 각5조건의 모든 값은 같은 CSV와 원JSON에 포함돼 있다.

전 조건 우월성은 아니다. 제약 군집은 B의2성분 NMSE와 최약NMSE에서 v2보다
나쁘다. C의3성분 최약NMSE는 두 배정 방법 모두 부모보다 크다.

일반 연구 검토용 임시 그림이며 특정 학술지 규격 준수를 주장하지 않는다.
그림 작성에는 Scientific Visualization 스킬을 사용했다. 도구 출처:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
''')
    manifest=dict(status='EXPORTED_PENDING_VISUAL_INSPECTION',input_sha256=sha(INPUT),source_sha256=sha(Path(__file__)),
        size_inches=[12,7],png_dpi=300,matplotlib_version=matplotlib.__version__,
        transformation='Arithmetic mean per fixed count; all19multisource cases per group; no smoothing',
        uncertainty='None; dependent development conditions',single_source_data_in_csv=True,
        scales=['log10','linear dB','log10'],axes_shared_by_column=True,
        alt_text='Six-panel plot showing mean and every-case waveform reconstruction metrics before and after two reference-free routing methods. B improves more than C; weak three-source C reconstruction remains worse than the parent.',
        exports={p.name:sha(p) for p in files if p.suffix!='.json'})
    files[4].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('ROUTING_FIGURE_EXPORTED',PREFIX,flush=True)


if __name__=='__main__':main()
