"""Unsmoothed development means, with undefined scores explicitly shown."""
import argparse
import csv
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def run(source,destination):
    value=json.loads(source.read_text());epochs=value['epochs']
    if len(epochs)<2:raise ValueError('At least initial and one completed epoch required')
    for suffix in ('.pdf','.png','.csv','.json'):
        if destination.with_suffix(suffix).exists():raise ValueError('Refuse overwrite of existing figure')
    font=Path('/home/pyj/.fonts/pretendard/Pretendard-Regular.ttf')
    family=fm.FontProperties(fname=str(font)).get_name()
    fm.fontManager.ttflist=[f for f in fm.fontManager.ttflist if f.name!=family]
    fm.fontManager.addfont(str(font));fm.fontManager.addfont(str(font.with_name('Pretendard-Bold.ttf')))
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':True,
        'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(9.4,7.6),layout='constrained',sharey='row')
    methods=[('spectral','고정 스펙트럼 분배','#D55E00','--'),
             ('frame_spectral','프레임별 스펙트럼 분배','#009E73',':')]
    rows=[];undefined=[]
    for e in epochs:
        for g in e['by_count']:rows.append(dict(method='unet',epoch=e['epoch'],**g))
    for key,label,color,style in methods:
        if key in value:
            for g in value[key]['by_count']:rows.append(dict(method=key,epoch='',**g))
    for col,count in enumerate((2,3)):
        for row,metric in enumerate(('mean_nmse','mean_si_sdr')):
            ax=axes[row,col]
            x=[e['epoch'] for e in epochs]
            y=[next(g for g in e['by_count'] if g['count']==count)[metric] for e in epochs]
            ax.plot(x,[np.nan if v is None else v for v in y],color='#0072B2',marker='o',lw=1.8,label='U-Net')
            for key,label,color,style in methods:
                if key not in value:continue
                g=next(g for g in value[key]['by_count'] if g['count']==count)
                if g[metric] is not None:ax.axhline(g[metric],color=color,ls=style,lw=1.7,label=label)
                else:
                    n=g.get('nonfinite_si_sdr',0)
                    undefined.append(f'{count}신호 {label}: {n}성분 미정의')
            ax.set_title(f'{count}신호 · '+('NMSE ↓' if row==0 else '복소 SI-SDR ↑'))
            ax.set_xlabel('추가 학습 epoch (0 = 동일 관측의 초기 모델)')
            ax.set_ylabel('NMSE (로그 눈금)' if row==0 else '복소 SI-SDR (dB)')
            ax.set_xticks(range(max(x)+1));ax.grid(alpha=.2)
            if row==0:ax.set_yscale('log')
            else:ax.axhline(0,color='.6',lw=.6,zorder=0)
            ax.legend(fontsize=8,loc='best')
    fig.suptitle('수신 주파수 간격을 보존한 합성 I/Q 복원',fontsize=15,fontweight='bold')
    caption='RFUAV 공통 관측 대역 · 조건별 동일 210혼합 · seed 0\n반복 개발 검증 평균; 독립 시험·물리 드론 대수·원기록 전체 복원을 뜻하지 않음'
    if undefined:caption+='\n'+' / '.join(undefined)+' → SI-SDR 평균 미표시'
    fig.supxlabel(caption,fontsize=8.5)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination.with_suffix('.pdf'))
    fig.savefig(destination.with_suffix('.png'),dpi=300,transparent=False)
    plt.close(fig)
    fields=['method','epoch','count','cases','mean_nmse','mean_si_sdr','nonfinite_si_sdr','weakest_nmse','construction_count_accuracy']
    with destination.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({k:r.get(k) for k in fields})
    meta=dict(source_report=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),font=str(font),
        dpi=300,panels_counts=[2,3],csv_counts=[1,2,3],epochs=[e['epoch'] for e in epochs],
        independent_test=False,uncertainty_bands=False,means_smoothed=False,shared_y_axis_per_metric=True,
        undefined_scores='blank CSV and annotated absent line, never replaced by zero or finite-only mean',
        limitations='Repeated development means; one seed; same records; observation band limited by common digital coverage')
    destination.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.report,args.output)
