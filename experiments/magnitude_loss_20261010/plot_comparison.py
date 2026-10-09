"""Standalone, actual-epoch waveform comparisons; no fabricated error bars."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(source,output):
    d=json.loads(source.read_text());assert d['status']=='PASS'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.3))
    labels=['Parent\n+0 updates','Waveform\n+75 updates','Magnitude: all\n+75 updates','Magnitude: 2/3\n+75 updates']
    colors=['#3B5670','#8A929A','#CA7739','#3A8A79']
    values=[]
    for row,count in enumerate((2,3)):
        groups=[next(g for g in m['by_count'] if g['count']==count) for m in d['models']]
        for col,(key,title) in enumerate((('mean_nmse','I/Q NMSE (lower is better)'),('mean_si_sdr','Complex SI-SDR, dB (higher is better)'))):
            ax=axes[row,col];ys=[g[key] for g in groups];values.append(dict(count=count,metric=key,values=ys))
            ax.bar(np.arange(4),ys,color=colors,width=.65,zorder=3)
            ax.axhline(0,color='#545A60',linewidth=.8);ax.grid(axis='y',color='#D9DEE3',linewidth=.7,zorder=0)
            ax.set_xticks(range(4),labels,fontsize=8.5)
            ax.set_title(f'{count} source contributions\n{title}',fontsize=10.5,pad=12)
            for i,y in enumerate(ys):
                ax.annotate(f'{y:.4f}' if col==0 else f'{y:.3f}',(i,y),xytext=(0,5 if y>=0 else -12),
                            textcoords='offset points',ha='center',fontsize=9)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylim(0,.85)
            elif count==2:ax.set_ylim(0,4.5)
            else:ax.set_ylim(-4.3,.2)
    fig.suptitle('Native RF separation: actual results after a fixed training budget',fontsize=15,y=.965)
    fig.text(.5,.075,'RFUAV · single inference · DEV630 (210 per count) · 5 original recording groups · seed 0',ha='center',fontsize=9)
    fig.text(.5,.05,'Same retained parent; each trained arm has 75 updates. Actual e1 is shown, including failures.',ha='center',fontsize=9)
    fig.text(.5,.025,'Repeated development comparisons; selective supervision was an adaptive follow-up. No independent held-out test.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.985,top=.875,bottom=.18,hspace=.43,wspace=.23)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'),metadata={'Title':'Native RF fixed-budget waveform comparisons'})
    fig.savefig(output.with_suffix('.png'),dpi=250)
    plt.close(fig)
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(status='GENERATED_PENDING_VISUAL_REVIEW',source_sha256=h(source),plotter_sha256=h(Path(__file__)),
                values=values,artifacts={s:h(output.with_suffix(s)) for s in ('.pdf','.png')},
                model_updates=0,recorded_iq_reads=0)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Generated figure from audited report',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.source,a.output)
