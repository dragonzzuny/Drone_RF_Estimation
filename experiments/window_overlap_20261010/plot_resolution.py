"""Visualize all existing mixed TRAIN cases, no cherry-picked subset."""
import argparse
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(public):
    path=public/'SOURCE_RESOLUTION_RESULT.json';value=json.loads(path.read_text())
    assert value['status']=='COMPLETE_CHECKED' and len(value['rows'])==500
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8.4,4.2));fig.subplots_adjust(left=.085,right=.98,bottom=.31,top=.84,wspace=.34)
    colors={2:'#0072B2',3:'#D55E00'}
    for n in (2,3):
        summary=sorted((q for q in value['summary'] if q['count']==n),key=lambda q:q['frequency_pool'])
        x=[q['frequency_pool'] for q in summary];y=[100*q['median_weak_coverage'] for q in summary]
        axes[0].plot(x,y,'o-' if n==2 else '^-',color=colors[n],label=f'{n} sources (n={summary[0]["cases"]})',markersize=4)
        rows=[q for q in value['rows'] if q['count']==n and q['weakest'] and q['frequency_pool']==1]
        axes[1].scatter([100*q['source_energy_in_winning_bins'] for q in rows],[q['parent_nmse'] for q in rows],
            s=30,c=colors[n],alpha=.8,marker='o' if n==2 else '^',edgecolors='white',linewidths=.4)
    axes[0].set_xscale('log',base=2);axes[0].set_xticks([1,2,4,8,16],['1','2','4','8','16'])
    axes[0].set_xlabel('Frequency pooling factor (195.3 kHz at 1)')
    axes[0].set_ylabel('Median own energy in dominant bins (%)')
    axes[0].set_title('(a) Reference power pooling only');axes[0].set_ylim(0,103);axes[0].legend(loc='center right',fontsize=8,frameon=False)
    axes[1].set_xlabel('Own energy in dominant bins (%)')
    axes[1].set_ylabel('Parent weak-source I/Q NMSE')
    axes[1].set_title('(b) Original resolution: all 38 mixtures');axes[1].set_xlim(-3,103);axes[1].set_ylim(0,1.30)
    axes[1].axhline(1,color='.4',lw=.8,ls='--')
    case=next(q for q in value['rows'] if q['index']==37 and q['weakest'] and q['frequency_pool']==1)
    axes[1].annotate('TRAIN case 37',xy=(100*case['source_energy_in_winning_bins'],case['parent_nmse']),
        xytext=(47,1.19),fontsize=8,arrowprops={'arrowstyle':'-','lw':.7,'color':'.35'})
    for ax in axes:
        ax.grid(True,color='.88',linewidth=.5);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Weak-source representation: target coverage and current reconstruction',y=.97,fontsize=11)
    fig.text(.085,.055,'Fixed TRAIN48: 24 two-source and 14 three-source cases; 10 single-source cases omitted.\n'
        'Reference-power pooling is not a U-Net feature intervention. Dashed line: zero-output NMSE = 1.\n'
        'No new training or held-out evaluation.',fontsize=8,color='.25')
    paths=[]
    for suffix in ('pdf','svg','png'):
        out=public/f'SOURCE_RESOLUTION.{suffix}';fig.savefig(out,dpi=220);paths.append(out)
        if suffix=='svg':
            out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')
    plt.close(fig)
    receipt=dict(status='CREATED',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_cases={'two':24,'three':14},all_weak_mixed_cases_included=True,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (public/'SOURCE_RESOLUTION_FIGURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);run(p.parse_args().public)
