"""Plot every saved point and source from the completed TRAIN4 diagnosis."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/2026-10-11'


def main():
    files=[PUBLIC/'TFGRIDNET_FIT_RESULT.json',PUBLIC/'TFGRIDNET_CONTINUATION_RESULT.json']
    first,last=[json.loads(p.read_text()) for p in files]
    assert first['status']==last['status']=='COMPLETE'
    points=first['history']+last['history'][1:]
    assert len({p['step'] for p in points})==len(points)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'drone-rf-fixed-train4'})
    fig,axes=plt.subplots(2,2,figsize=(13.7,8.8),constrained_layout=True,
        gridspec_kw={'width_ratios':[1,1.35]})
    x=[p['step'] for p in points]
    for count,color,marker in [(2,'#0072B2','o'),(3,'#D55E00','s')]:
        values=[next(v for v in p['by_count'] if v['count']==count) for p in points]
        for ax,key in [(axes[0,0],'mean_nmse'),(axes[0,1],'mean_si_sdr'),(axes[1,0],'weakest_nmse')]:
            ax.plot(x,[v[key] for v in values],color=color,marker=marker,ms=4,lw=1.7,label=f'{count} sources')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0,0].set(title='A  Mean waveform error',ylabel='NMSE (log scale; lower is better)',yscale='log')
    axes[0,1].set(title='B  Waveform structure',ylabel='Complex SI-SDR (dB; higher is better)')
    axes[0,1].axhline(0,color='gray',lw=.7,ls=':')
    axes[1,0].set(title='C  Weakest source in each mixture',ylabel='Mean NMSE (log scale; lower is better)',yscale='log')
    later=[p for p in points if p['step']>=32]
    values=np.array([[v for row in p['rows'] for v in row['nmse']] for p in later]).T
    labels=[f"{row['count']} src / mix {row['index']} / {c.replace('DJI ','').replace(' COMBO','')}"
        for row in later[0]['rows'] for c in row['categories']]
    assert values.shape==(10,len(later)) and np.isfinite(values).all() and values.min()>0
    ax=axes[1,1];norm=LogNorm(vmin=values.min()*.95,vmax=values.max()*1.05)
    im=ax.imshow(values,aspect='auto',cmap='viridis_r',norm=norm)
    ax.set_yticks(range(10),labels,fontsize=8)
    ax.set_xticks(range(len(later)),[str(p['step'])+('*' if p['step']==last['selected']['step'] else '') for p in later])
    ax.set(title='D  Every source: NMSE',xlabel='Optimizer updates (*selected diagnostic state)')
    for i in range(10):
        for j in range(len(later)):
            v=values[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,
                color='white' if norm(v)>.62 else '#111111')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label='NMSE (log color scale)')
    fig.suptitle('TF-GridNet RF — fixed TRAIN mixtures only\n4 mixtures / 10 source contributions / seed 0 / no new-recording validation',fontsize=13)
    for suffix in ['pdf','svg']:
        metadata={'Creator':'Drone RF Estimation research; see plot_fit.py'}
        if suffix=='pdf':metadata.update(CreationDate=None,ModDate=None)
        else:metadata.update(Date=None)
        fig.savefig(PUBLIC/f'TFGRIDNET_FIT_CURVES.{suffix}',bbox_inches='tight',facecolor='white',metadata=metadata)
        if suffix=='svg':
            target=PUBLIC/f'TFGRIDNET_FIT_CURVES.{suffix}'
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    preview=ROOT/'local/tfgridnet_fit_curves_20261011.png';fig.savefig(preview,dpi=130,bbox_inches='tight',facecolor='white');plt.close(fig)
    with (PUBLIC/'TFGRIDNET_FIT_CURVES.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['update','mixture_index','count','source','nmse','complex_si_sdr_db','weakest'])
        for p in points:
            for row in p['rows']:
                for j,c in enumerate(row['categories']):writer.writerow([p['step'],row['index'],row['count'],c,row['nmse'][j],row['si_sdr'][j],j==row['weakest_index']])
    receipts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*files]}
    (PUBLIC/'TFGRIDNET_FIT_CURVES_PROVENANCE.json').write_text(json.dumps(dict(input_sha256=receipts,
        matplotlib=matplotlib.__version__,no_interpolated_or_omitted_observations=True,
        scope='fixed TRAIN4 only; no error bars because these are not independent replications'),indent=2)+'\n')
    print(preview)


if __name__=='__main__':main()
