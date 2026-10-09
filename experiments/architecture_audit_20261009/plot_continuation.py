"""Plot all actual epochs of the audited matched learning-rate experiments."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import watch_epochs as watch


def main(source, reference, output):
    low, high=watch.read(source),watch.read(reference)
    if low['status']!='COMPLETE' or high['status']!='COMPLETE':
        raise ValueError('Only completed audited summaries may be plotted')
    if low['parent']!=high['parent']:
        raise ValueError('Different parent metrics')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.2),sharex=True)
    series=[]
    for col,count in enumerate((2,3)):
        for row,key in enumerate(('mean_nmse','mean_si_sdr')):
            ax=axes[row,col]
            for arm,color in [('original','#0072B2'),('log1p_nmse','#D55E00')]:
                for label,summary,style,marker in [('1e-4',low,'-','o'),('5e-4',high,'--','s')]:
                    history=summary['histories'][arm]
                    if [r['epoch'] for r in history]!=[0,1,2,3]:
                        raise ValueError('Missing an actual epoch')
                    values=[next(g[key] for g in h['by_count'] if g['count']==count) for h in history]
                    series.append(dict(count=count,metric=key,loss=arm,lr=label,values=values))
                    ax.plot(range(4),values,color=color,linestyle=style,marker=marker,
                            linewidth=1.8,markersize=4,label=f"{'Original' if arm=='original' else 'log1p'} / lr {label}")
            ax.grid(alpha=.2);ax.set_xticks(range(4));ax.set_xlim(-.08,3.08)
            ax.set_title(f'{count} recorded contributions')
            ax.set_ylabel('Raw complex I/Q NMSE (lower)' if row==0
                          else 'Complex SI-SDR, dB (higher)')
            if row==1:ax.set_xlabel('Additional fine-tuning epoch')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.suptitle('Full U-Net: matched loss and learning-rate trajectories',fontsize=14)
    fig.text(.5,.015,'RFUAV development split; 630 fixed mixtures; seed 0. '
             'Actual epochs, not selected checkpoints.\n'
             'Low-lr epoch 1 imported exactly; epochs 2-3 continue optimizer and RNG. '
             'Adaptive development comparison.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.95))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'),dpi=220)
    plt.close(fig)
    watch.write(output.with_suffix('.json'),dict(status='COMPLETE',
        source_sha256=watch.digest(Path(__file__)),low_summary_sha256=watch.digest(source),
        high_summary_sha256=watch.digest(reference),series=series,
        pdf_sha256=watch.digest(output.with_suffix('.pdf')),
        png_sha256=watch.digest(output.with_suffix('.png')),error_bars=False,
        independent_recording_intervals_claimed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.source,a.reference,a.output)
