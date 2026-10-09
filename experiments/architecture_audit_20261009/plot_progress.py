"""Audited per-epoch waveform curves; saved validation JSON only, no raw I/Q."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import watch_epochs as watch

STYLES = {
    'wavenet_cycle10': ('WaveNet, cycle 10', '#0072B2', 'o', '-'),
    'wavenet_cycle15': ('WaveNet, cycle 15', '#D55E00', 's', '--'),
    'unet_mean': ('STFT U-Net', '#009E73', '^', '-.'),
}
FIELDS = [('mean_nmse', 'I/Q NMSE (log scale)', True),
          ('mean_si_sdr', 'Complex SI-SDR (dB)', False),
          ('weakest_nmse', 'Weakest-component NMSE (log scale)', True)]


def render(report, output):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(3,2,figsize=(10.8,10),sharex=True,
                             gridspec_kw={'hspace':.30,'wspace':.27})
    table=[]
    for column,count in enumerate((2,3)):
        for row,(field,label,log) in enumerate(FIELDS):
            ax=axes[row,column]
            for arm,history in report['histories'].items():
                name,color,marker,style=STYLES[arm]
                values=[next(g for g in h['by_count'] if g['count']==count)[field] for h in history]
                epochs=[h['epoch'] for h in history]
                ax.plot(epochs,[float('nan') if v is None else v for v in values],
                        color=color,marker=marker,linestyle=style,linewidth=1.8,
                        markersize=5,label=name)
            ax.set_ylabel(label,fontsize=9.5)
            if log:
                ax.set_yscale('log')
                ax.axhline(1,color='#999999',lw=.8,ls=':')
            else:
                ax.axhline(0,color='#999999',lw=.8,ls=':')
            ax.set_xlim(-.1,5.1)
            ax.set_xticks(range(6))
            ax.grid(axis='y',which='major',alpha=.18)
            if row==0:ax.set_title(f'{count} recorded components',fontweight='bold',pad=10)
            if row==2:ax.set_xlabel('Completed epoch (0 = untrained)')
    for arm,history in report['histories'].items():
        for epoch in history:
            for group in epoch['by_count'][1:]:
                table.append(dict(arm=arm,epoch=epoch['epoch'],updates=epoch['epoch']*75,
                    **{k:group[k] for k in ('count','cases','mean_nmse','mean_si_sdr','weakest_nmse')}))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.955),ncol=3,frameon=False)
    fig.suptitle('Drone RF waveform reconstruction: architecture comparison',fontsize=15,y=.995)
    fig.text(.5,.964,f"RFUAV same-band development validation | seed 0 | common completed budget: {report['common_epoch']} epochs",
             ha='center',fontsize=10)
    fig.subplots_adjust(top=.89,bottom=.13,left=.10,right=.98)
    fig.text(.10,.055,'210 mixtures per source count; 75 updates/epoch; model capacities differ. Raw epoch values, no smoothing.',fontsize=9)
    fig.text(.10,.035,'Lower NMSE and higher SI-SDR are better. Dotted NMSE=1 is the zero-output reference.',fontsize=9)
    fig.text(.10,.015,'NMSE axes are logarithmic. Reused development validation, not a held-out test.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        path=output.with_suffix('.'+suffix)
        temp=path.with_name(path.stem+'.tmp.'+suffix)
        fig.savefig(temp,dpi=250,facecolor='white')
        temp.replace(path)
    plt.close(fig)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    watch.write(output.with_suffix('.csv'),stream.getvalue())
    watch.write(output.with_suffix('.json'),dict(protocol_sha256=report['protocol_sha256'],
        reported_at=report['reported_at'],common_epoch=report['common_epoch'],
        rows=table,validation_sha256={f"{e['arm']}/{e['epoch']}":e['validation_sha256'] for e in report['events']},
        smoothing=False,all_completed_epochs_included=True,heldout_iq_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    sources={str(Path(__file__).resolve()):watch.digest(Path(__file__)),
             str(Path(watch.__file__).resolve()):watch.digest(Path(watch.__file__))}
    snapshot=args.run/'plot_source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name,digest in sources.items():
        shutil.copyfile(name,snapshot/Path(name).name)
    watch.write(snapshot/'SOURCES.json',sources)
    previous=None
    while True:
        if any(watch.digest(Path(name))!=digest for name,digest in sources.items()):
            raise ValueError('Plot worker source changed; restart explicitly')
        result=watch.snapshot(args.run.resolve())
        signature=tuple((a,len(h)) for a,h in result['histories'].items())
        if result['events'] and signature!=previous:
            render(result,args.output)
            previous=signature
            print(json.dumps(dict(rendered_events=len(result['events']),common_epoch=result['common_epoch'])),flush=True)
        if not args.watch or result['status']!='WORKER_ALIVE':
            break
        time.sleep(30)
