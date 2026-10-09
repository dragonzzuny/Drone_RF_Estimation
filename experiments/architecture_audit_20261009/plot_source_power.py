"""Plot all selected U-Net multi-source development outcomes, no waveform reads."""
import argparse
import csv
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import watch_epochs as watch

STYLE = {
    'DJI AVATA2': ('Avata 2', '#0072B2', 'o'),
    'DJI FPV COMBO': ('FPV', '#D55E00', 's'),
    'DJI MAVIC3 PRO': ('Mavic 3 Pro', '#009E73', '^'),
    'DJI MINI3': ('Mini 3', '#CC79A7', 'D'),
    'DJI MINI4 PRO': ('Mini 4 Pro', '#E69F00', 'v'),
}


def render(study, output):
    snapshot = watch.snapshot(study)
    if snapshot['status'] != 'COMPLETED' or snapshot['common_epoch'] != 5:
        raise ValueError('Only completed original architecture study is allowed')
    selected = snapshot['common_selected']['unet_mean']['epoch']
    path = study/'unet_mean'/f'VALIDATION_{selected:03d}.json'
    watch.validate(path, None)
    points = []
    for row in watch.read(path)['rows']:
        if row['count'] == 1:
            continue
        for j, category in enumerate(row['categories']):
            p = row['reference_power']
            points.append(dict(mixture_index=row['index'], source_index=j, count=row['count'],
                category=category, sir_db=10*math.log10(p[j]/sum(x for i,x in enumerate(p) if i != j)),
                nmse=row['nmse'][j], si_sdr_db=row['si_sdr'][j]))
    if len(points) != 1050 or any(not math.isfinite(p[k]) for p in points
                                  for k in ('sir_db','nmse','si_sdr_db')):
        raise ValueError('Missing or nonfinite source case; no filtering allowed')
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans',
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(11.5,8.2),sharex=True,sharey='row')
    x = [p['sir_db'] for p in points]
    xmin,xmax = min(x)-2,max(x)+2
    for col,count in enumerate((2,3)):
        for category,(label,color,marker) in STYLE.items():
            subset = [p for p in points if p['count']==count and p['category']==category]
            if not subset:
                continue
            for row,key in enumerate(('nmse','si_sdr_db')):
                axes[row,col].scatter([p['sir_db'] for p in subset],[p[key] for p in subset],
                    s=20,alpha=.62,c=color,marker=marker,edgecolors='none',label=label)
        axes[0,col].set_yscale('log')
        axes[0,col].axhline(1,color='#555555',ls=':',lw=1)
        axes[1,col].axhline(0,color='#555555',ls=':',lw=1)
        axes[0,col].set_title(f'{count} recorded components | 210 mixtures',fontweight='bold',pad=12)
        for ax in axes[:,col]:
            ax.grid(axis='y',which='major',alpha=.15)
            ax.set_axisbelow(True)
            ax.set_xlim(xmin,xmax)
        axes[1,col].set_xlabel('Local reference-component SIR (dB)')
    axes[0,0].set_ylabel('I/Q NMSE (log scale; lower is better)')
    axes[1,0].set_ylabel('Complex SI-SDR (dB; higher is better)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.942),
               ncol=5,frameon=False,markerscale=1.3,handletextpad=.4,columnspacing=1.2)
    fig.suptitle('Where reconstruction errors remain',fontsize=17,x=.51,y=.995)
    fig.text(.51,.953,f'STFT U-Net | selected epoch {selected} after 5 epochs / 375 updates | RFUAV development | seed 0',
             fontsize=10,ha='center')
    fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.16,hspace=.17,wspace=.12)
    fig.text(.09,.09,'All 1,050 component outcomes shown; no error-based exclusion or output gain correction.',fontsize=9)
    fig.text(.09,.066,'SIR = reference power / sum of other reference powers. It excludes complex cross terms.',fontsize=9)
    fig.text(.09,.042,'Dotted lines: NMSE = 1 (zero estimate); SI-SDR = 0 dB. Categories and recording conditions are confounded.',fontsize=9)
    fig.text(.09,.018,'Correlated crops from 5 recording groups; descriptive development results, not independent test evidence.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png'):
        fig.savefig(output.with_suffix('.'+extension),dpi=300,facecolor='white')
    plt.close(fig)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(points[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(points);watch.write(output.with_suffix('.csv'),stream.getvalue())
    manifest=dict(status='COMPLETE',source_sha256=watch.digest(Path(__file__)),
        validation_sha256=watch.digest(path),study_protocol_sha256=snapshot['protocol_sha256'],
        selected_epoch=selected,all_source_cases=1050,mixtures=420,waveform_reads=0,heldout_read=False,
        no_filtering=True,independent_test=False,
        files={ext:watch.digest(output.with_suffix('.'+ext)) for ext in ('pdf','png','csv')})
    watch.write(output.with_suffix('.json'),manifest)
    print(manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.study.resolve(),args.output.resolve())
