"""Render all predefined lag-product measurements and surrogate controls."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(source,output):
    d=json.loads(source.read_text());assert d['status']=='COMPLETE' and len(d['rows'])==8
    ratios=[];frequencies=[];labels=[];ledger=[]
    for row in d['rows']:
        labels.append(row['category'].replace('DJI ','').replace(' COMBO','')+' / '+row['pack_id'].split('VTSBW=')[1].split('/')[0]+' MHz')
        r=[];f=[]
        for k,a in enumerate(row['actual']):
            controls=[s[k]['peak_fraction_total_lag_product_energy'] for s in row['surrogates']]
            ratio=a['peak_fraction_total_lag_product_energy']/max(controls)
            assert np.isfinite(ratio) and ratio>0
            r.append(ratio);f.append(abs(a['peak_frequency_hz'])/1000)
            ledger.append(dict(clip_id=row['clip_id'],lag_samples=a['lag_samples'],ratio_to_largest_of_three_surrogate_peaks=ratio,
                actual_peak_hz=a['peak_frequency_hz'],actual_peak_fraction=a['peak_fraction_total_lag_product_energy'],surrogate_peak_fractions=controls))
        ratios.append(r);frequencies.append(f)
    ratios=np.array(ratios);frequencies=np.array(frequencies)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.8),gridspec_kw={'width_ratios':[1,1]})
    a=axes[0].imshow(np.log10(ratios),cmap='viridis',aspect='auto',vmin=-.3,vmax=3)
    b=axes[1].imshow(frequencies,cmap='cividis',aspect='auto',vmin=0,vmax=100)
    for ax,data,fmt in ((axes[0],ratios,'.1f'),(axes[1],frequencies,'.1f')):
        ax.set_xticks(range(4),['0','33.34','66.67','142.70']);ax.set_xlabel('Product lag (microseconds)')
        ax.set_yticks(range(8),labels if ax is axes[0] else [])
        for i in range(8):
            for j in range(4):
                if ax is axes[0]:color='black' if np.log10(data[i,j])>1.7 else 'white'
                else:color='black' if data[i,j]>60 else 'white'
                ax.text(j,i,format(data[i,j],fmt),ha='center',va='center',color=color,fontsize=9)
    axes[0].set_title('Original peak / largest surrogate peak')
    axes[1].set_title('Original peak frequency (absolute kHz)')
    c=fig.colorbar(a,ax=axes[0],fraction=.045,pad=.03);c.set_label('log10 peak ratio')
    c=fig.colorbar(b,ax=axes[1],fraction=.045,pad=.03);c.set_label('kHz')
    fig.suptitle('TRAIN lag-product structure: spectrum-preserving phase controls',fontsize=13)
    fig.text(.5,.025,'One predefined clip per TRAIN group; three randomized-phase controls each; 1-100 kHz peak search.\n'
        'Peak concentration is descriptive, not a p-value, protocol label, or separation score.\n'
        'Zero lag measures envelope power; transients can create peaks without sustained repetition.\n'
        'MHz labels are VTSBW metadata, not measured occupied bandwidth.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.95));fig.savefig(output.with_suffix('.pdf'));fig.savefig(output.with_suffix('.png'),dpi=200);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(status='COMPLETE',source_sha256=digest(source),plotter_sha256=digest(Path(__file__)),
        rows=ledger,pdf_sha256=digest(output.with_suffix('.pdf')),png_sha256=digest(output.with_suffix('.png'))),indent=2)+'\n')
    print(dict(status='COMPLETE',groups=8,lag_conditions=32))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve())
