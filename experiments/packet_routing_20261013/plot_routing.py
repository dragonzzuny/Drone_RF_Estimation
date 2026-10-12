"""Vector figure of observed routing, with matching axes and explicit labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT.parent/'uav_analysis/rf_detection/separator_detector_20261012/retest_best_e079'
OUT=ROOT/'reports/2026-10-13/figures'


def main():
    OUT.mkdir(exist_ok=True)
    refs=np.load(OLD/'SEPARATED_EVENT.npz')['references'][:2].astype(np.complex128)
    ys={e:np.load(ROOT/f'local/packet_routing_20261013_e{e:03d}_v{2 if e==79 else 1}/ASSEMBLED.npz')['C8_hann'].astype(np.complex128) for e in (79,93)}
    rows=[]
    for lo in range(0,200000,8192):
        hi=min(lo+8192,200000);r=refs[:,lo:hi];scale=np.sqrt(np.sum(abs(r)**2,axis=-1))
        design=(r/scale[:,None]).T
        row=dict(time_ms=(lo+hi)/2/100000,power=np.mean(abs(r)**2,axis=-1).tolist())
        for e in ys:
            a=np.linalg.lstsq(design,ys[e][:,lo:hi].T,rcond=None)[0]/scale[:,None]
            row[str(e)]=(abs(a[0])**2).tolist()
        rows.append(row)
    x=np.array([r['time_ms'] for r in rows]);power=np.array([r['power'] for r in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.labelsize':9,'axes.titlesize':10})
    fig,axes=plt.subplots(3,1,figsize=(7.2,7.1),sharex=True,layout='constrained')
    axes[0].plot(x,power[:,0],color='#0072B2',label='FPV reference')
    axes[0].plot(x,power[:,1],color='#D55E00',label='Mavic 3 Pro reference')
    axes[0].set_ylabel('Mean |I/Q|²');axes[0].set_title('(a) Reference power in the fixed 2 ms mixture',loc='left')
    axes[0].legend(frameon=False,ncol=2,loc='upper right')
    for ax,e in zip(axes[1:],(79,93)):
        c=np.array([r[str(e)] for r in rows])
        ax.plot(x,c[:,2],color='#0072B2',lw=1.7,label='FPV-assigned output (index 2)')
        ax.plot(x,c[:,1],color='#D55E00',lw=1.7,ls='--',label='Mavic-assigned output (index 1)')
        ax.set_ylim(0,1.32);ax.set_yticks([0,.25,.5,.75,1.]);ax.set_ylabel('FPV coefficient |a|²')
        ax.set_title(f'({"b" if e==79 else "c"}) Epoch {e}: FPV reference component in each output',loc='left')
        ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=7.4)
    for ax in axes:
        ax.axvspan(.65,1.35,color='#999999',alpha=.10,lw=0)
        ax.grid(axis='y',alpha=.18);ax.set_xlim(0,2)
    axes[-1].set_xlabel('Time within the development event (ms)')
    fig.suptitle('Output routing before and after further training',fontsize=12)
    fig.supxlabel('C8 + Hann; global output assignment. Shading: fixed packet diagnostic interval.\n'
                  'Joint reference regression is diagnostic; coefficients are not additive power fractions.',fontsize=8)
    for suffix in ('svg','pdf','png'):
        fig.savefig(OUT/f'routing_e079_e093.{suffix}',dpi=300,facecolor='white')
    svg=OUT/'routing_e079_e093.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    (OUT/'routing_e079_e093_data.json').write_text(json.dumps(dict(rows=rows,block_samples=8192,
        source='Previously exposed RFUAV B development event; not independent validation',
        method='Joint complex least squares on both references; fixed whole-event assignment',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('FIGURE_COMPLETE')


if __name__=='__main__':main()
