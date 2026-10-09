"""Plot audited actual epochs; do not substitute selected epoch zero curves."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'reports/2026-10-10'


def run():
    def read(p):return json.loads(p.read_text())
    root=ROOT/'local/tf_axis_continuation_20261010_v1'
    p=read(root/'PROTOCOL.json');old=p['original_protocol']
    assert read(PUBLIC/'TF_AXIS_CONTINUATION_AUDIT.json')['status']=='PASS'
    first=Path(p['predecessor'])
    # Use the same independently reaggregated validation helper as the audit.
    import continue_training as training
    parent,ids=training.w.validate(Path(old['baseline']),None)
    candidate_summaries=[parent,training.w.validate(first/'VALIDATION_001.json',ids)[0]]
    candidate_summaries += [training.w.validate(root/f'VALIDATION_{i:03d}.json',ids)[0] for i in (2,3)]
    control_summaries=[parent]+[training.w.validate(Path(old['study'])/f'retained_unet/VALIDATION_{i:03d}.json',ids)[0] for i in (1,2,3)]
    def value(summary,count,key):return next(x[key] for x in summary['by_count'] if x['count']==count)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.axisbelow':True})
    fig,axes=plt.subplots(3,2,figsize=(7.4,7.0),sharex=True,layout='constrained')
    keys=[('mean_nmse','Mean I/Q NMSE ↓'),('mean_si_sdr','Complex SI-SDR (dB) ↑'),('weakest_nmse','Weakest-source NMSE ↓')]
    for col,count in enumerate((2,3)):
        for row,(key,label) in enumerate(keys):
            ax=axes[row,col]
            ax.axhline(value(parent,count,key),color='#666666',linestyle=':',linewidth=1.2,label='Starting checkpoint')
            ax.plot([0,75,150,225],[value(s,count,key) for s in control_summaries],color='#2474A4',marker='o',markersize=5,linewidth=1.4,label='Original U-Net continuation')
            ax.plot([0,75,150,225],[value(s,count,key) for s in candidate_summaries],color='#C65B24',marker='s',markersize=4,markerfacecolor='white',linestyle='--',linewidth=1.4,label='U-Net + TF-axis BLSTM')
            ax.set_ylabel(label);ax.grid(alpha=.23);ax.set_xticks([0,75,150,225])
            if row==0:ax.set_title(f'{count} source contributions')
            if row==2:ax.set_xlabel('Additional optimizer updates')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=1,frameon=False)
    fig.supxlabel('One seed; reused DEV mixtures (210 per count), not an independent test.',fontsize=8)
    for suffix in ('pdf','svg','png'):
        fig.savefig(PUBLIC/f'TF_AXIS_CONTINUATION_CURVES.{suffix}',dpi=300)
    plt.close(fig)
    evidence=dict(status='COMPLETE',actual_epochs=[0,1,2,3],additional_updates=[0,75,150,225],
        candidate=candidate_summaries,control=control_summaries,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256((PUBLIC/'TF_AXIS_CONTINUATION_AUDIT.json').read_bytes()).hexdigest(),
        uncertainty_bars='Not drawn: single-seed descriptive trajectory with clustered reused development records',
        heldout_read=False,recorded_iq_reads=0)
    (PUBLIC/'TF_AXIS_CONTINUATION_CURVES.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print({'status':'COMPLETE','plot':'TF_AXIS_CONTINUATION_CURVES.pdf'})


if __name__=='__main__':run()
