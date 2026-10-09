"""Matched waveform assignments and double-precision spectral diagnostics."""
import os
from pathlib import Path
import statistics
import time
import torch
from objective import spectral_terms, worker
from drone_rf.waveform import waveform_metrics

w=worker.watch
KEYS=('relative_l1','magnitude_nmse','phase_interaction','spectral_complex_nmse','spectral_rms_ratio')


@torch.no_grad()
def run(root,p,net,data):
    results=[]
    for arm,checkpoint,validation_path in (
        ('magnitude_e1',root/'ACTUAL_001.pt',root/'VALIDATION_001.json'),
        ('parent',Path(p['parent_checkpoint']),root/'VALIDATION_000.json'),
        ('retained_control_e1',Path(p['study'])/'retained_unet/ACTUAL_001.pt',Path(p['study'])/'retained_unet/VALIDATION_001.json')):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(saved['model']);del saved;net.eval()
        expected=w.read(validation_path)['rows'];rows=[];started=time.time()
        for index in range(len(data)):
            item=worker.fit.base.batch([data[index]])
            output,logits=worker.predict(net,item)
            metrics=waveform_metrics(output,item['references'],item['active'],item['mixture'])
            active=item['active'][0]
            nmse=metrics['nmse'][0][active].cpu().tolist()
            ref=expected[index];count=int(active.sum())
            assert ref['index']==index and ref['count']==count
            assert max(abs(a-b) for a,b in zip(nmse,ref['nmse']))<2e-6
            assert metrics['assignment'][0].cpu().tolist()==ref['assignment']
            assert int(logits.argmax(-1)[0])+1==ref['predicted_count']
            terms=spectral_terms(output.to(torch.complex128),item['references'].to(torch.complex128),
                item['active'],item['mixture'].to(torch.complex128),metrics['assignment'])
            torch.testing.assert_close((terms['magnitude_nmse']+terms['phase_interaction'])[item['active']],
                terms['spectral_complex_nmse'][item['active']],rtol=1e-10,atol=1e-10)
            row={key:ref[key] for key in ('index','count','categories','pack_ids','nominal_levels_db','reference_power','weakest_index')}
            row.update(nmse=nmse,si_sdr=ref['si_sdr'],assignment=ref['assignment'])
            for key in KEYS:
                value=terms[key][0][active];assert torch.isfinite(value).all()
                row[key]=value.cpu().tolist()
            rows.append(row)
            if (index+1)%50==0:
                w.write(root/'STATE.json',dict(status='SPECTRAL_EVALUATION',arm=arm,cases=index+1,total=630,pid=os.getpid(),time=time.time()))
        groups=[]
        for count in (1,2,3):
            part=[r for r in rows if r['count']==count];assert len(part)==210
            group=dict(count=count,cases=210)
            for key in KEYS:
                group['mean_'+key]=statistics.mean(v for row in part for v in row[key])
                group['weakest_'+key]=statistics.mean(row[key][row['weakest_index']] for row in part)
            groups.append(group)
        results.append(dict(arm=arm,checkpoint_sha256=w.digest(checkpoint),validation_sha256=w.digest(validation_path),
            rows=rows,by_count=groups,seconds=time.time()-started))
    w.write(root/'SPECTRAL.json',dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),arms=results,
        decomposition='abs(E-S)^2=(abs(E)-abs(S))^2+2*(abs(E)*abs(S)-Re(E conj(S)))',
        interpretation='Exact STFT-bin energy decomposition; not independent causal contributions and not numerically equal to sample NMSE under weighted overlapping STFT',
        assignment='Same whole-window waveform PIT as primary metrics; no per-bin oracle matching',
        heldout_read=False,independent_test=False))
