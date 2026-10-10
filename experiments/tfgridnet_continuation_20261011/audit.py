"""Final checkpoint, parameter-update and numerical-receipt audit on CPU.

This does not rerun full-length CPU inference; that limitation is explicit.
"""
from pathlib import Path
import os
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tfgridnet_rf_20261011'))
import fit
from model import build
w=fit.w
RUN=ROOT/'local/tfgridnet_continuation_20261011_v1'
OUT=ROOT/'local/tfgridnet_continuation_audit_20261011_v1'


def main():
    OUT.mkdir(exist_ok=False);start=time.time()
    while not (RUN/'COMPLETE.json').exists():
        if (RUN/'FAILURE.json').exists():raise RuntimeError('Training failed')
        if time.time()-start>3600:raise TimeoutError('Run did not complete')
        w.write(OUT/'STATE.json',dict(status='WAITING_COMPLETION',pid=os.getpid(),time=time.time()))
        time.sleep(15)
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    w.write(OUT/'STATE.json',dict(status='CPU_AUDIT',pid=os.getpid(),time=time.time()))
    p=w.read(RUN/'PROTOCOL.json');fit.verify(RUN,p)
    result=w.read(RUN/'COMPLETE.json');ph=w.digest(RUN/'PROTOCOL.json');assert result['protocol_sha256']==ph
    ck=torch.load(RUN/'LAST.pt',map_location='cpu',weights_only=False)
    assert ck['updates']==result['updates'] and ck['protocol_sha256']==ph
    assert result['final']['checkpoint_sha256']==w.digest(RUN/'LAST.pt')
    net=build();initial=torch.load(ROOT/'local/tfgridnet_fit_20261011_v1/LAST.pt',map_location='cpu',weights_only=False)['model']
    net.load_state_dict(ck['model']);assert sum(t.numel() for t in net.parameters())==p['parameters']
    assert all(torch.isfinite(t).all() for t in net.parameters())
    assert {int(s['step']) for s in ck['optimizer']['state'].values()}=={result['updates']}
    for values in ck['optimizer']['state'].values():
        for value in values.values():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    changes={}
    for block in range(6):
        keys=[key for key in initial if key.startswith(f'blocks.{block}.')]
        changes[block]=sum(float((ck['model'][key].double()-initial[key].double()).square().sum()) for key in keys)**.5
        assert changes[block]>0
    maxsum=0.
    for h in result['history']:
        assert len(h['rows'])==4 and sum(len(x['nmse']) for x in h['rows'])==10
        assert [x['index'] for x in h['rows']]==p['train_indices']
        for row in h['rows']:
            assert np.isfinite(row['nmse']+row['si_sdr']).all()
            assert sorted(row['assignment'])==[0,1,2]
            maxsum=max(maxsum,row['sum_relative_error']);assert row['sum_relative_error']<1e-9
        for group in h['by_count']:
            rows=[x for x in h['rows'] if x['count']==group['count']]
            expect=dict(mean_nmse=np.mean([v for row in rows for v in row['nmse']]),
                mean_si_sdr=np.mean([v for row in rows for v in row['si_sdr']]),
                weakest_nmse=np.mean([row['nmse'][row['weakest_index']] for row in rows]),
                max_source_nmse=max(v for row in rows for v in row['nmse']))
            for key,value in expect.items():assert abs(group[key]-value)<1e-12
    selected=result['selected'];assert w.digest(Path(selected['checkpoint']))==selected['checkpoint_sha256']
    chosen=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    assert chosen['updates']==selected['step']
    assert {int(s['step']) for s in chosen['optimizer']['state'].values()}=={selected['step']}
    last_accepted=result['history'][0]
    for point in result['history'][1:]:
        passed=all(b['mean_nmse']<a['mean_nmse'] and b['mean_si_sdr']>a['mean_si_sdr'] and b['weakest_nmse']<=a['weakest_nmse']
            for a,b in zip(last_accepted['by_count'],point['by_count']))
        assert passed==point['continuation_pass']
        if passed:last_accepted=point
        else:assert point['step']==result['updates'] and result['reason'] in ('NO_JOINT_IMPROVEMENT_REVIEW','FIT_TARGET')
    assert selected['step']==last_accepted['step']
    fit.verify(RUN,p)
    record=dict(status='PASS',updates=result['updates'],parameters=p['parameters'],
        per_block_parameter_change_since_update32_l2=changes,all_receipt_summaries_recomputed=True,
        selected_update=selected['step'],selected_checkpoint_sha256=selected['checkpoint_sha256'],stop_decisions_recomputed=True,
        maximum_sum_relative_error=maxsum,checkpoint_sha256=w.digest(RUN/'LAST.pt'),
        protocol_sha256=ph,audit_source_sha256=w.digest(Path(__file__)),
        independent_full_length_cpu_inference=False,optimizer_updates=0,
        validation_read=False,heldout_read=False,time=time.time())
    w.write(OUT/'COMPLETE.json',record);w.write(ROOT/'reports/2026-10-11/TFGRIDNET_CONTINUATION_AUDIT.json',record)
    w.write(OUT/'STATE.json',dict(status='COMPLETED',pid=None,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        w.write(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
