"""Audit the complete dual-axis model, initialization and ordinary updates."""
import argparse
import importlib.util
import math
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('axis_training_for_audit',HERE/'train.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
w=training.w


def audit(root):
    torch.set_num_threads(2)
    p=w.read(root/'PROTOCOL.json');training.verify(root,p);ph=w.digest(root/'PROTOCOL.json')
    complete=w.read(root/'COMPLETE.json');assert complete['protocol_sha256']==ph
    initial,ids=w.validate(root/'VALIDATION_000.json',None)
    parent,_=w.validate(Path(p['baseline']),ids)
    actual,_=w.validate(root/'VALIDATION_001.json',ids)
    control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',ids)
    assert initial==complete['initial'] and actual==complete['actual']
    for a,b in zip(initial['by_count'],parent['by_count']):
        for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
    selected=initial if initial['selection_nmse']<=actual['selection_nmse'] else actual
    assert selected==complete['selected']
    rows=training.base.compare(actual,parent,control);assert rows==complete['comparison']
    criterion=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))
    assert criterion==complete['criterion_met']
    saved=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    current=torch.load(root/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    best=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    assert all(v['protocol_sha256']==ph for v in (saved,current,best))
    assert saved['epoch']==current['epoch']==1 and saved['updates']==current['updates']==75
    assert best['best']==saved['best']==dict(epoch=selected['epoch'],metric=selected['selection_nmse'])
    expected=(training.make_model('retained_unet',Path(p['parent_checkpoint'])).state_dict()
              if selected['epoch']==0 else current['model'])
    assert current['model'].keys()==saved['model'].keys()==expected.keys()==best['model'].keys()
    for key,value in current['model'].items():
        assert torch.isfinite(value).all() and torch.equal(value,saved['model'][key])
        assert torch.equal(expected[key],best['model'][key])
    model=training.base.worker.make_model('retained_unet');model.load_state_dict(current['model'])
    params=list(model.parameters());assert sum(v.numel() for v in params)==37406475
    optimizer=saved['optimizer'];group,=optimizer['param_groups']
    assert group['lr']==1e-5 and group['weight_decay']==1e-4
    assert len(group['params'])==len(params)==len(optimizer['state'])
    assert {int(s['step']) for s in optimizer['state'].values()}=={75}
    for param,index in zip(params,group['params']):
        for key in ('exp_avg','exp_avg_sq'):
            value=optimizer['state'][index][key];assert value.shape==param.shape and torch.isfinite(value).all()
    stored=w.read(root/'GRADIENT_UPDATES.json');assert stored['protocol_sha256']==ph
    records=stored['updates'];assert len(records)==75
    total=np.zeros(3,dtype=int)
    for index,row in enumerate(records,1):
        assert row['update']==index and row['examples']==index*32 and sum(row['count_examples'])==32
        total+=row['count_examples'];a=np.array(row['gram']);one=np.ones(3)
        assert np.allclose(a,a.T) and np.linalg.eigvalsh(a).min()>-1e-5*max(float(np.trace(a)),1.)
        norm=np.sqrt(max(float(one@a@one),0.))
        assert math.isclose(norm,row['ordinary_norm'],rel_tol=2e-4,abs_tol=2e-5)
        assert row['ordinary_norm']==row['projected_norm'] and row['change_norm']==0.
        assert np.allclose(one@a,row['projected_direction_dot_original_tasks'],rtol=2e-4,atol=2e-5)
        assert math.isclose(row['projected_norm'],row['preclip_norm'],rel_tol=2e-4,abs_tol=2e-5)
        for i,dot in enumerate(row['original_gradient_dot_actual_delta']):
            assert abs(dot)<=np.sqrt(max(a[i,i],0.))*row['actual_adamw_delta_norm']+1e-5
    assert total.tolist()==[800,800,800]
    receipt=w.read(root/'EPOCH_001.json')
    assert receipt['validation']==actual and receipt['gradient_receipts_sha256']==w.digest(root/'GRADIENT_UPDATES.json')
    return dict(status='PASS',protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),criterion_met=criterion,selected_epoch=selected['epoch'],
        all_630_rows_reaggregated=True,optimizer_75_steps_and_moments_checked=True,
        actual_selected_tensors_checked=True,source_hashes_checked=True,
        ordinary_sample_mean_direction_checked=True,full_parameters=37406475,
        gradient_tensors_independently_recomputed=False,heldout_read=False,gpu_use=False,recorded_iq_reads=0,
        hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','VALIDATION_001.json','GRADIENT_UPDATES.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run.resolve());w.write(a.output,result);print(result,flush=True)
