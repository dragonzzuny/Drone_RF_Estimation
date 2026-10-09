"""Independent CPU reaggregation, checkpoint checks and Gram-space replay."""
import argparse
import importlib.util
from pathlib import Path
import math
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('count_pcgrad_training',HERE/'train.py')
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
    rows=training.compare(actual,parent,control);assert rows==complete['comparison']
    criterion=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))
    assert criterion==complete['criterion_met']
    saved=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    current=torch.load(root/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    best=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    assert all(v['protocol_sha256']==ph for v in (saved,current,best))
    assert saved['epoch']==current['epoch']==1 and saved['updates']==current['updates']==75
    assert best['best']==saved['best']==dict(epoch=selected['epoch'],metric=selected['selection_nmse'])
    expected=(torch.load(p['parent_checkpoint'],map_location='cpu',weights_only=False)['model']
              if selected['epoch']==0 else current['model'])
    assert current['model'].keys()==saved['model'].keys()==expected.keys()==best['model'].keys()
    for key,value in current['model'].items():
        assert torch.isfinite(value).all() and torch.equal(value,saved['model'][key])
        assert torch.equal(expected[key],best['model'][key])
    model=training.worker.make_model('retained_unet');model.load_state_dict(current['model'])
    params=list(model.parameters());assert sum(v.numel() for v in params)==32142859
    optimizer=saved['optimizer'];group,=optimizer['param_groups']
    assert group['lr']==1e-5 and group['weight_decay']==1e-4
    assert len(group['params'])==len(params)==len(optimizer['state'])
    assert {int(s['step']) for s in optimizer['state'].values()}=={75}
    for param,index in zip(params,group['params']):
        for key in ('exp_avg','exp_avg_sq'):
            v=optimizer['state'][index][key];assert v.shape==param.shape and torch.isfinite(v).all()
    records=w.read(root/'GRADIENT_UPDATES.json');assert records['protocol_sha256']==ph
    records=records['updates'];assert len(records)==75
    count_total=np.zeros(3,dtype=int);conflicts=np.zeros((3,3),dtype=int);surgery_batches=0
    for number,r in enumerate(records,1):
        assert r['update']==number and r['examples']==number*32 and sum(r['count_examples'])==32
        count_total+=r['count_examples']
        gram=np.array(r['gram'],dtype=np.float64)
        assert np.isfinite(gram).all() and np.allclose(gram,gram.T,atol=1e-6,rtol=1e-5)
        assert np.linalg.eigvalsh(gram).min()>-1e-5*max(np.trace(gram),1.)
        conflicts+=(gram<0).astype(int)
        c=np.eye(3);events=[]
        for i,order in enumerate(r['projection_orders']):
            assert sorted(order)==[j for j in range(3) if i!=j]
            for j in order:
                dot=c[i]@gram[:,j]
                if dot<0 and gram[j,j]>0:
                    value=dot/gram[j,j];c[i,j]-=value;events.append((i+1,j+1,value))
        assert len(events)==len(r['projections'])
        for (i,j,value),event in zip(events,r['projections']):
            assert (i,j)==(event['task'],event['reference'])
            assert math.isclose(value,event['coefficient'],rel_tol=2e-4,abs_tol=2e-5)
        combined=c.sum(0);change=combined-np.ones(3)
        for key,vector in [('ordinary_norm',np.ones(3)),('projected_norm',combined),('change_norm',change)]:
            norm=math.sqrt(max(float(vector@gram@vector),0.))
            assert math.isclose(norm,r[key],rel_tol=2e-4,abs_tol=2e-5),(number,key,norm,r[key])
        assert np.allclose(combined@gram,r['projected_direction_dot_original_tasks'],rtol=2e-4,atol=2e-5)
        assert math.isclose(r['projected_norm'],r['preclip_norm'],rel_tol=2e-4,abs_tol=2e-5)
        for i,dot in enumerate(r['original_gradient_dot_actual_delta']):
            assert abs(dot)<=math.sqrt(max(gram[i,i],0.))*r['actual_adamw_delta_norm']+1e-5
        surgery_batches+=bool(r['projections'])
    assert count_total.tolist()==[800,800,800]
    receipt=w.read(root/'EPOCH_001.json')
    assert receipt['validation']==actual and receipt['gradient_receipts_sha256']==w.digest(root/'GRADIENT_UPDATES.json')
    return dict(status='PASS',criterion_met=criterion,selected_epoch=selected['epoch'],
        protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),auditor_sha256=w.digest(Path(__file__)),
        all_630_rows_reaggregated=True,optimizer_75_steps_and_moments_checked=True,
        actual_selected_tensors_checked=True,source_hashes_checked=True,gram_space_projection_replay=True,
        gradient_tensors_independently_recomputed=False,surgery_batches=surgery_batches,
        negative_gram_batches=conflicts.tolist(),count_examples=count_total.tolist(),
        heldout_read=False,gpu_use=False,recorded_iq_reads=0,
        hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','VALIDATION_001.json','GRADIENT_UPDATES.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run.resolve());w.write(a.output,result);print(result,flush=True)
