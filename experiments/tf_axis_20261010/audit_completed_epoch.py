"""Audit a saved continuation epoch while the next epoch is training.

LAST must still refer to that epoch; fail if a later completion replaces it.
"""
import argparse
from pathlib import Path
import torch
import continue_training as training

w=training.w


def audit(root,epoch):
    torch.set_num_threads(2);p=w.read(root/'PROTOCOL.json');training.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');old=p['original_protocol']
    parent,ids=w.validate(Path(old['baseline']),None)
    actual,_=w.validate(root/f'VALIDATION_{epoch:03d}.json',ids)
    control,_=w.validate(Path(old['study'])/f'retained_unet/VALIDATION_{epoch:03d}.json',ids)
    event=w.read(root/f'EPOCH_{epoch:03d}.json')
    assert event['epoch']==epoch and event['updates']==75*epoch and event['protocol_sha256']==ph
    assert event['validation']==actual
    compared=training.base.compare(actual,parent,control)
    for row in compared:
        if row['reference']=='retained_control_e1':row['reference']=f'retained_control_e{epoch}'
    assert compared==event['comparison']
    passed=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in compared if r['count'] in (2,3))
    assert passed==event['criterion_met']
    names=[f'ACTUAL_{epoch:03d}.pt','LAST.pt',f'VALIDATION_{epoch:03d}.json',f'GRADIENT_UPDATES_{epoch:03d}.json']
    hashes={name:w.digest(root/name) for name in names}
    actual_state=torch.load(root/names[0],map_location='cpu',weights_only=False)
    last=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    assert actual_state['epoch']==last['epoch']==epoch and actual_state['updates']==last['updates']==75*epoch
    assert actual_state['protocol_sha256']==last['protocol_sha256']==ph
    assert actual_state['model'].keys()==last['model'].keys()
    for key,value in actual_state['model'].items():assert torch.isfinite(value).all() and torch.equal(value,last['model'][key])
    net=training.first.make_model('retained_unet');net.load_state_dict(actual_state['model']);params=list(net.parameters())
    assert sum(q.numel() for q in params)==37406475
    group,=last['optimizer']['param_groups'];states=last['optimizer']['state']
    assert group['lr']==1e-5 and group['weight_decay']==1e-4
    assert len(group['params'])==len(params)==len(states) and {int(v['step']) for v in states.values()}=={75*epoch}
    for param,key in zip(params,group['params']):
        for name in ('exp_avg','exp_avg_sq'):
            value=states[key][name];assert value.shape==param.shape and torch.isfinite(value).all()
    receipts=w.read(root/f'GRADIENT_UPDATES_{epoch:03d}.json')
    assert receipts['protocol_sha256']==ph and len(receipts['updates'])==75
    assert event['gradient_receipts_sha256']==hashes[f'GRADIENT_UPDATES_{epoch:03d}.json']
    for name,sha in hashes.items():assert w.digest(root/name)==sha,'Checkpoint changed during audit'
    return dict(status='PASS',epoch=epoch,updates=epoch*75,protocol_sha256=ph,
        auditor_sha256=w.digest(Path(__file__)),actual=actual,comparison=compared,criterion_met=passed,
        all_630_rows_reaggregated=True,actual_and_last_tensors_equal=True,optimizer_steps_and_moments_checked=True,
        source_hashes_checked=True,hashes=hashes,heldout_read=False,recorded_iq_reads=0,gpu_use=False,
        limitation='Epoch receipt audit; final best-selection and full direction algebra are checked after e3')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.run.resolve(),a.epoch);w.write(a.output,result);print(result,flush=True)
