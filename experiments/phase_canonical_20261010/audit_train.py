"""Read-only tensor/optimizer/630-row audit, independent of training loop."""
import argparse
from pathlib import Path
import torch
import train as worker

w=worker.w


def same(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def audit(root):
    torch.set_num_threads(2)
    p=w.read(root/'PROTOCOL.json');worker.verify(root,p)
    ph=w.digest(root/'PROTOCOL.json');complete=w.read(root/'COMPLETE.json')
    assert complete['status']=='COMPLETE' and complete['protocol_sha256']==ph
    parent,ids=w.validate(Path(p['baseline']),None)
    initial,_=w.validate(root/'VALIDATION_000.json',ids)
    actual,_=w.validate(root/'VALIDATION_001.json',ids)
    control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',ids)
    assert w.digest(root/'VALIDATION_000.json')==p['canonical_initial_sha256']
    chosen=min((initial,actual),key=lambda v:v['selection_nmse'])
    receipt=w.read(root/'EPOCH_001.json')
    assert receipt['updates']==75 and receipt['epoch']==1 and receipt['protocol_sha256']==ph
    assert len(receipt['preclip_gradient_norms'])==75
    assert receipt['validation']==actual and receipt['best']['epoch']==chosen['epoch']
    assert w.close(receipt['best']['metric'],chosen['selection_nmse'])
    assert complete['initial']==initial and complete['actual']==actual and complete['selected']==chosen
    model=worker.CanonicalPhaseSeparator(worker.worker.make_model('retained_unet',Path(p['parent_checkpoint'])))
    initial_state=model.state_dict()
    assert sum(q.numel() for q in model.parameters())==p['parameters']==32142859
    actual_saved=torch.load(root/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    last=torch.load(root/'LAST.pt',map_location='cpu',weights_only=False)
    best=torch.load(root/'BEST.pt',map_location='cpu',weights_only=False)
    for saved in (actual_saved,last):
        assert saved['updates']==75 and saved['epoch']==1 and saved['protocol_sha256']==ph
        assert saved['model'].keys()==initial_state.keys()
        assert all(v.shape==initial_state[k].shape and torch.isfinite(v).all() for k,v in saved['model'].items())
    assert same(actual_saved['model'],last['model'])
    assert best['protocol_sha256']==ph and best['best']['epoch']==chosen['epoch']
    assert same(best['model'],initial_state if chosen['epoch']==0 else actual_saved['model'])
    optimizer=last['optimizer'];parameters=list(model.parameters())
    assert len(optimizer['param_groups'])==1
    group=optimizer['param_groups'][0]
    assert group['lr']==1e-5 and group['weight_decay']==1e-4
    assert len(group['params'])==len(parameters)==len(optimizer['state'])
    assert {int(s['step']) for s in optimizer['state'].values()}=={75}
    for tensor,index in zip(parameters,group['params']):
        for name in ('exp_avg','exp_avg_sq'):
            value=optimizer['state'][index][name]
            assert value.shape==tensor.shape and torch.isfinite(value).all()
    comparisons=[]
    for scope,value in [('actual_e1',actual),('selected',chosen)]:
        for name,reference in [('parent',parent),('retained_control_e1',control)]:
            for count in (2,3):
                a,b=[next(g for g in v['by_count'] if g['count']==count) for v in (reference,value)]
                comparisons.append(dict(scope=scope,reference=name,count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
    assert comparisons==complete['comparison']
    passed=all(v['nmse_delta']<0 and v['si_sdr_delta']>0 and v['weakest_nmse_delta']<=0 for v in comparisons if v['scope']=='selected')
    assert passed==complete['criterion_met']
    return dict(status='PASS',study_protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),criterion_met=passed,selected_epoch=chosen['epoch'],
        all_630_rows_reaggregated=True,source_and_data_hashes_checked=True,
        actual_selected_tensors_checked=True,optimizer_75_steps_lr_and_moments_checked=True,
        hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','VALIDATION_000.json','VALIDATION_001.json')},
        model_updates=0,recorded_iq_reads=0,gpu_use=False,heldout_read=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=audit(a.study.resolve());w.write(a.output,result);print(result,flush=True)
