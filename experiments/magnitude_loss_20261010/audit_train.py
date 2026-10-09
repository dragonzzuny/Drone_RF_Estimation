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
    assert initial['epoch']==0
    for a,b in zip(initial['by_count'],parent['by_count']):
        for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
    chosen=min((initial,actual),key=lambda v:v['selection_nmse'])
    receipt=w.read(root/'EPOCH_001.json')
    assert receipt['updates']==75 and receipt['epoch']==1 and receipt['protocol_sha256']==ph
    assert len(receipt['preclip_gradient_norms'])==75
    assert receipt['validation']==actual and receipt['best']['epoch']==chosen['epoch']
    assert w.close(receipt['best']['metric'],chosen['selection_nmse'])
    assert complete['initial']==initial and complete['actual']==actual and complete['selected']==chosen
    model=worker.worker.make_model('retained_unet',Path(p['parent_checkpoint']))
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
    spectral_audit(root,p,ph)
    return dict(status='PASS',spectral_rows_reaggregated=True,spectral_error_identity_checked=True,study_protocol_sha256=ph,complete_sha256=w.digest(root/'COMPLETE.json'),
        auditor_sha256=w.digest(Path(__file__)),criterion_met=passed,selected_epoch=chosen['epoch'],
        all_630_rows_reaggregated=True,source_and_data_hashes_checked=True,
        actual_selected_tensors_checked=True,optimizer_75_steps_lr_and_moments_checked=True,
        hashes={name:w.digest(root/name) for name in ('ACTUAL_001.pt','LAST.pt','BEST.pt','VALIDATION_000.json','VALIDATION_001.json')},
        model_updates=0,recorded_iq_reads=0,gpu_use=False,heldout_read=False)


def spectral_audit(root,p,ph):
    import statistics
    value=w.read(root/'SPECTRAL.json')
    assert value['status']=='COMPLETE' and value['protocol_sha256']==ph
    complete=w.read(root/'COMPLETE.json')
    assert complete['spectral_sha256']==w.digest(root/'SPECTRAL.json')
    assert [a['arm'] for a in value['arms']]==['magnitude_e1','parent','retained_control_e1']
    keys=('relative_l1','magnitude_nmse','phase_interaction','spectral_complex_nmse','spectral_rms_ratio')
    for arm in value['arms']:
        if arm['arm']=='magnitude_e1':checkpoint=root/'ACTUAL_001.pt';val=root/'VALIDATION_001.json'
        elif arm['arm']=='parent':checkpoint=Path(p['parent_checkpoint']);val=root/'VALIDATION_000.json'
        else:checkpoint=Path(p['study'])/'retained_unet/ACTUAL_001.pt';val=Path(p['study'])/'retained_unet/VALIDATION_001.json'
        assert arm['checkpoint_sha256']==w.digest(checkpoint) and arm['validation_sha256']==w.digest(val)
        rows=arm['rows'];ref=w.read(val)['rows']
        assert len(rows)==630 and [r['index'] for r in rows]==list(range(630))
        for r,b in zip(rows,ref):
            for key in ('count','categories','pack_ids','nominal_levels_db','reference_power','weakest_index','assignment','si_sdr'):
                assert r[key]==b[key]
            assert len(r['nmse'])==r['count'] and max(abs(x-y) for x,y in zip(r['nmse'],b['nmse']))<2e-6
            for key in keys:assert len(r[key])==r['count']
            for mag,phase,total in zip(r['magnitude_nmse'],r['phase_interaction'],r['spectral_complex_nmse']):
                assert mag>=0 and phase>=-1e-10 and w.close(mag+phase,total)
        for count in (1,2,3):
            part=[r for r in rows if r['count']==count];assert len(part)==210
            g=next(v for v in arm['by_count'] if v['count']==count)
            for key in keys:
                assert w.close(g['mean_'+key],statistics.mean(x for r in part for x in r[key]))
                assert w.close(g['weakest_'+key],statistics.mean(r[key][r['weakest_index']] for r in part))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=audit(a.study.resolve());w.write(a.output,result);print(result,flush=True)
