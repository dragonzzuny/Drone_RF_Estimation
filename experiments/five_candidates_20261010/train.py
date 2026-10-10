"""Sequential continuation worker; full models, exact saved optimizer resume."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from workflow import ROOT, RUN, SPECS, MILESTONES, read, write, digest, verify, state, schedule
from factory import build, restore, equal, predict_train, guard_components, module, worker

w = worker.watch
validation = module('five_validation', 'experiments/recursive_20261010/validation.py')


def monitor_update(opt, monitor, epoch, metric):
    if epoch >= 20:
        if monitor['best'] is None or metric < monitor['best'] - 1e-5:
            monitor.update(best=metric, bad=0)
        else:
            monitor['bad'] += 1
            if monitor['bad'] >= 10:
                for group in opt.param_groups: group['lr'] *= .5
                monitor['bad'] = 0; monitor['reductions'] += 1


def precheck(root, name):
    p = read(root/'PROTOCOL.json'); verify(root,p)
    spec = SPECS[name]; folder = root/name
    torch.set_num_threads(2); torch.manual_seed(0)
    if spec['origin']:
        net,opt,params,meta = restore(name,folder/'RESUME_INPUT.pt',1,'cpu')
        saved = torch.load(folder/'RESUME_INPUT.pt',map_location='cpu',weights_only=False,mmap=True)
        assert meta['protocol_sha256'] == p['candidates'][name]['origin_protocol_sha256']
        parent,ids = w.validate(Path(p['baseline']),None)
        actual,_ = w.validate(folder/'VALIDATION_001.json',ids)
        event = read(folder/'EPOCH_001.json')
        assert actual == event['validation'] and event['best'] == meta['best']
        best = torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False,mmap=True)
        assert best['best'] == meta['best']
        if best['best']['epoch'] == 1: assert equal(best['model'],saved['model'])
        restored = dict(exact_model_optimizer_cpu_rng=True,epoch=1,updates=75,
                        cuda_rng_present=bool(saved['cuda_rng']),origin_protocol=meta['protocol_sha256'])
    else:
        net,opt,params = build(name,p['parent_checkpoint'])
        parent = torch.load(p['parent_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        for key,value in parent['model'].items():
            mapped = 'output.base.'+key[len('output.'):] if key.startswith('output.') else key
            assert equal(net.state_dict()[mapped],value),key
        assert torch.count_nonzero(net.output.readout.weight) == 0
        assert len(opt.state) == 0
        restored = dict(epoch=0,updates=0,exact_parent_tensors=True,zero_new_readout=True)
    histogram = {s:sum(schedule(name,e)==s for e in range(1,51)) for s in range(1,6)}
    assert set(histogram.values()) == {10}
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS',candidate=name,restore=restored,
        parameters=sum(x.numel() for x in net.parameters()),trainable=sum(x.numel() for x in params),
        optimizer_groups=[dict(lr=g['lr'],weight_decay=g['weight_decay'],parameters=sum(x.numel() for x in g['params'])) for g in opt.param_groups],
        schedule_histogram=histogram,protocol_sha256=digest(root/'PROTOCOL.json'),
        full_model=True,gpu_initialized=False,optimizer_steps=0,recorded_iq_reads=0,heldout_read=False,
        previous_full_forward_checks=p['candidates'][name]['checks'],time=time.time())
    write(folder/'PREFLIGHT.json',result);print(result,flush=True)


def check_frozen(folder, net):
    original = torch.load(folder/'RESUME_INPUT.pt',map_location='cpu',weights_only=False,mmap=True)
    for key,value in original['model'].items():
        if not key.startswith('tf_axes.'): assert equal(net.state_dict()[key],value),key


def train_epoch(root,p,name,epoch,dev,ids):
    folder=root/name; ph=digest(root/'PROTOCOL.json'); previous=epoch-1
    if epoch == 1:
        assert name=='balanced_head'
        torch.manual_seed(0);net,opt,params=build(name,p['parent_checkpoint'],'cuda')
        state(root, status='VALIDATING_INITIAL',candidate=name,epoch=0)
        validation.validate(net,dev,folder/'VALIDATION_000.json',0,worker.predict,worker)
        initial,_=w.validate(folder/'VALIDATION_000.json',ids)
        parent,_=w.validate(Path(p['baseline']),ids)
        for a,b in zip(initial['by_count'],parent['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'): assert abs(a[key]-b[key])<2e-6
        best=dict(epoch=0,metric=initial['selection_nmse'])
        worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        meta=dict(best=best,lr_monitor=dict(best=None,bad=0,reductions=0),projection_rng=None)
        torch.manual_seed(0)
        provenance=dict(parent_sha256=p['parent_checkpoint_sha256'],fresh_optimizer=True)
    else:
        last=folder/'LAST.pt' if (folder/'LAST.pt').exists() else folder/'RESUME_INPUT.pt'
        net,opt,params,meta=restore(name,last,previous,'cuda')
        if last.name=='LAST.pt':
            event=read(folder/f'EPOCH_{previous:03d}.json')
            assert digest(last)==event['last_sha256'] and meta['protocol_sha256']==ph
        else:
            assert meta['protocol_sha256']==p['candidates'][name]['origin_protocol_sha256']
        saved_best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved_best['best']==meta['best'];del saved_best
        provenance=dict(input_sha256=digest(last),previous_epoch=previous,
                        exact_model_optimizer_rng=True,source_protocol=meta['protocol_sha256'])
    best,monitor=meta['best'],meta['lr_monitor']
    train=worker.NativeMixtures(p['preparation'],'train_pack',schedule(name,epoch))
    assert len(train)==2400
    guard=None
    if name=='wave_guard':
        Accumulator,correct=guard_components();guard=Accumulator(net.named_parameters())
    net.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    started=time.time();norms=[];counts=[0,0,0];loss_sum=0.;guard_receipts=[]
    for index in range(2400):
        item=worker.fit.base.batch([train[index]])
        output,logits=predict_train(name,net,item)
        loss=worker.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
        ce=.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        loss=loss+ce;assert torch.isfinite(loss)
        count=int(item['construction_count']);counts[count-1]+=1
        if guard is not None and count in (2,3):
            grads=torch.autograd.grad(ce/32,guard.ce_parameters,retain_graph=True,allow_unused=True)
            guard.collect_ce(count,grads);del grads
        (loss/32).backward();loss_sum+=float(loss.detach())
        if guard is not None: guard.collect(count)
        if (index+1)%32==0:
            if guard is not None:
                flat=guard.groups.sum(0);guard.assign(flat)
                protected=guard.waveform_groups()
            assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in params)
            norms.append(float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)))
            if guard is not None: before=torch.cat([q.detach().flatten() for q in params])
            opt.step()
            if guard is not None:
                delta,detail=correct(params,before,protected)
                guard_receipts.append(dict(update=len(norms),count_examples=list(guard.counts),**detail))
                guard.reset();del flat,protected,before,delta
            opt.zero_grad(set_to_none=True)
            state(root,status='TRAINING',candidate=name,epoch=epoch,schedule_epoch=train.epoch,
                  updates_in_epoch=len(norms),cumulative_updates=75*previous+len(norms),examples=index+1,
                  mean_training_loss_so_far=loss_sum/(index+1),seconds=time.time()-started)
    elapsed=time.time()-started
    assert len(norms)==75 and counts==[800,800,800]
    assert {int(v['step']) for v in opt.state.values()}=={75*epoch}
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    if name=='frozen_tf': check_frozen(folder,net)
    if guard is not None:
        write(folder/f'GUARD_{epoch:03d}.json',dict(updates=guard_receipts,protocol_sha256=ph))
        del guard
    state(root,status='VALIDATING',candidate=name,epoch=epoch,updates=75*epoch)
    vp=folder/f'VALIDATION_{epoch:03d}.json'
    validation.validate(net,dev,vp,epoch,worker.predict,worker)
    actual,_=w.validate(vp,ids)
    if name=='frozen_tf':
        before_rows=read(folder/'VALIDATION_000.json')['rows']; after_rows=read(vp)['rows']
        assert all(a['predicted_count']==b['predicted_count'] for a,b in zip(before_rows,after_rows))
    if actual['selection_nmse']<best['metric']:
        best=dict(epoch=epoch,metric=actual['selection_nmse'])
        worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
    used=[g['lr'] for g in opt.param_groups];monitor_update(opt,monitor,epoch,actual['selection_nmse'])
    actual_path=folder/'LATEST_ACTUAL.pt'
    worker.atomic_torch(actual_path,dict(model=net.state_dict(),epoch=epoch,updates=75*epoch,protocol_sha256=ph))
    worker.atomic_torch(folder/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
        updates=75*epoch,best=best,lr_monitor=monitor,protocol_sha256=ph,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),projection_rng=meta.get('projection_rng')))
    saved=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False,mmap=True)
    assert equal(saved['model'],net.state_dict()) and equal(saved['optimizer'],opt.state_dict())
    assert equal(saved['torch_rng'],torch.get_rng_state()) and equal(saved['cuda_rng'],torch.cuda.get_rng_state_all())
    del saved
    if epoch in MILESTONES:
        destination=folder/f'ACTUAL_{epoch:03d}.pt'
        assert not destination.exists();os.link(actual_path,destination)
    event=dict(candidate=name,epoch=epoch,updates=75*epoch,protocol_sha256=ph,
        validation=actual,best=best,schedule_epoch=train.epoch,rows_sha256=train.rows_hash,
        mean_training_loss=loss_sum/2400,count_examples=counts,train_seconds=elapsed,
        peak_bytes=torch.cuda.max_memory_allocated(),preclip_norm_mean=sum(norms)/75,
        preclip_norm_max=max(norms),used_learning_rates=used,next_learning_rates=[g['lr'] for g in opt.param_groups],
        lr_monitor=monitor,resume_provenance=provenance,last_sha256=digest(folder/'LAST.pt'),
        actual_weights_sha256=digest(actual_path),best_weights_sha256=digest(folder/'BEST.pt'),
        validation_sha256=digest(vp),exact_roundtrip=True,heldout_read=False)
    write(folder/f'EPOCH_{epoch:03d}.json',event)
    print(dict(event='EPOCH_COMPLETE',**event),flush=True)


def audit(root,p,name):
    folder=root/name;_,ids=w.validate(Path(p['baseline']),None)
    epochs=sorted(int(f.stem.split('_')[1]) for f in folder.glob('EPOCH_*.json'))
    assert epochs==list(range(1,51))
    hist={i:0 for i in range(1,6)};events=[]
    for epoch in epochs:
        event=read(folder/f'EPOCH_{epoch:03d}.json')
        actual,_=w.validate(folder/f'VALIDATION_{epoch:03d}.json',ids)
        assert actual==event['validation'] and event['updates']==75*epoch
        expected=schedule(name,epoch);hist[expected]+=1
        if not event.get('imported'):
            assert event['schedule_epoch']==expected and event['protocol_sha256']==digest(root/'PROTOCOL.json')
            assert event['validation_sha256']==digest(folder/f'VALIDATION_{epoch:03d}.json')
        events.append(event)
    assert set(hist.values())=={10}
    last=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False,mmap=True)
    assert last['epoch']==50 and last['updates']==3750 and digest(folder/'LAST.pt')==events[-1]['last_sha256']
    actual=torch.load(folder/'LATEST_ACTUAL.pt',map_location='cpu',weights_only=False,mmap=True)
    assert equal(actual['model'],last['model'])
    best=torch.load(folder/'BEST.pt',map_location='cpu',weights_only=False,mmap=True)
    assert best['best']==last['best'] and digest(folder/'BEST.pt')==events[-1]['best_weights_sha256']
    initial,_=w.validate(folder/'VALIDATION_000.json',ids)
    expected=min([(0,initial['selection_nmse'])]+[(e['epoch'],e['validation']['selection_nmse']) for e in events],key=lambda x:x[1])
    assert best['best']==dict(epoch=expected[0],metric=expected[1])
    verify(root,p)
    result=dict(status='BUDGET_COMPLETE_AUDITED',candidate=name,epochs=50,updates=3750,
        validation_rows_checked=50*630,schedule_histogram=hist,best=last['best'],
        last_sha256=digest(folder/'LAST.pt'),protocol_sha256=digest(root/'PROTOCOL.json'),
        heldout_read=False,convergence_claimed=False,time=time.time())
    write(folder/'COMPLETE.json',result)


def run(root,name):
    p=read(root/'PROTOCOL.json');folder=root/name
    assert read(folder/'PREFLIGHT.json')['status']=='PASS'
    with (folder/'.run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with worker.fit.base.LOCK.open('r') as lock:
            state(root,status='WAITING_GPU_LOCK',candidate=name)
            fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
            assert torch.cuda.is_available()
            write(folder/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(-1,p['gpu_display_policy']))
            torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            epochs=sorted(int(f.stem.split('_')[1]) for f in folder.glob('EPOCH_*.json'))
            assert epochs==list(range(1,len(epochs)+1))
            _,ids=w.validate(Path(p['baseline']),None)
            dev=worker.NativeMixtures(p['preparation'],'validation_pack',1);assert len(dev)==630
            for epoch in range(len(epochs)+1,51):
                verify(root,p);assert shutil.disk_usage(root).free>2*1024**3,'Insufficient checkpoint disk space'
                train_epoch(root,p,name,epoch,dev,ids)
                gc.collect();torch.cuda.empty_cache()
            audit(root,p,name)
            state(root,status='CANDIDATE_BUDGET_COMPLETE',candidate=name,epochs=50)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--candidate',choices=list(SPECS),required=True)
    parser.add_argument('--precheck',action='store_true');a=parser.parse_args();root=a.run.resolve()
    try:
        precheck(root,a.candidate) if a.precheck else run(root,a.candidate)
    except Exception:
        write(root/a.candidate/('PREFLIGHT_FAILURE.json' if a.precheck else 'FAILURE.json'),
              dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise
