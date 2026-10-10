"""Conditional, full-capacity successive two-pass I/Q separation comparison."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from torch.utils.checkpoint import checkpoint
from successive import predict as successive_predict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from validation import validate
w=core.w;worker=core.base.worker
ARMS=('retained_two_pass_control','successive_pit')
PLAN=ROOT/'reports/2026-10-10/SUCCESSIVE_PLAN_KO.md'
CHECK=ROOT/'reports/2026-10-10/SUCCESSIVE_FULL_MODEL_CHECK.json'
ALGEBRA=ROOT/'reports/2026-10-10/SUCCESSIVE_ALGEBRA_CHECK.json'
CONTROL=ROOT/'reports/2026-10-10/SUCCESSIVE_CONTROL_CHECK.json'

def passed(rows):
    return all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))

def bounded_predict(net,item):
    def forward(mix,context,position):
        return worker.predict(net,dict(mixture=mix,context_features=context,crop_start=position))
    return checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)

def infer(net,item,arm):
    if arm==ARMS[0]:return worker.predict(net,item)
    if arm!=ARMS[1]:raise ValueError(arm)
    output,logits,_=successive_predict(net,item,worker.predict)
    return output,logits

def main_loss(output,logits,item):
    value=worker.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
    return value+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)

def accumulate(net,item,arm):
    if arm==ARMS[0]:
        total=0.
        for repeat in range(2):
            output,logits=bounded_predict(net,item);loss=main_loss(output,logits,item)
            assert torch.isfinite(loss);(.5*loss/32).backward();total+=.5*float(loss.detach())
        return total
    if arm!=ARMS[1]:raise ValueError(arm)
    output,logits,_=successive_predict(net,item,bounded_predict)
    loss=main_loss(output,logits,item);assert torch.isfinite(loss)
    (loss/32).backward();return float(loss.detach())

def register(root,predecessor):
    assert not (root/'PROTOCOL.json').exists()
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    checked=w.read(CHECK);assert checked['status']=='PASS' and checked['parameters']==32142859 and checked['optimizer_steps']==0
    cp=ROOT/'local/successive_full_check_20261010_v1/PROTOCOL.json'
    checked_protocol=w.read(cp);assert checked['protocol_sha256']==w.digest(cp)
    assert checked_protocol['parent_sha256']==old['parent_checkpoint_sha256']
    assert w.read(ALGEBRA)['status']=='PASS'
    control=w.read(CONTROL);assert control['status']=='PASS'
    assert w.read(ROOT/'reports/2026-10-10/PREDICTED_REMOVAL_AUDIT.json')['status']=='PASS'
    sources=dict(checked_protocol['sources']);sources.update(control['sources'])
    for name in ('train.py','validation.py','audit.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_CONDITIONAL_SUCCESSIVE_PIT',original_protocol=old,
        predecessor=str(predecessor),predecessor_protocol_sha256=w.digest(predecessor/'PROTOCOL.json'),
        source_sha256=sources,plan_sha256=w.digest(PLAN),check_sha256=w.digest(CHECK),algebra_sha256=w.digest(ALGEBRA),control_sha256=w.digest(CONTROL),
        arms=list(ARMS),parameters=32142859,updates_per_arm=75,examples_per_arm=2400,epochs_per_arm=1,
        seed=0,train_schedule_epoch=1,effective_batch=32,microbatch=1,lr=1e-5,weight_decay=1e-4,clip_norm=1.,
        precision='FP32 TF32off; non-reentrant activation checkpoint in both arms',
        forwards_per_example=2,backbone_passes_matched=True,
        control='Two identical original forwards, half loss each; repeated view adds no new example',
        candidate='Shared U-Net strongest predicted source, subtract, repeat, third output is residual; zero background',
        inference='No reference/count/category. Original mixture long context at both passes. Always3 outputs.',
        loss='Original final-output PIT NMSE+coherence+inactive/background +0.1 original-context count CE',
        selection='Minimum mean counts2/3 NMSE including each arm own e0; e0 functions differ',
        acceptance='Control vs parent; candidate vs BOTH parent and fresh control: lower2/3 NMSE, higher complex SI-SDR, nonincreased weakest NMSE',
        limitation='One seed reused DEV630, altered computation graph and background constraint; not OR-PIT reproduction or count-generalization proof',
        heldout_read=False,independent_test=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p);return p

def verify(root,p):
    for rel,sha in p['source_sha256'].items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    old=p['original_protocol']
    for path,key in ((Path(old['parent_checkpoint']),'parent_checkpoint_sha256'),
        (Path(old['preparation'])/'PREPARATION.json','preparation_sha256'),(Path(old['baseline']),'baseline_sha256')):
        assert w.digest(path)==old[key]
    for path,key in ((PLAN,'plan_sha256'),(CHECK,'check_sha256'),(ALGEBRA,'algebra_sha256'),(CONTROL,'control_sha256'),
        (Path(p['predecessor'])/'PROTOCOL.json','predecessor_protocol_sha256')):assert w.digest(path)==p[key]

def run(root,p,public):
    old=p['original_protocol'];ph=w.digest(root/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        pred=Path(p['predecessor']);audit=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_AUDIT.json'
        assert w.read(audit)['status']=='PASS' and w.read(audit)['complete_sha256']==w.digest(pred/'COMPLETE.json')
        w.write(root/'PREDECESSOR_RECEIPT.json',dict(complete_sha256=w.digest(pred/'COMPLETE.json'),audit_sha256=w.digest(audit)))
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(pred/'STATE.json')['pid'],old['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(old['preparation'],'train_pack',1);dev=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        parent,ids=w.validate(Path(old['baseline']),None);events=[]
        assert len(train)==2400 and len(dev)==630
        # GPU backward check uses the same TRAIN3 case checked on CPU; no step.
        net=worker.make_model('retained_unet',Path(old['parent_checkpoint'])).cuda().train()
        item=worker.fit.base.batch([train[2]]);torch.cuda.reset_peak_memory_stats()
        check_loss=accumulate(net,item,ARMS[1]);check_norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'),error_if_nonfinite=True))
        assert check_norm>0
        expected=next(r['objective'] for r in w.read(CHECK)['rows'] if r['index']==2)
        assert abs(check_loss-expected)<max(1e-3,abs(expected)*1e-4)
        w.write(root/'GPU_PREFLIGHT.json',dict(status='PASS',train_index=2,loss=check_loss,cpu_loss=expected,
            gradient_norm=check_norm,optimizer_steps=0,peak_bytes=torch.cuda.max_memory_allocated()))
        del net,item;gc.collect();torch.cuda.empty_cache()
        for number,arm in enumerate(ARMS):
            verify(root,p);folder=root/arm;folder.mkdir(exist_ok=True)
            net=worker.make_model('retained_unet',Path(old['parent_checkpoint'])).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
            predictor=lambda model,item:infer(model,item,arm)
            w.write(root/'STATE.json',dict(status='VALIDATING_INITIAL',arm=arm,pid=os.getpid(),time=time.time()))
            validate(net,dev,folder/'VALIDATION_000.json',0,predictor,worker)
            initial,_=w.validate(folder/'VALIDATION_000.json',ids)
            if number==0:
                for a,b in zip(parent['by_count'],initial['by_count']):
                    for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
            best=dict(epoch=0,metric=initial['selection_nmse'])
            worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True)
            norms=[];counts=[0,0,0];loss_sum=0.;started=time.time();torch.cuda.reset_peak_memory_stats()
            for index in range(len(train)):
                item=worker.fit.base.batch([train[index]]);loss_sum+=accumulate(net,item,arm)
                counts[int(item['construction_count'])-1]+=1
                if (index+1)%32==0:
                    assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in net.parameters())
                    norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)))
                    opt.step();opt.zero_grad(set_to_none=True)
                    w.write(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=1,updates=len(norms),
                        total_updates=number*75+len(norms),target_total_updates=150,examples=index+1,total_examples=2400,
                        seconds=time.time()-started,pid=os.getpid(),time=time.time()))
            elapsed=time.time()-started;assert len(norms)==75 and counts==[800,800,800]
            assert len(opt.state)==len(list(net.parameters())) and {int(q['step']) for q in opt.state.values()}=={75}
            assert all(torch.isfinite(q).all() for q in net.state_dict().values())
            w.write(root/'STATE.json',dict(status='VALIDATING',arm=arm,epoch=1,updates=75,pid=os.getpid(),time=time.time()))
            validate(net,dev,folder/'VALIDATION_001.json',1,predictor,worker);actual,_=w.validate(folder/'VALIDATION_001.json',ids)
            if actual['selection_nmse']<best['metric']:
                best=dict(epoch=1,metric=actual['selection_nmse'])
                worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            worker.atomic_torch(folder/'ACTUAL_001.pt',dict(model=net.state_dict(),arm=arm,epoch=1,updates=75,protocol_sha256=ph))
            worker.atomic_torch(folder/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),arm=arm,epoch=1,
                updates=75,best=best,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            comparisons=core.base.compare(actual,parent,parent if number==0 else events[0]['validation'])
            if number==0:comparisons=[r for r in comparisons if r['reference']=='parent']
            else:
                for row in comparisons:
                    if row['reference']=='retained_control_e1':row['reference']='matched_two_pass_control'
            event=dict(arm=arm,epoch=1,updates=75,protocol_sha256=ph,initial=initial,validation=actual,best=best,
                comparison=comparisons,criterion_met=passed(comparisons),train_seconds=elapsed,mean_training_loss=loss_sum/2400,
                preclip_gradient_norms=norms,count_examples=counts,peak_bytes=torch.cuda.max_memory_allocated())
            w.write(folder/'EPOCH_001.json',event);events.append(event)
            w.write(public,dict(status='PARTIAL',protocol_sha256=ph,events=events,heldout_read=False))
            print(dict(event='EPOCH_COMPLETE',**event),flush=True)
            del net,opt,item;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,events=events,total_updates=150,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','predecessor','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True);pred=a.predecessor.resolve()
    try:
        with (root/'.run.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);protocol=register(root,pred)
            w.write(root/'STATE.json',dict(status='WAITING_FOR_SOURCE_AFFINITY',pid=os.getpid(),time=time.time()))
            while not (pred/'COMPLETE.json').exists():
                if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                state=w.read(pred/'STATE.json')
                if state['status'].startswith('SKIPPED'):
                    w.write(root/'STATE.json',dict(status='SKIPPED_PREDECESSOR_NOT_NEEDED',pid=os.getpid(),time=time.time()));sys.exit(0)
                time.sleep(5)
            verify(root,protocol)
            if any(e['criterion_met'] for e in w.read(pred/'COMPLETE.json')['events']):
                w.write(root/'STATE.json',dict(status='SKIPPED_AFFINITY_MET_CRITERION',pid=os.getpid(),time=time.time()))
            else:
                while not (ROOT/'reports/2026-10-10/SOURCE_AFFINITY_AUDIT.json').exists():
                    if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor audit failed')
                    time.sleep(5)
                run(root,protocol,a.public.resolve());torch.cuda.empty_cache()
                w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=150,pid=os.getpid(),time=time.time()))
                subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,str(Path(__file__).with_name('audit.py')),
                    '--run',str(root),'--public',str(a.public.resolve().parent)],check=True)
                w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=150,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
