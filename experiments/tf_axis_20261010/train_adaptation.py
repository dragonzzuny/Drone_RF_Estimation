"""Matched full-model new-layer learning rate, with/without backbone freezing."""
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
import train as first
from adaptation import configure,ARMS

ROOT=first.ROOT;w=first.w;base=first.base;worker=base.worker
PLAN=ROOT/'reports/2026-10-10/TF_AXIS_ADAPTATION_PLAN_KO.md'
CHECK=ROOT/'reports/2026-10-10/TF_AXIS_ADAPTATION_CPU_CHECK.json'


def register(root,pred):
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    predecessor=w.read(pred/'PROTOCOL.json');original=predecessor['original_protocol']
    first.verify(Path(predecessor['predecessor']),original)
    check=w.read(CHECK);assert check['status']=='PASS'
    audit=ROOT/'reports/2026-10-10/TF_AXIS_CONTINUATION_AUDIT.json'
    assert w.read(audit)['status']=='PASS' and w.read(audit)['complete_sha256']==w.digest(pred/'COMPLETE.json')
    sources=dict(original['source_sha256']);sources.update(check['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('audit_adaptation.py')):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_TF_AXIS_ADAPTATION',arms=list(ARMS),original_protocol=original,
        predecessor=str(pred),predecessor_protocol_sha256=w.digest(pred/'PROTOCOL.json'),
        predecessor_complete_sha256=w.digest(pred/'COMPLETE.json'),predecessor_audit_sha256=w.digest(audit),
        original_axis_root=predecessor['predecessor'],source_sha256=sources,
        plan_sha256=w.digest(PLAN),cpu_check_sha256=w.digest(CHECK),full_parameters=37406475,
        trainable_parameters={'joint_adapter_lr':37406475,'frozen_backbone':5263616},
        epochs_per_arm=1,updates_per_arm=75,total_updates=150,examples_per_arm=2400,
        effective_batch=32,microbatch=1,train_schedule_epoch=1,seed=0,
        adapter_lr=1e-4,backbone_lr=1e-5,weight_decay=1e-4,clip_norm=1.,precision='FP32 TF32off',
        loss=original['loss'],selection=original['selection'],acceptance=original['acceptance'],
        gradient_accumulation='Original microbatch order, not count-group order',
        limitation='One seed, repeatedly used DEV; same full inference model, different trainable parameter counts',
        heldout_read=False,independent_test=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    first.verify(Path(p['original_axis_root']),p['original_protocol'])
    for rel,sha in p['source_sha256'].items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    for name,key in [('PROTOCOL.json','predecessor_protocol_sha256'),('COMPLETE.json','predecessor_complete_sha256')]:
        assert w.digest(Path(p['predecessor'])/name)==p[key]
    assert w.digest(PLAN)==p['plan_sha256'] and w.digest(CHECK)==p['cpu_check_sha256']


def passed(rows):
    return all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))


def run(root,p,public):
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        old=p['original_protocol'];ph=w.digest(root/'PROTOCOL.json')
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['predecessor'])/'STATE.json')['pid'],old['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(old['preparation'],'train_pack',1)
        validation=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        assert len(train)==2400 and len(validation)==630
        parent,ids=w.validate(Path(old['baseline']),None)
        control,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
        events=[]
        for arm_index,arm in enumerate(ARMS):
            verify(root,p);folder=root/arm;folder.mkdir(exist_ok=True)
            net=first.make_model('retained_unet',Path(old['parent_checkpoint'])).cuda()
            opt,params=configure(net,arm)
            assert sum(q.numel() for q in net.parameters())==p['full_parameters']
            assert sum(q.numel() for q in params)==p['trainable_parameters'][arm]
            frozen={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('tf_axes.')} if arm=='frozen_backbone' else None
            w.write(root/'STATE.json',dict(status='VALIDATING_INITIAL',arm=arm,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,folder/'VALIDATION_000.json',0)
            initial,_=w.validate(folder/'VALIDATION_000.json',ids)
            for a,b in zip(parent['by_count'],initial['by_count']):
                for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
            best=dict(epoch=0,metric=initial['selection_nmse'])
            worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True);started=time.time();norms=[];loss_sum=0.;counts=[0,0,0]
            torch.cuda.reset_peak_memory_stats()
            for index in range(len(train)):
                item=worker.fit.base.batch([train[index]]);estimates,logits=worker.predict(net,item)
                loss=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
                loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                assert torch.isfinite(loss);(loss/32).backward();loss_sum+=float(loss.detach())
                counts[int(item['construction_count'])-1]+=1
                if (index+1)%32==0:
                    assert all(q.grad is not None for q in params)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);norms.append(float(norm))
                    opt.step();opt.zero_grad(set_to_none=True)
                    w.write(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=1,updates=len(norms),
                        total_updates=arm_index*75+len(norms),target_total_updates=150,examples=index+1,total=2400,
                        seconds=time.time()-started,pid=os.getpid(),time=time.time()))
            elapsed=time.time()-started;assert len(norms)==75 and counts==[800,800,800]
            assert {int(s['step']) for s in opt.state.values()}=={75} and len(opt.state)==len(params)
            assert all(torch.isfinite(v).all() for v in net.state_dict().values())
            if frozen is not None:
                for key,value in frozen.items():assert torch.equal(net.state_dict()[key].cpu(),value)
            w.write(root/'STATE.json',dict(status='VALIDATING',arm=arm,epoch=1,updates=75,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,folder/'VALIDATION_001.json',1)
            actual,_=w.validate(folder/'VALIDATION_001.json',ids)
            if frozen is not None:
                before=w.read(folder/'VALIDATION_000.json')['rows'];after=w.read(folder/'VALIDATION_001.json')['rows']
                assert all(a['predicted_count']==b['predicted_count'] for a,b in zip(before,after))
            if actual['selection_nmse']<best['metric']:
                best=dict(epoch=1,metric=actual['selection_nmse'])
                worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            worker.atomic_torch(folder/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,arm=arm,protocol_sha256=ph))
            worker.atomic_torch(folder/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,updates=75,
                best=best,arm=arm,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            comparison=base.compare(actual,parent,control)
            event=dict(arm=arm,epoch=1,updates=75,protocol_sha256=ph,validation=actual,best=best,
                comparison=comparison,criterion_met=passed(comparison),train_seconds=elapsed,mean_training_loss=loss_sum/2400,
                preclip_gradient_norms=norms,peak_bytes=torch.cuda.max_memory_allocated(),count_examples=counts,
                frozen_backbone_checked=frozen is not None)
            w.write(folder/'EPOCH_001.json',event);events.append(event)
            w.write(public,dict(status='PARTIAL',protocol_sha256=ph,events=events,heldout_read=False))
            print(dict(event='EPOCH_COMPLETE',**event),flush=True)
            del net,opt,params,frozen,item,estimates,logits,loss,norm;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,events=events,total_updates=150,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',updates=150,pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','predecessor','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True);pred=a.predecessor.resolve()
    try:
        with (root/'.run.lock').open('a') as own_lock:
            fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            w.write(root/'STATE.json',dict(status='WAITING_FOR_FULL_TF_AXIS_COMPARISON',pid=os.getpid(),time=time.time()))
            while not (pred/'COMPLETE.json').exists():
                if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                time.sleep(5)
            result=w.read(pred/'COMPLETE.json');first_root=Path(w.read(pred/'PROTOCOL.json')['predecessor'])
            criteria=[w.read(first_root/'COMPLETE.json')['criterion_met']]+[e['criterion_met'] for e in result['events']]
            if any(criteria):w.write(root/'STATE.json',dict(status='SKIPPED_PRIOR_STRUCTURE_MET_CRITERION',pid=os.getpid(),time=time.time()))
            else:
                audit=ROOT/'reports/2026-10-10/TF_AXIS_CONTINUATION_AUDIT.json'
                while not audit.exists():
                    if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor audit failed')
                    time.sleep(5)
                p=register(root,pred);run(root,p,a.public.resolve());torch.cuda.empty_cache()
                w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=150,pid=os.getpid(),time=time.time()))
                subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,
                    str(Path(__file__).with_name('audit_adaptation.py')),'--run',str(root),'--public',str(a.public.resolve().parent)],check=True)
                w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=150,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
