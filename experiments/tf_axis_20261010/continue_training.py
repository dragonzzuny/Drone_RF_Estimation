"""Continue the actual dual-axis e1 to e3 with unchanged optimizer and budget.

Separate registration: the original e1 protocol and outputs remain immutable.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
import train as first

base=first.base;worker=base.worker;w=first.w;ROOT=first.ROOT
PLAN=ROOT/'reports/2026-10-10/TF_AXIS_CONTINUATION_PLAN_KO.md'


def register(root,predecessor):
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    original=w.read(predecessor/'PROTOCOL.json');first.verify(predecessor,original)
    audit_path=ROOT/'reports/2026-10-10/TF_AXIS_FINAL_AUDIT.json';audit=w.read(audit_path)
    assert audit['status']=='PASS' and audit['complete_sha256']==w.digest(predecessor/'COMPLETE.json')
    hashes=dict(original['source_sha256'])
    for file in (Path(__file__),Path(__file__).with_name('audit_continuation.py')):
        hashes[str(file.relative_to(ROOT))]=w.digest(file)
    p=dict(status='REGISTERED_TF_AXIS_CONTINUATION',predecessor=str(predecessor),
        predecessor_protocol_sha256=w.digest(predecessor/'PROTOCOL.json'),
        predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        start_checkpoint_sha256=w.digest(predecessor/'LAST.pt'),start_best_sha256=w.digest(predecessor/'BEST.pt'),
        predecessor_audit_sha256=w.digest(audit_path),source_sha256=hashes,plan_sha256=w.digest(PLAN),
        parameters=37406475,additional_epochs=[2,3],additional_updates=150,total_updates=225,
        loss=original['loss'],lr=original['lr'],weight_decay=original['weight_decay'],clip_norm=1.,
        effective_batch=32,examples_per_epoch=2400,seed=0,precision='FP32 TF32off',
        selection=original['selection'],acceptance='Both counts2/3 NMSE down, SI-SDR up, weakest NMSE nonincreased versus parent and same-epoch original control',
        original_protocol=original,heldout_read=False,independent_test=False,
        control_validation_sha256={str(e):w.digest(Path(original['study'])/f'retained_unet/VALIDATION_{e:03d}.json') for e in (2,3)},
        continuation_rule='Fixed e2/e3 continuation of actual e1 regardless of e1 development outcome; no optimizer reset',
        registered_at=time.time())
    for rel,sha in hashes.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    pred=Path(p['predecessor']);first.verify(pred,p['original_protocol'])
    for rel,sha in p['source_sha256'].items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    for name,key in [('PROTOCOL.json','predecessor_protocol_sha256'),('COMPLETE.json','predecessor_complete_sha256'),
        ('LAST.pt','start_checkpoint_sha256'),('BEST.pt','start_best_sha256')]:assert w.digest(pred/name)==p[key]
    assert w.digest(PLAN)==p['plan_sha256']
    for epoch,sha in p['control_validation_sha256'].items():
        assert w.digest(Path(p['original_protocol']['study'])/f'retained_unet/VALIDATION_{int(epoch):03d}.json')==sha


def run(root,p,public):
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        original=p['original_protocol'];pred=Path(p['predecessor']);ph=w.digest(root/'PROTOCOL.json')
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(pred/'STATE.json')['pid'],original['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        parent,identities=w.validate(Path(original['baseline']),None)
        validation=worker.NativeMixtures(original['preparation'],'validation_pack',1)
        saved=torch.load(pred/'LAST.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==p['predecessor_protocol_sha256'] and saved['epoch']==1 and saved['updates']==75
        net=first.make_model('retained_unet').cuda();net.load_state_dict(saved['model'])
        opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
        opt.load_state_dict(saved['optimizer'])
        for key,value in net.state_dict().items():assert torch.equal(value.cpu(),saved['model'][key])
        for key,state in opt.state_dict()['state'].items():
            for name,value in state.items():
                if torch.is_tensor(value):assert torch.equal(value.cpu(),saved['optimizer']['state'][key][name])
        assert {int(s['step']) for s in opt.state.values()}=={75}
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
        rng=base.seeded_rng(0);rng.setstate(saved['projection_rng']);best=saved['best'];del saved
        old_best=torch.load(pred/'BEST.pt',map_location='cpu',weights_only=False)
        assert old_best['best']==best and old_best['protocol_sha256']==p['predecessor_protocol_sha256']
        old_best['origin_protocol_sha256']=old_best['protocol_sha256'];old_best['protocol_sha256']=ph
        worker.atomic_torch(root/'BEST.pt',old_best);del old_best
        w.write(root/'RESUME_CHECK.json',dict(status='PASS',actual_e1_not_selected_e0=True,
            all_model_tensors_and_optimizer_state_equal=True,torch_and_cuda_rng_equal=True,
            optimizer_steps=75,source_checkpoint_sha256=p['start_checkpoint_sha256']))
        events=[]
        for epoch in (2,3):
            verify(root,p);train=worker.NativeMixtures(original['preparation'],'train_pack',epoch)
            assert len(train)==2400
            accumulator=base.CountAccumulator(net.parameters());net.train();opt.zero_grad(set_to_none=True)
            started=time.time();updates=[];loss_sum=0.;counts=[0,0,0]
            torch.cuda.reset_peak_memory_stats()
            for index in range(len(train)):
                item=worker.fit.base.batch([train[index]])
                estimates,logits=worker.predict(net,item)
                loss=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
                loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                assert torch.isfinite(loss);(loss/32).backward()
                count=int(item['construction_count']);accumulator.collect(count);counts[count-1]+=1;loss_sum+=float(loss.detach())
                if (index+1)%32==0:
                    flat,stats=first.ordinary(accumulator.groups,rng);accumulator.assign(flat)
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True);accumulator.reset();del flat
                    stats.update(update=(epoch-1)*75+len(updates)+1,examples=index+1,preclip_norm=float(norm));updates.append(stats)
                    w.write(root/'STATE.json',dict(status='TRAINING',epoch=epoch,epoch_updates=len(updates),
                        updates=(epoch-1)*75+len(updates),total_updates=225,examples=index+1,total=2400,
                        seconds=time.time()-started,pid=os.getpid(),time=time.time()))
            elapsed=time.time()-started;assert len(updates)==75 and counts==[800,800,800]
            assert {int(s['step']) for s in opt.state.values()}=={epoch*75}
            assert all(torch.isfinite(v).all() for v in net.state_dict().values())
            w.write(root/f'GRADIENT_UPDATES_{epoch:03d}.json',dict(protocol_sha256=ph,updates=updates))
            w.write(root/'STATE.json',dict(status='VALIDATING',epoch=epoch,updates=epoch*75,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,root/f'VALIDATION_{epoch:03d}.json',epoch)
            actual,_=w.validate(root/f'VALIDATION_{epoch:03d}.json',identities)
            control,_=w.validate(Path(original['study'])/f'retained_unet/VALIDATION_{epoch:03d}.json',identities)
            if actual['selection_nmse']<best['metric']:
                best=dict(epoch=epoch,metric=actual['selection_nmse'])
                worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
            worker.atomic_torch(root/f'ACTUAL_{epoch:03d}.pt',dict(model=net.state_dict(),epoch=epoch,updates=epoch*75,protocol_sha256=ph))
            worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=epoch*75,
                best=best,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),projection_rng=rng.getstate()))
            comparison=base.compare(actual,parent,control)
            for row in comparison:
                if row['reference']=='retained_control_e1':row['reference']=f'retained_control_e{epoch}'
            criterion=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in comparison if r['count'] in (2,3))
            event=dict(epoch=epoch,updates=epoch*75,protocol_sha256=ph,validation=actual,best=best,
                criterion_met=criterion,comparison=comparison,train_seconds=elapsed,
                mean_training_loss=loss_sum/2400,peak_bytes=torch.cuda.max_memory_allocated(),
                gradient_receipts_sha256=w.digest(root/f'GRADIENT_UPDATES_{epoch:03d}.json'))
            w.write(root/f'EPOCH_{epoch:03d}.json',event);events.append(event)
            w.write(public,dict(status='PARTIAL',protocol_sha256=ph,events=events,heldout_read=False))
            print(dict(event='EPOCH_COMPLETE',**event),flush=True)
            del accumulator,train,item,estimates,logits,loss,norm
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,events=events,best=best,additional_updates=150,total_updates=225,
            predecessor_complete_sha256=p['predecessor_complete_sha256'],heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',epoch=3,updates=225,pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','predecessor','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True);pred=a.predecessor.resolve()
    try:
        with (root/'.run.lock').open('a') as own_lock:
            fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            w.write(root/'STATE.json',dict(status='WAITING_FOR_TF_AXIS_E1_AUDIT',pid=os.getpid(),time=time.time()))
            skip=False
            while not (pred/'COMPLETE.json').exists():
                if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                if (pred/'STATE.json').exists() and w.read(pred/'STATE.json')['status'].startswith('SKIPPED_'):skip=True;break
                time.sleep(5)
            if skip:w.write(root/'STATE.json',dict(status='SKIPPED_PRIOR_METHOD_MET_CRITERION',pid=os.getpid(),time=time.time()))
            else:
                audit=ROOT/'reports/2026-10-10/TF_AXIS_FINAL_AUDIT.json'
                while not audit.exists():time.sleep(5)
                p=register(root,pred);run(root,p,a.public.resolve())
                torch.cuda.empty_cache()
                w.write(root/'STATE.json',dict(status='CPU_AUDITING',epoch=3,updates=225,pid=os.getpid(),time=time.time()))
                subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,
                    str(Path(__file__).with_name('audit_continuation.py')),'--run',str(root),
                    '--public',str(a.public.resolve().parent)],check=True)
                w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',epoch=3,updates=225,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
