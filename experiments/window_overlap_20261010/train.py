"""Conditional matched paired-window supervision/consistency experiment."""
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
import diagnose
from paired import paired_items
from training_step import accumulate

ROOT=diagnose.ROOT;base=diagnose.base;w=diagnose.w;worker=base.worker
ARMS=('paired_supervision','paired_consistency')
PLAN=ROOT/'reports/2026-10-10/PAIRED_WINDOW_PLAN_KO.md'
CHECK=ROOT/'reports/2026-10-10/PAIRED_WINDOW_TRAINING_CHECK.json'


def passed(rows):
    return all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))


def register(root,predecessor):
    assert not (root/'PROTOCOL.json').exists()
    old=w.read(predecessor/'PROTOCOL.json')['original_protocol']
    check=w.read(CHECK);assert check['status']=='PASS' and check['all_parameter_gradients_finite']
    algebra=ROOT/'reports/2026-10-10/PAIRED_WINDOW_ALGEBRA_CHECK.json'
    assert w.read(algebra)['status']=='PASS'
    audit=ROOT/'reports/2026-10-10/TF_AXIS_ADAPTATION_AUDIT.json'
    assert w.read(audit)['status']=='PASS' and w.read(audit)['complete_sha256']==w.digest(predecessor/'COMPLETE.json')
    sources=dict(check['source_sha256'])
    for name in ('train.py','audit_training.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_PAIRED_WINDOW_TRAINING',original_protocol=old,
        predecessor=str(predecessor),predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        predecessor_audit_sha256=w.digest(audit),plan_sha256=w.digest(PLAN),check_sha256=w.digest(CHECK),
        algebra_sha256=w.digest(algebra),source_sha256=sources,arms=list(ARMS),parameters=32142859,
        epochs_per_arm=1,updates_per_arm=75,mixture_pairs_per_arm=2400,views_per_arm=4800,
        forward_calls_per_pair=3,backward_calls_per_pair=2,effective_batch_pairs=32,
        seed=0,train_schedule_epoch=1,lr=1e-5,weight_decay=1e-4,clip_norm=1.,
        consistency_weight={'paired_supervision':0.,'paired_consistency':.1},
        consistency_counts=[2,3],offset_samples=16384,precision='FP32 TF32off',
        loss='Mean original supervised loss on two physically overlapping crops; add0.1 common-sample consistency only for2/3 in candidate',
        correspondence='Independent full-window supervised PIT assignments used only inside the training objective',
        selection='Minimum mean counts2/3 NMSE including e0',
        acceptance='Control: joint gain versus parent. Candidate: joint gain versus BOTH parent and paired supervised control; weakest NMSE nonincrease required.',
        limitation='One seed, reused DEV630; two views of same recordings, not new independent examples. Twice supervised views and three forwards per original scheduled case in BOTH arms.',
        heldout_read=False,independent_test=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    old=p['original_protocol']
    assert w.digest(Path(old['parent_checkpoint']))==old['parent_checkpoint_sha256']
    assert w.digest(Path(old['preparation'])/'PREPARATION.json')==old['preparation_sha256']
    assert w.digest(Path(p['predecessor'])/'COMPLETE.json')==p['predecessor_complete_sha256']
    assert w.digest(PLAN)==p['plan_sha256'] and w.digest(CHECK)==p['check_sha256']
    assert w.digest(ROOT/'reports/2026-10-10/PAIRED_WINDOW_ALGEBRA_CHECK.json')==p['algebra_sha256']
    assert w.digest(ROOT/'reports/2026-10-10/TF_AXIS_ADAPTATION_AUDIT.json')==p['predecessor_audit_sha256']
    assert w.digest(Path(old['baseline']))==old['baseline_sha256']


def run(root,p,public):
    old=p['original_protocol'];ph=w.digest(root/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
        assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['predecessor'])/'STATE.json')['pid'],old['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(old['preparation'],'train_pack',1)
        validation=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        parent,ids=w.validate(Path(old['baseline']),None)
        historical,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
        assert len(data)==2400 and len(validation)==630
        events=[]
        for arm_number,arm in enumerate(ARMS):
            folder=root/arm;folder.mkdir(exist_ok=True);verify(root,p)
            net=worker.make_model('retained_unet',Path(old['parent_checkpoint'])).cuda()
            assert sum(q.numel() for q in net.parameters())==p['parameters']
            assert not any(isinstance(m,(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm)) for m in net.modules())
            opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
            w.write(root/'STATE.json',dict(status='VALIDATING_INITIAL',arm=arm,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,folder/'VALIDATION_000.json',0)
            initial,_=w.validate(folder/'VALIDATION_000.json',ids)
            for a,b in zip(parent['by_count'],initial['by_count']):
                for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
            best=dict(epoch=0,metric=initial['selection_nmse'])
            worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True)
            norms=[];totals=dict(loss=0.,main=0.,consistency=0.);counts=[0,0,0];offsets={-16384:0,16384:0}
            started=time.time();torch.cuda.reset_peak_memory_stats()
            for index in range(len(data)):
                raw_a,raw_b,offset=paired_items(data,index)
                a,b=worker.fit.base.batch([raw_a]),worker.fit.base.batch([raw_b])
                count=int(a['construction_count']);weight=p['consistency_weight'][arm] if count in (2,3) else 0.
                value=accumulate(net,a,b,offset,worker.predict,worker.pit_waveform_loss,weight,32)
                for key in totals:totals[key]+=value[key]
                counts[count-1]+=1;offsets[offset]+=1
                if (index+1)%32==0:
                    assert all(q.grad is not None for q in net.parameters())
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                    norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True)
                    w.write(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=1,updates=len(norms),
                        total_updates=arm_number*75+len(norms),target_total_updates=150,
                        pairs=index+1,total_pairs=2400,views=2*(index+1),seconds=time.time()-started,pid=os.getpid(),time=time.time()))
            elapsed=time.time()-started
            assert len(norms)==75 and counts==[800,800,800] and sum(offsets.values())==2400
            assert {int(q['step']) for q in opt.state.values()}=={75}
            assert all(torch.isfinite(q).all() for q in net.state_dict().values())
            w.write(root/'STATE.json',dict(status='VALIDATING',arm=arm,epoch=1,updates=75,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,folder/'VALIDATION_001.json',1)
            actual,_=w.validate(folder/'VALIDATION_001.json',ids)
            if actual['selection_nmse']<best['metric']:
                best=dict(epoch=1,metric=actual['selection_nmse'])
                worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            worker.atomic_torch(folder/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,arm=arm,protocol_sha256=ph))
            worker.atomic_torch(folder/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,updates=75,
                best=best,arm=arm,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            comparator=historical if arm_number==0 else events[0]['validation']
            comparisons=base.compare(actual,parent,comparator)
            for q in comparisons:
                if q['reference']=='retained_control_e1':q['reference']='historical_single_view_e1' if arm_number==0 else 'matched_paired_supervision'
            criterion_rows=[q for q in comparisons if arm_number==1 or q['reference']=='parent']
            event=dict(arm=arm,epoch=1,updates=75,protocol_sha256=ph,validation=actual,best=best,
                comparison=comparisons,criterion_met=passed(criterion_rows),train_seconds=elapsed,
                mean_training_values={k:v/2400 for k,v in totals.items()},preclip_gradient_norms=norms,
                count_pairs=counts,offset_counts=offsets,peak_bytes=torch.cuda.max_memory_allocated(),
                exact_recomputed_second_forward_all_pairs=True)
            w.write(folder/'EPOCH_001.json',event);events.append(event)
            w.write(public,dict(status='PARTIAL',protocol_sha256=ph,events=events,heldout_read=False))
            print(dict(event='EPOCH_COMPLETE',**event),flush=True)
            del net,opt,a,b,raw_a,raw_b,norm;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,events=events,total_updates=150,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','predecessor','public'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True);pred=a.predecessor.resolve()
    try:
        with (root/'.run.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            w.write(root/'STATE.json',dict(status='WAITING_FOR_TF_AXIS_ADAPTATION',pid=os.getpid(),time=time.time()))
            while not (pred/'COMPLETE.json').exists():
                if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                time.sleep(5)
            if any(e['criterion_met'] for e in w.read(pred/'COMPLETE.json')['events']):
                w.write(root/'STATE.json',dict(status='SKIPPED_ADAPTATION_MET_CRITERION',pid=os.getpid(),time=time.time()))
            else:
                audit=ROOT/'reports/2026-10-10/TF_AXIS_ADAPTATION_AUDIT.json'
                while not audit.exists():
                    if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor audit failed')
                    time.sleep(5)
                p=register(root,pred);run(root,p,a.public.resolve());torch.cuda.empty_cache()
                w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=150,pid=os.getpid(),time=time.time()))
                subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,
                    str(Path(__file__).with_name('audit_training.py')),'--run',str(root),'--public',str(a.public.resolve().parent)],check=True)
                w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=150,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
