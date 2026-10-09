"""Frozen, conditional full-model source-affinity three-arm comparison."""
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
from torch.nn import functional as F
from affinity import Capture,targets,loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from drone_rf.waveform import analyze
w=core.w;worker=core.base.worker;base=core.base
ARMS=('affinity_control','hard_affinity','soft_affinity')
PLAN=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_PLAN_KO.md'
CHECK=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_TRAINING_CHECK.json'
ALGEBRA=ROOT/'reports/2026-10-10/SOURCE_AFFINITY_ALGEBRA_CHECK.json'


def passed(rows):
    return all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))


def register(root,predecessor):
    assert not (root/'PROTOCOL.json').exists()
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    check=w.read(CHECK);assert check['status']=='PASS' and check['initial_predictions_exact']
    assert check['parent_sha256']==old['parent_checkpoint_sha256']
    assert w.read(ALGEBRA)['status']=='PASS'
    sources=dict(check['source_sha256'])
    for name in ('train.py','audit.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_CONDITIONAL_SOURCE_AFFINITY',original_protocol=old,
        predecessor=str(predecessor),predecessor_protocol_sha256=w.digest(predecessor/'PROTOCOL.json'),
        source_sha256=sources,plan_sha256=w.digest(PLAN),check_sha256=w.digest(CHECK),algebra_sha256=w.digest(ALGEBRA),
        arms=list(ARMS),parameters=32143899,base_parameters=32142859,aux_parameters=1040,
        updates_per_arm=75,examples_per_arm=2400,epochs_per_arm=1,effective_batch=32,microbatch=1,
        validation_cases=630,seed=0,train_schedule_epoch=1,lr=1e-5,weight_decay=1e-4,clip_norm=1.,
        precision='FP32 backbone TF32off; FP64 auxiliary Gram reductions',
        auxiliary_weight={'affinity_control':0.,'hard_affinity':.1,'soft_affinity':.1},
        auxiliary_counts=[2,3],embedding_dimension=16,reference_power_pool=2,
        target='hard argmax power onehot versus sqrt of fractional reference powers',
        weight='Mean of per-source energy-normalized TF distributions; identical hard/soft',
        loss='Unchanged waveform PIT plus0.1 count CE plus weighted auxiliary on decoder embedding',
        inference='Parent mixture-only complex output; auxiliary disabled, no clustering or target/count input',
        selection='Minimum mean counts2/3 NMSE including e0',
        acceptance='Control vs parent; candidates vs BOTH parent and fresh control: lower NMSE, higher complex SI-SDR, nonincreased weakest NMSE in2/3',
        soft_vs_hard='Separate same joint criterion; not presumed better',
        limitation='One seed, reused DEV630; experimental application, not original deep clustering/Chimera reproduction',
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
    for path,key in ((PLAN,'plan_sha256'),(CHECK,'check_sha256'),(ALGEBRA,'algebra_sha256'),
        (Path(p['predecessor'])/'PROTOCOL.json','predecessor_protocol_sha256')):assert w.digest(path)==p[key]


def make_model(parent=None):
    net=worker.make_model('retained_unet',parent)
    capture=Capture(net)
    return net,capture


def run(root,p,public):
    old=p['original_protocol'];ph=w.digest(root/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        audit=ROOT/'reports/2026-10-10/PAIRED_WINDOW_AUDIT.json'
        assert w.read(audit)['status']=='PASS' and w.read(audit)['complete_sha256']==w.digest(Path(p['predecessor'])/'COMPLETE.json')
        w.write(root/'PREDECESSOR_RECEIPT.json',dict(complete_sha256=w.digest(Path(p['predecessor'])/'COMPLETE.json'),audit_sha256=w.digest(audit)))
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['predecessor'])/'STATE.json')['pid'],old['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=worker.NativeMixtures(old['preparation'],'train_pack',1)
        validation=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        parent,ids=w.validate(Path(old['baseline']),None)
        historical,_=w.validate(Path(old['study'])/'retained_unet/VALIDATION_001.json',ids)
        assert len(data)==2400 and len(validation)==630;events=[]
        for number,arm in enumerate(ARMS):
            folder=root/arm;folder.mkdir(exist_ok=True);verify(root,p)
            net,capture=make_model(Path(old['parent_checkpoint']));net=net.cuda()
            assert sum(q.numel() for q in net.parameters())==p['parameters']
            opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
            w.write(root/'STATE.json',dict(status='VALIDATING_INITIAL',arm=arm,pid=os.getpid(),time=time.time()))
            worker.fit.base.validate(net,validation,folder/'VALIDATION_000.json',0)
            initial,_=w.validate(folder/'VALIDATION_000.json',ids)
            for a,b in zip(parent['by_count'],initial['by_count']):
                for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
            best=dict(epoch=0,metric=initial['selection_nmse'])
            worker.atomic_torch(folder/'BEST.pt',dict(model=net.state_dict(),best=best,arm=arm,protocol_sha256=ph))
            torch.manual_seed(0);net.train();capture.enabled=True;opt.zero_grad(set_to_none=True)
            totals=dict(main=0.,auxiliary=0.,loss=0.);norms=[];counts=[0,0,0];started=time.time()
            torch.cuda.reset_peak_memory_stats();kind='hard' if arm=='hard_affinity' else 'soft'
            for index in range(len(data)):
                item=worker.fit.base.batch([data[index]]);count=int(item['construction_count'])
                estimates,logits=worker.predict(net,item);embedding=capture.take()
                with torch.no_grad():
                    power=F.avg_pool2d(analyze(item['references'][0]).abs().square()[None],2)
                    y,weight=targets(power,kind)
                auxiliary=loss(embedding,y,weight)
                main=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
                main=main+.1*F.cross_entropy(logits,item['construction_count']-1)
                lam=p['auxiliary_weight'][arm] if count in (2,3) else 0.
                objective=main+lam*auxiliary
                assert torch.isfinite(objective);(objective/32).backward()
                totals['main']+=float(main.detach());totals['auxiliary']+=float(auxiliary.detach());totals['loss']+=float(objective.detach())
                counts[count-1]+=1
                if (index+1)%32==0:
                    assert all(q.grad is not None for q in net.parameters())
                    norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)))
                    opt.step();opt.zero_grad(set_to_none=True)
                    w.write(root/'STATE.json',dict(status='TRAINING',arm=arm,epoch=1,updates=len(norms),
                        total_updates=number*75+len(norms),target_total_updates=225,examples=index+1,
                        total_examples=2400,seconds=time.time()-started,pid=os.getpid(),time=time.time()))
                del estimates,logits,embedding,power,y,weight,auxiliary,main,objective
            elapsed=time.time()-started;capture.enabled=False
            assert len(norms)==75 and counts==[800,800,800]
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
            comparator=historical if number==0 else events[0]['validation']
            comparisons=base.compare(actual,parent,comparator)
            for r in comparisons:
                if r['reference']=='retained_control_e1':r['reference']='historical_e1' if number==0 else 'matched_affinity_control'
            criterion=passed([q for q in comparisons if number>0 or q['reference']=='parent'])
            hard_comparison=base.compare(actual,parent,events[1]['validation'])[3:] if number==2 else []
            for r in hard_comparison:r['reference']='matched_hard_affinity'
            event=dict(arm=arm,epoch=1,updates=75,protocol_sha256=ph,validation=actual,best=best,
                comparison=comparisons,criterion_met=criterion,hard_comparison=hard_comparison,
                soft_beats_hard=passed(hard_comparison) if number==2 else None,
                train_seconds=elapsed,mean_training_values={k:v/2400 for k,v in totals.items()},
                preclip_gradient_norms=norms,count_examples=counts,peak_bytes=torch.cuda.max_memory_allocated())
            w.write(folder/'EPOCH_001.json',event);events.append(event)
            w.write(public,dict(status='PARTIAL',protocol_sha256=ph,events=events,heldout_read=False))
            print(dict(event='EPOCH_COMPLETE',**event),flush=True)
            capture.close();del net,opt,capture,item;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,events=events,total_updates=225,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','predecessor','public'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True);pred=a.predecessor.resolve()
    try:
        with (root/'.run.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);p=register(root,pred)
            w.write(root/'STATE.json',dict(status='WAITING_FOR_PAIRED_WINDOW',pid=os.getpid(),time=time.time()))
            while not (pred/'COMPLETE.json').exists():
                if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                time.sleep(5)
            verify(root,p)
            if any(e['criterion_met'] for e in w.read(pred/'COMPLETE.json')['events']):
                w.write(root/'STATE.json',dict(status='SKIPPED_PAIRED_WINDOW_MET_CRITERION',pid=os.getpid(),time=time.time()))
            else:
                while not (ROOT/'reports/2026-10-10/PAIRED_WINDOW_AUDIT.json').exists():
                    if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor audit failed')
                    time.sleep(5)
                run(root,p,a.public.resolve());torch.cuda.empty_cache()
                w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=225,pid=os.getpid(),time=time.time()))
                subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,
                    str(Path(__file__).with_name('audit.py')),'--run',str(root),'--public',str(a.public.resolve().parent)],check=True)
                w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=225,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
