"""One adaptive ordered-context comparison; no automatic downstream queue."""
import argparse
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from torch.utils.checkpoint import checkpoint
from model import augment, parent_key
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ordered_fresh_parent',ROOT/'experiments/fresh_schedule_20261010/train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
w=t.w;worker=t.worker
PUBLIC=ROOT/'reports/2026-10-10'
PLAN=PUBLIC/'ORDERED_CONTEXT_PLAN_KO.md'


def make_net(parent=None):
    return augment(worker.make_model('retained_unet',parent))


def register(root):
    assert not (root/'PROTOCOL.json').exists()
    fr=ROOT/'local/fresh_schedule_20261010_v1';fresh=w.read(fr/'PROTOCOL.json');t.verify(fr,fresh)
    assert w.read(PUBLIC/'FRESH_SCHEDULE_AUDIT.json')['status']=='PASS'
    check=w.read(PUBLIC/'ORDERED_CONTEXT_CHECK.json');assert check['status']=='PASS'
    old=fresh['original_protocol'];sources=dict(fresh['source_sha256'])
    for f in ('model.py','train.py','audit.py','check.py'):
        path=Path(__file__).with_name(f);sources[str(path.relative_to(ROOT))]=w.digest(path)
    for rel,sha in check['source_sha256'].items():assert sources[rel]==sha
    pins={str(path):w.digest(path) for path in (PLAN,PUBLIC/'ORDERED_CONTEXT_CHECK.json',
        fr/'PROTOCOL.json',fr/'VALIDATION_001.json',PUBLIC/'FRESH_SCHEDULE_AUDIT.json',
        Path(old['parent_checkpoint']),Path(old['baseline']),Path(old['preparation'])/'PREPARATION.json',
        Path(old['preparation'])/'features/train_pack_003.json')}
    assert pins[old['parent_checkpoint']]==old['parent_checkpoint_sha256']
    p=dict(status='REGISTERED_ORDERED_CONTEXT',original_protocol=old,source_sha256=sources,pinned_files=pins,
        parameters=32142924,new_parameters=65,seed=0,train_schedule_epoch=3,examples=2400,epochs=1,updates=75,
        effective_batch=32,microbatch=1,parent_lr=1e-5,gate_lr=1e-2,clip_norm=1.,
        comparison_control=str(fr/'VALIDATION_001.json'),control_protocol_sha256=w.digest(fr/'PROTOCOL.json'),
        acceptance='Both counts2/3: NMSE lower, complex SI-SDR higher, weakest NMSE nonincreased vs parent and same-schedule control',
        selection='min mean NMSE counts2/3 including e0',heldout_read=False,independent_test=False,
        precision='FP32 TF32off non-reentrant checkpoint',registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha,rel
    for path,sha in p['pinned_files'].items():assert w.digest(Path(path))==sha,path


def compare(actual,parent,control):
    rows=[]
    for label,reference in [('parent',parent),('same_schedule_control',control)]:
        for a,b in zip(actual['by_count'],reference['by_count']):
            row=dict(reference=label,count=a['count'])
            for out,key in [('nmse_delta','mean_nmse'),('si_sdr_delta','mean_si_sdr'),('weakest_nmse_delta','weakest_nmse')]:
                row[out]=a[key]-b[key] if a[key] is not None and b[key] is not None else None
            rows.append(row)
    return rows


def passed(rows):
    return all(all(r[k] is not None and math.isfinite(r[k]) for k in ('nmse_delta','si_sdr_delta','weakest_nmse_delta'))
        and r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))


def run(root,p):
    old=p['original_protocol'];ph=w.digest(root/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        w.write(root/'STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(763979,old['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(old['preparation'],'train_pack',3)
        dev=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        parent,ids=w.validate(Path(old['baseline']),None);control,_=w.validate(Path(p['comparison_control']),ids)
        torch.manual_seed(0);net=make_net(Path(old['parent_checkpoint'])).cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        gate=net.context_encoder.order_gate
        base=[q for name,q in net.named_parameters() if name!='context_encoder.order_gate']
        assert sum(q.numel() for q in base)==32142859
        opt=torch.optim.AdamW([dict(params=base,lr=p['parent_lr'],weight_decay=1e-4),
            dict(params=[gate],lr=p['gate_lr'],weight_decay=0.)],foreach=False)
        shutil.copyfile(old['baseline'],root/'VALIDATION_000.json')
        best=dict(epoch=0,metric=parent['selection_nmse'])
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        norms=[];gate_history=[];counts=[0,0,0];loss_sum=0.;started=time.time()
        torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        for index in range(len(train)):
            item=worker.fit.base.batch([train[index]])
            def forward(mix,context,position):return worker.predict(net,dict(mixture=mix,context_features=context,crop_start=position))
            output,logits=checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)
            loss=worker.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
            loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
            assert torch.isfinite(loss);(loss/32).backward();loss_sum+=float(loss.detach())
            counts[int(item['construction_count'])-1]+=1
            if (index+1)%32==0:
                assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in net.parameters())
                norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)))
                opt.step();opt.zero_grad(set_to_none=True)
                gate_history.append(gate.detach().tanh().cpu().tolist())
                w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=len(norms),target_updates=75,
                    examples=index+1,gate_mean_abs=float(gate.detach().tanh().abs().mean()),
                    seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        elapsed=time.time()-started;assert len(norms)==75 and counts==[800,800,800]
        assert {int(q['step']) for q in opt.state.values()}=={75}
        worker.atomic_torch(root/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,protocol_sha256=ph))
        w.write(root/'STATE.json',dict(status='VALIDATING',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        t.validate(net,dev,root/'VALIDATION_001.json',1,worker.predict,worker)
        actual,_=w.validate(root/'VALIDATION_001.json',ids)
        if actual['selection_nmse']<best['metric']:
            best=dict(epoch=1,metric=actual['selection_nmse'])
            worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,updates=75,
            best=best,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        comparison=compare(actual,parent,control)
        event=dict(epoch=1,updates=75,protocol_sha256=ph,validation=actual,best=best,comparison=comparison,
            criterion_met=passed(comparison),mean_training_loss=loss_sum/2400,count_examples=counts,
            preclip_gradient_norms=norms,gate_history=gate_history,train_seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated())
        w.write(root/'EPOCH_001.json',event);verify(root,p)
        result=dict(status='COMPLETE',event=event,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(PUBLIC/'ORDERED_CONTEXT_RESULT.json',result)
        print(dict(event='EPOCH_COMPLETE',**event),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{14,15});os.nice(10)
    try:
        with (root/'.run.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);p=register(root);run(root,p);torch.cuda.empty_cache()
            w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=75,pid=os.getpid(),time=time.time()))
            subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,str(Path(__file__).with_name('audit.py')),'--run',str(root)],check=True)
            w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=75,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
