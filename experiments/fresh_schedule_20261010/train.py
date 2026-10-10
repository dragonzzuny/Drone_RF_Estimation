"""Adaptive diagnosis: original full U-Net, fresh-to-parent mixture schedule 3.

Not a new recording split or architecture. The historical replay control is
reused, including its failed actual e1. No downstream experiment is autoqueued.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
sys.path.insert(0, str(ROOT/'experiments/recursive_20261010'))
from validation import validate
w = core.w
worker = core.base.worker
PLAN = ROOT/'reports/2026-10-10/FRESH_SCHEDULE_PLAN_KO.md'


def recipe(row):
    n = int(row['count'])
    # Exclude the bookkeeping epoch; preserve physical mixture/crop settings.
    return (n, tuple(row['indices'][:n]), tuple(row['levels'][:n]),
            tuple(row['phases'][:n]), int(row['crop_start']))


def schedule_audit(preparation):
    data = [worker.NativeMixtures(preparation, 'train_pack', k, use_features=False) for k in (1, 2, 3)]
    seen = set(recipe(r) for d in data[:2] for r in d.rows)
    summaries = []
    for d in data:
        strata = Counter()
        packs = Counter()
        for r in d.rows:
            n = int(r['count']); clips = [d.library.clips[int(i)] for i in r['indices'][:n]]
            strata[str((n, tuple(c['category'] for c in clips), tuple(float(v) for v in r['levels'][:n])))] += 1
            packs.update(c['pack_id'] for c in clips)
        summaries.append(dict(schedule_epoch=d.epoch, rows_sha256=d.rows_hash, examples=len(d),
            overlap_with_parent_native_epochs1_2=sum(recipe(r) in seen for r in d.rows),
            count_histogram={str(n): int(np.sum(d.rows['count']==n)) for n in (1,2,3)},
            category_power_strata=dict(sorted(strata.items())), pack_occurrences=dict(sorted(packs.items()))))
    assert summaries[0]['overlap_with_parent_native_epochs1_2']==2400
    assert summaries[2]['overlap_with_parent_native_epochs1_2']==0
    a, b = [Counter(s['category_power_strata']) for s in (summaries[0], summaries[2])]
    return dict(status='PASS', schedules=summaries, recorded_iq_reads=0,
        same_category_power_histogram=(a==b), histogram_l1=sum(abs(a[k]-b[k]) for k in a.keys()|b.keys()),
        qualification='Schedule3 unseen in selected native e2 gradients; underlying TRAIN recordings are reused. '
        'Schedule3 has appeared in other development trials and is not an independent test or new recording.')


def register(root):
    assert not (root/'PROTOCOL.json').exists()
    old = w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    audit = schedule_audit(old['preparation']); w.write(root/'SCHEDULE_AUDIT.json', audit)
    parent = torch.load(old['parent_checkpoint'], map_location='cpu', weights_only=False)
    assert parent['best']['epoch']==2
    del parent
    sources = dict(old['source_sha256'])
    for path in (Path(__file__), Path(__file__).with_name('audit.py'),
                 ROOT/'experiments/window_overlap_20261010/diagnose.py',
                 ROOT/'experiments/recursive_20261010/validation.py'):
        sources[str(path.relative_to(ROOT))] = w.digest(path)
    control = Path(old['study'])/'retained_unet/VALIDATION_001.json'
    p = dict(status='REGISTERED_FRESH_TO_PARENT_SCHEDULE_DIAGNOSIS', original_protocol=old,
        source_sha256=sources, plan_sha256=w.digest(PLAN), schedule_audit_sha256=w.digest(root/'SCHEDULE_AUDIT.json'),
        feature_receipt_sha256=w.digest(Path(old['preparation'])/'features/train_pack_003.json'),
        historical_control=str(control), historical_control_sha256=w.digest(control),
        parameters=32142859, seed=0, train_schedule_epoch=3, examples=2400, epochs=1, updates=75,
        effective_batch=32, microbatch=1, lr=1e-5, weight_decay=1e-4, clip_norm=1.,
        precision='FP32 TF32off; non-reentrant activation checkpoint',
        changed_factor='Mixture recipe schedule1 -> schedule3, including stochastic stratum/pack composition',
        unchanged='Same selected parent, original full U-Net, waveform/count loss, optimizer reset, update budget, DEV630',
        control_reuse='Original one-pass replay e1; independently reproduced by completed zero-affinity control',
        selection='min mean NMSE counts2/3 including e0',
        acceptance='Both counts2/3: lower NMSE, higher complex SI-SDR, nonincreased weakest NMSE vs both parent and replay control',
        next_step='Decision after audited e1; no automatic architecture queue',
        heldout_read=False, independent_test=False, registered_at=time.time())
    for rel, sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p)
    return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    old=p['original_protocol']
    for path,sha in [(Path(old['parent_checkpoint']),old['parent_checkpoint_sha256']),
        (Path(old['preparation'])/'PREPARATION.json',old['preparation_sha256']),
        (Path(old['baseline']),old['baseline_sha256']), (PLAN,p['plan_sha256']),
        (root/'SCHEDULE_AUDIT.json',p['schedule_audit_sha256']),
        (Path(old['preparation'])/'features/train_pack_003.json',p['feature_receipt_sha256']),
        (Path(p['historical_control']),p['historical_control_sha256'])]:
        assert w.digest(path)==sha, str(path)


def passed(rows):
    return all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0
               for r in rows if r['count'] in (2,3))


def run(root,p,public):
    old=p['original_protocol']; ph=w.digest(root/'PROTOCOL.json')
    with worker.fit.base.LOCK.open('r') as lock:
        w.write(root/'STATE.json',dict(status='WAITING_GPU_LOCK',pid=os.getpid(),time=time.time()))
        fcntl.flock(lock,fcntl.LOCK_EX); verify(root,p)
        assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(699817,old['gpu_display_policy']))
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(old['preparation'],'train_pack',3)
        dev=worker.NativeMixtures(old['preparation'],'validation_pack',1)
        assert len(train)==2400 and len(dev)==630
        parent,ids=w.validate(Path(old['baseline']),None)
        control,_=w.validate(Path(p['historical_control']),ids)
        net=worker.make_model('retained_unet',Path(old['parent_checkpoint'])).cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        # e0 is exactly the pinned parent function: reuse its audited full DEV result.
        shutil.copyfile(old['baseline'],root/'VALIDATION_000.json')
        best=dict(epoch=0,metric=parent['selection_nmse'])
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
        torch.manual_seed(0); net.train(); opt.zero_grad(set_to_none=True)
        norms=[]; counts=[0,0,0]; loss_sum=0.; started=time.time(); torch.cuda.reset_peak_memory_stats()
        for index in range(len(train)):
            item=worker.fit.base.batch([train[index]])
            def forward(mix,context,position):
                return worker.predict(net,dict(mixture=mix,context_features=context,crop_start=position))
            output,logits=checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)
            loss=worker.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']
            loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
            assert torch.isfinite(loss); (loss/32).backward(); loss_sum+=float(loss.detach())
            counts[int(item['construction_count'])-1]+=1
            if (index+1)%32==0:
                assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in net.parameters())
                norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)))
                opt.step(); opt.zero_grad(set_to_none=True)
                w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,schedule_epoch=3,updates=len(norms),
                    target_updates=75,examples=index+1,seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        elapsed=time.time()-started
        assert len(norms)==75 and counts==[800,800,800]
        assert {int(q['step']) for q in opt.state.values()}=={75}
        worker.atomic_torch(root/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,protocol_sha256=ph))
        w.write(root/'STATE.json',dict(status='VALIDATING',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        validate(net,dev,root/'VALIDATION_001.json',1,worker.predict,worker)
        actual,_=w.validate(root/'VALIDATION_001.json',ids)
        if actual['selection_nmse']<best['metric']:
            best=dict(epoch=1,metric=actual['selection_nmse'])
            worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,
            updates=75,best=best,protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        comparison=core.base.compare(actual,parent,control)
        event=dict(epoch=1,updates=75,schedule_epoch=3,protocol_sha256=ph,validation=actual,best=best,
            comparison=comparison,criterion_met=passed(comparison),mean_training_loss=loss_sum/2400,
            count_examples=counts,preclip_gradient_norms=norms,train_seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated())
        w.write(root/'EPOCH_001.json',event); verify(root,p)
        result=dict(status='COMPLETE',event=event,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result); w.write(public/'FRESH_SCHEDULE_RESULT.json',result)
        print(dict(event='EPOCH_COMPLETE',**event),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True); a=parser.parse_args()
    root=a.run.resolve(); root.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{14,15}); os.nice(10)
    try:
        with (root/'.run.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root); run(root,p,a.public.resolve()); torch.cuda.empty_cache()
            w.write(root/'STATE.json',dict(status='CPU_AUDITING',updates=75,pid=os.getpid(),time=time.time()))
            subprocess.run(['taskset','-c','12,13','nice','-n','5',sys.executable,
                str(Path(__file__).with_name('audit.py')),'--run',str(root),'--public',str(a.public.resolve())],check=True)
            w.write(root/'STATE.json',dict(status='COMPLETE_AUDITED',updates=75,pid=os.getpid(),time=time.time()))
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time())); raise
