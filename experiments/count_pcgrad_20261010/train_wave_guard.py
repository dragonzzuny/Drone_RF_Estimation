"""Matched full U-Net with actual AdamW displacement protection for waveform2/3.

Derived from the frozen count-PCGrad worker; original gradient and AdamW moments
are retained. Only the proposed parameter displacement is corrected.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from gradient import seeded_rng
from wave_update_guard import WaveAccumulator, correct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
w=worker.watch
CHECK=ROOT/'reports/2026-10-10/WAVE_UPDATE_GUARD_CPU_CHECK.json'


def register(root, study, predecessor):
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    old=w.read(study/'PROTOCOL.json')
    audit=w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_FINAL_AUDIT.json')
    assert audit['status']=='PASS' and audit['study_protocol_sha256']==w.digest(study/'PROTOCOL.json')
    assert w.read(predecessor/'COMPLETE.json')['status']=='COMPLETE'
    check=w.read(CHECK);assert check['status']=='PASS'
    sources=dict(old['source_sha256']);sources.update(check['source_sha256'])
    for path in (Path(__file__),HERE/'audit_wave_guard.py'):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_WAVE_UPDATE_GUARD',source_sha256=sources,
        study=str(study),study_protocol_sha256=w.digest(study/'PROTOCOL.json'),
        predecessor=str(predecessor),predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        parent_checkpoint=old['parent_checkpoint'],parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        preparation=old['preparation'],preparation_sha256=old['preparation_sha256'],
        baseline=old['validation_identity_template'],baseline_sha256=old['validation_identity_sha256'],
        control_validation_sha256=w.digest(study/'retained_unet/VALIDATION_001.json'),
        control_checkpoint_sha256=w.digest(study/'retained_unet/ACTUAL_001.pt'),
        cpu_check_sha256=w.digest(CHECK),gpu_display_policy=old['gpu_display_policy'],
        plan_sha256=w.digest(ROOT/'reports/2026-10-10/WAVE_UPDATE_GUARD_PLAN_KO.md'),
        parameters=32142859,epochs=1,updates=75,examples=2400,validation_cases=630,
        effective_batch=32,microbatch=1,train_schedule_epoch=1,seed=0,projection_order_seed=0,
        lr=1e-5,weight_decay=1e-4,clip_norm=1.,precision='FP32 TF32off',
        loss='Original waveform PIT NMSE+coherence+inactive/background +0.1 count CE; no magnitude auxiliary',
        grouping='g_k=sum gradients of samples with constructed source count k /32; no renormalization by group size',
        gradient_update='Original mean gradient then clip and AdamW; project proposed parameter displacement onto waveform-count2/3 nonincrease halfspaces; optimizer moments unchanged',
        extra_derivative='Separate weighted count CE gradient for 2/3 examples is subtracted from original main gradients; no extra waveform forward',
        inference='Unchanged full U-Net, mixture only; no count/category/reference input',
        selection='Minimum mean counts2/3 NMSE including e0',
        acceptance='Counts2/3 NMSE lower, complex SI-SDR higher and weakest NMSE nonincreased than BOTH parent and matched control e1',
        limitation='One seed, reused DEV630, adaptive method selection; count tasks contain different examples, not a count-only causal experiment',
        heldout_read=False,independent_test=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        assert w.digest(ROOT/rel)==sha and w.digest(root/'source_snapshot'/rel)==sha
    for path,key in ((Path(p['study'])/'PROTOCOL.json','study_protocol_sha256'),
        (Path(p['predecessor'])/'COMPLETE.json','predecessor_complete_sha256'),
        (Path(p['parent_checkpoint']),'parent_checkpoint_sha256'),
        (Path(p['preparation'])/'PREPARATION.json','preparation_sha256'),
        (Path(p['baseline']),'baseline_sha256'),(CHECK,'cpu_check_sha256'),
        (ROOT/'reports/2026-10-10/WAVE_UPDATE_GUARD_PLAN_KO.md','plan_sha256'),
        (Path(p['study'])/'retained_unet/VALIDATION_001.json','control_validation_sha256'),
        (Path(p['study'])/'retained_unet/ACTUAL_001.pt','control_checkpoint_sha256')):
        assert w.digest(path)==p[key],str(path)


def compare(actual,parent,control):
    rows=[]
    for name,reference in [('parent',parent),('retained_control_e1',control)]:
        for count in (1,2,3):
            a,b=[next(g for g in v['by_count'] if g['count']==count) for v in (reference,actual)]
            rows.append(dict(reference=name,count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],
                si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
    return rows


def run(root,p,public):
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p);assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['predecessor'])/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(p['preparation'],'train_pack',1)
        validation=worker.NativeMixtures(p['preparation'],'validation_pack',1)
        assert len(train)==2400 and len(validation)==630
        parent,ids=w.validate(Path(p['baseline']),None)
        control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',ids)
        net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        w.write(root/'STATE.json',dict(status='VALIDATING_INITIAL',pid=os.getpid(),time=time.time()))
        worker.fit.base.validate(net,validation,root/'VALIDATION_000.json',0)
        initial,_=w.validate(root/'VALIDATION_000.json',ids)
        for a,b in zip(parent['by_count'],initial['by_count']):
            for key in ('mean_nmse','mean_si_sdr','weakest_nmse'):assert abs(a[key]-b[key])<2e-6
        opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
        accumulator=WaveAccumulator(net.named_parameters());rng=seeded_rng(0)
        best=dict(epoch=0,metric=initial['selection_nmse']);ph=w.digest(root/'PROTOCOL.json')
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True)
        updates=[];loss_sum=0.;count_loss=[0.,0.,0.];count_examples=[0,0,0];started=time.time()
        torch.cuda.reset_peak_memory_stats()
        w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=0,examples=0,total=2400,pid=os.getpid(),time=time.time()))
        for index in range(len(train)):
            item=worker.fit.base.batch([train[index]])
            estimates,logits=worker.predict(net,item)
            values=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])
            loss=values['loss']+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
            assert torch.isfinite(loss)
            count=int(item['construction_count'].item())
            if count in (2,3):
                ce=.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
                ce_grads=torch.autograd.grad(ce/32,accumulator.ce_parameters,retain_graph=True,allow_unused=True)
                accumulator.collect_ce(count,ce_grads);del ce_grads,ce
            (loss/32).backward();accumulator.collect(count)
            current=float(loss.detach());loss_sum+=current;count_loss[count-1]+=current;count_examples[count-1]+=1
            if (index+1)%32==0:
                flat=accumulator.groups.sum(0);accumulator.assign(flat)
                gram=torch.stack([torch.stack([torch.dot(a,b) for b in accumulator.groups]) for a in accumulator.groups])
                stats=dict(gram=gram.tolist(),ordinary_norm=float(flat.norm()),projected_norm=float(flat.norm()),change_norm=0.,
                    projected_direction_dot_original_tasks=[float(torch.dot(flat,g)) for g in accumulator.groups])
                protected=accumulator.waveform_groups()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                before=torch.cat([q.detach().flatten() for q in net.parameters()])
                opt.step()
                delta,guard_stats=correct(net.parameters(),before,protected)
                stats['waveform_guard']=guard_stats
                del protected
                stats.update(update=len(updates)+1,examples=index+1,count_examples=list(accumulator.counts),
                    preclip_norm=float(norm),actual_adamw_delta_norm=float(delta.norm()),
                    original_gradient_dot_actual_delta=[float(torch.dot(g,delta)) for g in accumulator.groups])
                updates.append(stats)
                opt.zero_grad(set_to_none=True);accumulator.reset();del flat,before,delta
                # 75 receipts are small; save every actual update for live reporting.
                w.write(root/'GRADIENT_UPDATES.json',dict(protocol_sha256=ph,updates=updates))
                w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=len(updates),examples=index+1,total=2400,
                    seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        elapsed=time.time()-started
        assert len(updates)==75 and count_examples==[800,800,800]
        assert {int(s['step']) for s in opt.state.values()}=={75}
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        peak=torch.cuda.max_memory_allocated()
        w.write(root/'STATE.json',dict(status='VALIDATING',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        worker.fit.base.validate(net,validation,root/'VALIDATION_001.json',1)
        actual,_=w.validate(root/'VALIDATION_001.json',ids)
        if actual['selection_nmse']<best['metric']:
            best=dict(epoch=1,metric=actual['selection_nmse'])
            worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        worker.atomic_torch(root/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,protocol_sha256=ph))
        worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,updates=75,best=best,
            protocol_sha256=ph,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),projection_rng=rng.getstate()))
        receipt=dict(epoch=1,updates=75,parameters=p['parameters'],protocol_sha256=ph,train_seconds=elapsed,peak_bytes=peak,
            mean_training_loss=loss_sum/2400,mean_training_loss_by_count=[a/b for a,b in zip(count_loss,count_examples)],
            count_examples=count_examples,best=best,validation=actual,gradient_receipts_sha256=w.digest(root/'GRADIENT_UPDATES.json'))
        w.write(root/'EPOCH_001.json',receipt);print(dict(event='EPOCH_COMPLETE',**receipt),flush=True)
        rows=compare(actual,parent,control)
        passed=all(r['nmse_delta']<0 and r['si_sdr_delta']>0 and r['weakest_nmse_delta']<=0 for r in rows if r['count'] in (2,3))
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,initial=initial,actual=actual,
            selected=initial if best['epoch']==0 else actual,comparison=rows,criterion_met=passed,
            train_seconds=elapsed,updates=75,heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        print(dict(status='COMPLETE',criterion_met=passed),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','predecessor','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            predecessor=a.predecessor.resolve()
            w.write(root/'STATE.json',dict(status='WAITING_FOR_CAGRAD',pid=os.getpid(),time=time.time()))
            while not (predecessor/'COMPLETE.json').exists():
                if (predecessor/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
                time.sleep(5)
            cg=w.read(predecessor/'COMPLETE.json')
            pg=w.read(Path(w.read(predecessor/'PROTOCOL.json')['predecessor'])/'COMPLETE.json')
            if cg['criterion_met'] or pg['criterion_met']:
                w.write(root/'STATE.json',dict(status='SKIPPED_PRIOR_METHOD_MET_CRITERION',pid=os.getpid(),time=time.time()))
                print('Prior method met criterion; conditional guard experiment skipped',flush=True)
            else:
                p=register(root,a.study.resolve(),predecessor);run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
