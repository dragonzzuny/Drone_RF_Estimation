"""Full-capacity one-epoch canonical-phase adaptation; no oracle information."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from torch.nn import functional as F
import evaluate as prior
from canonical import CanonicalPhaseSeparator

w,worker=prior.w,prior.worker


def register(root,dependency,study):
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate phase training')
    base=w.read(dependency/'PROTOCOL.json');done=w.read(dependency/'COMPLETE.json')
    assert done['status']=='COMPLETE' and done['protocol_sha256']==w.digest(dependency/'PROTOCOL.json')
    audit=w.read(w.ROOT/'reports/2026-10-10/SOURCE_INTERACTION_FINAL_AUDIT.json')
    assert audit['status']=='PASS' and audit['study_protocol_sha256']==w.digest(study/'PROTOCOL.json')
    sources=dict(base['source_sha256']);sources[str(Path(__file__).relative_to(w.ROOT))]=w.digest(Path(__file__))
    p=dict(base)
    p.update(status='REGISTERED_CANONICAL_PHASE_ADAPTATION',source_sha256=sources,
        dependency=str(dependency),dependency_protocol_sha256=w.digest(dependency/'PROTOCOL.json'),
        dependency_complete_sha256=w.digest(dependency/'COMPLETE.json'),
        canonical_initial_sha256=w.digest(dependency/'CANONICAL.json'),
        study=str(study),study_protocol_sha256=w.digest(study/'PROTOCOL.json'),
        control_validation_sha256=w.digest(study/'retained_unet/VALIDATION_001.json'),
        control_receipt_sha256=w.digest(study/'retained_unet/EPOCH_001.json'),
        epochs=1,updates=75,examples=2400,effective_batch=32,microbatch=1,train_schedule_epoch=1,seed=0,
        lr=1e-5,weight_decay=1e-4,clip_norm=1.,precision='FP32 TF32off',
        optimizer='fresh AdamW, all32142859 original parameters trainable',
        initialization='Reload identical retained native parent e2, then train once under the fixed canonical-phase wrapper',
        operation='Canonicalize mixture phase, train the full original U-Net and restore output phase; one pass at inference',
        loss='unchanged PIT NMSE+coherence+inactive/background plus0.1 count CE',
        selection='minimum counts2/3 mean raw NMSE over canonical e0 and e1; also report actual e1',
        acceptance='Selected candidate AND actual e1 reported against original parent and equal-budget retained e1; advance only if selected NMSE2/3 lower, complex SI-SDR2/3 higher and weakest NMSE2/3 nonincreased against both',
        decision_before_initial_result='One adaptation epoch planned during canonical e0 evaluation, regardless of its performance',
        control_reuse='Audited retained_unet/EPOCH_001, identical source data, parent, optimizer and update budget',
        registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(w.ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(w.ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p);return p


def verify(root,p):
    prior.verify(root,p)
    for path,sha in [(Path(p['dependency'])/'PROTOCOL.json',p['dependency_protocol_sha256']),
        (Path(p['dependency'])/'COMPLETE.json',p['dependency_complete_sha256']),
        (Path(p['dependency'])/'CANONICAL.json',p['canonical_initial_sha256']),
        (Path(p['study'])/'PROTOCOL.json',p['study_protocol_sha256']),
        (Path(p['study'])/'retained_unet/VALIDATION_001.json',p['control_validation_sha256']),
        (Path(p['study'])/'retained_unet/EPOCH_001.json',p['control_receipt_sha256'])]:
        assert w.digest(path)==sha


def run(root,p,public):
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);verify(root,p)
        assert torch.cuda.is_available()
        w.write(root/'GPU_START_CHECK.json',worker.guard.wait_for_predecessor(w.read(Path(p['dependency'])/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(p['preparation'],'train_pack',1)
        validation=worker.NativeMixtures(p['preparation'],'validation_pack',1)
        assert len(train)==2400 and len(validation)==630
        parent,ids=w.validate(Path(p['baseline']),None)
        initial,_=w.validate(Path(p['dependency'])/'CANONICAL.json',ids)
        control,_=w.validate(Path(p['study'])/'retained_unet/VALIDATION_001.json',ids)
        shutil.copyfile(Path(p['dependency'])/'CANONICAL.json',root/'VALIDATION_000.json')
        net=CanonicalPhaseSeparator(worker.make_model('retained_unet',Path(p['parent_checkpoint']))).cuda()
        assert sum(q.numel() for q in net.parameters())==p['parameters']
        opt=torch.optim.AdamW(net.parameters(),lr=p['lr'],weight_decay=p['weight_decay'],foreach=False)
        best=dict(epoch=0,metric=initial['selection_nmse']);ph=w.digest(root/'PROTOCOL.json')
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True);norms=[];loss_sum=0.;started=time.time()
        w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=0,examples=0,total=2400,pid=os.getpid(),time=time.time()))
        for index in range(len(train)):
            item=worker.fit.base.batch([train[index]])
            estimates,logits=worker.predict(net,item)
            loss=worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']+.1*F.cross_entropy(logits,item['construction_count']-1)
            assert torch.isfinite(loss)
            (loss/32).backward();loss_sum+=float(loss.detach())
            if (index+1)%32==0:
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True)
                if len(norms)%5==0:
                    w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=len(norms),examples=index+1,total=2400,seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        train_seconds=time.time()-started
        assert len(norms)==75 and {int(s['step']) for s in opt.state.values()}=={75}
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        w.write(root/'STATE.json',dict(status='VALIDATING',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        worker.fit.base.validate(net,validation,root/'VALIDATION_001.json',1)
        actual,_=w.validate(root/'VALIDATION_001.json',ids)
        if actual['selection_nmse']<best['metric']:
            best=dict(epoch=1,metric=actual['selection_nmse'])
            worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        worker.atomic_torch(root/'ACTUAL_001.pt',dict(model=net.state_dict(),epoch=1,updates=75,protocol_sha256=ph))
        worker.atomic_torch(root/'LAST.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=1,updates=75,best=best,protocol_sha256=ph,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        w.write(root/'EPOCH_001.json',dict(epoch=1,updates=75,parameters=p['parameters'],protocol_sha256=ph,
            preclip_gradient_norms=norms,mean_training_loss=loss_sum/2400,train_seconds=train_seconds,best=best,validation=actual))
        selected=initial if best['epoch']==0 else actual;comparisons=[]
        for scope,value in [('actual_e1',actual),('selected',selected)]:
            for name,reference in [('parent',parent),('retained_control_e1',control)]:
                for count in (2,3):
                    a,b=[next(g for g in v['by_count'] if g['count']==count) for v in (reference,value)]
                    comparisons.append(dict(scope=scope,reference=name,count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
        passed=all(v['nmse_delta']<0 and v['si_sdr_delta']>0 and v['weakest_nmse_delta']<=0 for v in comparisons if v['scope']=='selected')
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,initial=initial,actual=actual,selected=selected,
            comparison=comparisons,criterion_met=passed,train_seconds=train_seconds,updates=75,
            heldout_read=False,independent_test=False,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',epoch=1,updates=75,pid=os.getpid(),time=time.time()))
        print(dict(status='COMPLETE',criterion_met=passed,actual=actual),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('dependency','study','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root,a.dependency.resolve(),a.study.resolve());run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
