"""One prespecified, full-capacity native-RF epoch with magnitude supervision."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from objective import objective, worker, ROOT

w=worker.watch
CHECK=ROOT/'reports/2026-10-10/MAGNITUDE_CPU_CHECK.json'


def register(root, study, predecessor):
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    old=w.read(study/'PROTOCOL.json');audit=w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_FINAL_AUDIT.json')
    assert audit['status']=='PASS' and audit['study_protocol_sha256']==w.digest(study/'PROTOCOL.json')
    checks=w.read(CHECK);assert checks['status']=='PASS'
    assert w.read(predecessor/'COMPLETE.json')['status']=='COMPLETE'
    sources=dict(old['source_sha256']);sources.update(checks['source_sha256'])
    for name in ('train.py','spectral_evaluation.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    p=dict(status='REGISTERED_NATIVE_NONLOG_MAGNITUDE',source_sha256=sources,
        study=str(study),study_protocol_sha256=w.digest(study/'PROTOCOL.json'),
        predecessor=str(predecessor),predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        parent_checkpoint=old['parent_checkpoint'],parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        preparation=old['preparation'],preparation_sha256=old['preparation_sha256'],
        baseline=old['validation_identity_template'],baseline_sha256=old['validation_identity_sha256'],
        control_validation_sha256=w.digest(study/'retained_unet/VALIDATION_001.json'),
        control_receipt_sha256=w.digest(study/'retained_unet/EPOCH_001.json'),
        control_checkpoint_sha256=w.digest(study/'retained_unet/ACTUAL_001.pt'),
        cpu_check_sha256=w.digest(CHECK),gpu_display_policy=old['gpu_display_policy'],
        parameters=32142859,epochs=1,updates=75,examples=2400,validation_cases=630,
        effective_batch=32,microbatch=1,train_schedule_epoch=1,seed=0,lr=1e-5,
        weight_decay=1e-4,clip_norm=1.,precision='FP32 TF32off',magnitude_weight=.1,
        magnitude='Mean active-source relative non-log STFT magnitude L1; FFT512 hop128 sqrt Hann, computed AFTER waveform reconstruction',
        magnitude_denominator='max(mean abs target STFT, 0.001 mean abs mixture STFT, 1e-12)',
        assignment='Original waveform NMSE+coherence whole-window PIT; magnitude uses the same assignment, not a separate TF matching or combined-cost assignment',
        initialization='Same retained native parent e2; all original parameters trainable; fresh AdamW',
        loss='Original PIT NMSE+coherence+inactive/background +0.1 count CE +0.1 relative magnitude L1',
        selection='Minimum mean counts2/3 waveform NMSE over parent e0 and actual e1; actual e1 always reported',
        acceptance='Selected and actual metrics reported against parent and matched retained e1; advance only if selected NMSE2/3 lower, complex SI-SDR2/3 higher and weakest NMSE2/3 nonincreased against BOTH',
        control_reuse='Previously audited retained_unet e1, same parent/data schedule/75 updates/optimizer/microbatch',
        diagnostic='Independent double-precision spectral magnitude/phase error decomposition for parent, control e1 and actual candidate e1',
        no_validation_weight_search=True,heldout_read=False,independent_test=False,
        limitation='One seed, repeated DEV630 from five recording groups; known adaptation, not a claim of RF novelty or resolved phase compensation',
        registered_at=time.time())
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
        (Path(p['study'])/'retained_unet/VALIDATION_001.json','control_validation_sha256'),
        (Path(p['study'])/'retained_unet/EPOCH_001.json','control_receipt_sha256'),
        (Path(p['study'])/'retained_unet/ACTUAL_001.pt','control_checkpoint_sha256')):
        assert w.digest(path)==p[key]


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
        best=dict(epoch=0,metric=initial['selection_nmse']);ph=w.digest(root/'PROTOCOL.json')
        worker.atomic_torch(root/'BEST.pt',dict(model=net.state_dict(),best=best,protocol_sha256=ph))
        torch.manual_seed(0);net.train();opt.zero_grad(set_to_none=True);norms=[];loss_sum=main_sum=mag_sum=0.;started=time.time()
        w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=0,examples=0,total=2400,pid=os.getpid(),time=time.time()))
        for index in range(len(train)):
            item=worker.fit.base.batch([train[index]])
            estimates,logits=worker.predict(net,item)
            values=objective(estimates,logits,item,p['magnitude_weight']);loss=values['loss']
            assert torch.isfinite(loss)
            (loss/32).backward();loss_sum+=float(loss.detach());main_sum+=float(values['main'].detach());mag_sum+=float(values['magnitude'].detach())
            if (index+1)%32==0:
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True)
                if len(norms)%5==0:
                    w.write(root/'STATE.json',dict(status='TRAINING',epoch=1,updates=len(norms),examples=index+1,total=2400,seconds=time.time()-started,pid=os.getpid(),time=time.time()))
        elapsed=time.time()-started
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
        receipt=dict(epoch=1,updates=75,parameters=p['parameters'],protocol_sha256=ph,
            preclip_gradient_norms=norms,mean_training_loss=loss_sum/2400,mean_main_loss=main_sum/2400,
            mean_magnitude_loss=mag_sum/2400,train_seconds=elapsed,best=best,validation=actual)
        w.write(root/'EPOCH_001.json',receipt);print(dict(event='EPOCH_COMPLETE',**receipt),flush=True)
        selected=initial if best['epoch']==0 else actual;comparisons=[]
        for scope,value in [('actual_e1',actual),('selected',selected)]:
            for name,reference in [('parent',parent),('retained_control_e1',control)]:
                for count in (2,3):
                    a,b=[next(g for g in v['by_count'] if g['count']==count) for v in (reference,value)]
                    comparisons.append(dict(scope=scope,reference=name,count=count,nmse_delta=b['mean_nmse']-a['mean_nmse'],si_sdr_delta=b['mean_si_sdr']-a['mean_si_sdr'],weakest_nmse_delta=b['weakest_nmse']-a['weakest_nmse']))
        passed=all(v['nmse_delta']<0 and v['si_sdr_delta']>0 and v['weakest_nmse_delta']<=0 for v in comparisons if v['scope']=='selected')
        del opt,values,loss,estimates,logits,item
        import spectral_evaluation
        spectral_evaluation.run(root,p,net,validation)
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=ph,initial=initial,actual=actual,selected=selected,
            comparison=comparisons,criterion_met=passed,train_seconds=elapsed,updates=75,
            spectral_sha256=w.digest(root/'SPECTRAL.json'),heldout_read=False,independent_test=False,time=time.time())
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
            p=register(root,a.study.resolve(),a.predecessor.resolve());run(root,p,a.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
