"""Matched full-model TRAIN4 probe of balanced source-head inputs.

Runs only after the registered head-LR probe. Both arms use the same fixed new
head LR=1e-3 and original LR=1e-5. This is an optimization/structure diagnostic,
not validation. All probe weights are discarded.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from torch.nn import functional as F
import head_learning_rate_probe as prior
from balanced_source_head import augment

worker, w = prior.worker, prior.w
ARMS = ('unbalanced', 'balanced')
CHECK = w.ROOT/'reports/2026-10-10/BALANCED_SOURCE_HEAD_CPU_CHECK.json'


def make_model(arm, parent):
    if arm == 'unbalanced':
        return worker.make_model('source_interaction', parent)
    net = worker.make_model('retained_unet', parent)
    return augment(net)


def register(root, dependency):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate balanced-head probe')
    base = w.read(dependency/'PROTOCOL.json')
    checks = w.read(CHECK)
    if checks['status'] != 'PASS':
        raise ValueError('CPU full-model check required')
    sources = dict(base['source_sha256'])
    sources.update(checks['source_sha256'])
    for path in (Path(__file__), Path(prior.__file__)):
        sources[str(path.relative_to(w.ROOT))] = w.digest(path)
    p = dict(status='REGISTERED_CPU_CHECKED_GPU_TRAIN4_BALANCE_PROBE', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=w.digest(dependency/'PROTOCOL.json'),
        cpu_check_sha256=w.digest(CHECK), preparation=base['preparation'],
        preparation_sha256=base['preparation_sha256'], parent_checkpoint=base['parent_checkpoint'],
        parent_checkpoint_sha256=base['parent_checkpoint_sha256'], gpu_display_policy=base['gpu_display_policy'],
        train_indices=base['train_indices'], arms=list(ARMS), updates_per_arm=64,
        effective_batch=4, microbatch=1, old_lr=1e-5, new_lr=1e-3,
        lr_rule='Fixed before the predecessor LR-probe result, identical in both arms; not selected as optimal',
        seed=0, parameters_per_arm=32_180_747, original_parameters=32_142_859, added_parameters=37_888,
        observations=[0,1,8,16,32,64],
        loss='Unmodified PIT NMSE+coherence+inactive/background plus0.1 count CE',
        optimizer='Fresh AdamW wd1e-4 clip1 FP32 TF32off, full backbone trainable',
        change='Parameter-free feature/source group normalization at the extra head input only',
        initialization='Same parent and zero readout; no preceding-probe checkpoint reuse',
        acceptance='At final64, counts2/3 raw NMSE lower AND complex SI-SDR higher than unbalanced control; report all observation steps and direct effect',
        selection='No checkpoint selection or retention', gpu_lock='existing shared neural_queue.lock',
        heldout_read=False, validation_read=False, weights_discarded=True,
        limitation='Same four previously examined TRAIN mixtures, no convergence or generalization claim',
        registered_at=time.time(), maximum_wait_seconds=24*3600)
    for rel, sha in sources.items():
        if w.digest(w.ROOT/rel) != sha:
            raise ValueError('Source changed before registration')
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(w.ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p)
    return p


def verify(root,p):
    for rel,sha in p['source_sha256'].items():
        if w.digest(w.ROOT/rel)!=sha or w.digest(root/'source_snapshot'/rel)!=sha:
            raise ValueError('Frozen probe source changed: '+rel)
    for path,sha in ((Path(p['dependency'])/'PROTOCOL.json',p['dependency_protocol_sha256']),
        (CHECK,p['cpu_check_sha256']),(Path(p['parent_checkpoint']),p['parent_checkpoint_sha256']),
        (Path(p['preparation'])/'PREPARATION.json',p['preparation_sha256'])):
        if w.digest(path)!=sha:raise ValueError('Frozen dependency changed')


def run(root,p,public):
    dependency=Path(p['dependency'])
    while not (dependency/'COMPLETE.json').exists():
        if (dependency/'FAILURE.json').exists():raise RuntimeError('Preceding probe failed')
        if time.time()-p['registered_at']>p['maximum_wait_seconds']:raise TimeoutError('Wait expired')
        w.write(root/'STATE.json',dict(status='WAITING_HEAD_LR_PROBE',updates=0,pid=os.getpid(),time=time.time()))
        time.sleep(30)
    before=w.read(dependency/'COMPLETE.json')
    if before['protocol_sha256']!=p['dependency_protocol_sha256'] or not before['initial_predictions_exactly_equal']:
        raise ValueError('Preceding probe identity failed')
    with worker.fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify(root,p)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        w.write(root/'GPU_START_CHECK.json',prior.guard.wait_for_predecessor(w.read(dependency/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=worker.NativeMixtures(p['preparation'],'train_pack',1)
        items=[worker.fit.base.batch([train[i]]) for i in p['train_indices']]
        initial=None;results=[]
        for arm in ARMS:
            verify(root,p)
            net=make_model(arm,Path(p['parent_checkpoint'])).cuda().eval()
            with torch.no_grad():values=[tuple(x.cpu() for x in worker.predict(net,item)) for item in items]
            if initial is None:initial=values
            else:
                for a,b in zip(initial,values):
                    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            del values
            torch.manual_seed(0)
            opt=prior.optimizer(net,p['new_lr'])
            history=[dict(step=0,**prior.output_fit.score(net,items))];began=time.time()
            for step in range(1,p['updates_per_arm']+1):
                net.train();opt.zero_grad(set_to_none=True)
                for item in items:
                    estimates,logits=worker.predict(net,item)
                    loss=prior.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*F.cross_entropy(logits,item['construction_count']-1)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    (loss/4).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                if step in p['observations']:
                    history.append(dict(step=step,**prior.output_fit.score(net,items)))
                    w.write(root/f'{arm}_HISTORY.json',dict(history=history,partial=step<64))
                    w.write(root/'STATE.json',dict(status='GPU_TRAIN4_BALANCE_PROBE',arm=arm,step=step,total=64,pid=os.getpid(),time=time.time()))
            if {int(s['step']) for s in opt.state.values()}!={64} or not all(torch.isfinite(v).all() for v in net.parameters()):
                raise ValueError('Invalid final optimizer or model')
            results.append(dict(arm=arm,updates=64,history=history,direct_head_effect=prior.effect(net,items),
                final_readout_norm=float(net.output.readout.weight.norm()),
                optimizer_groups=[dict(lr=g['lr'],parameters=sum(t.numel() for t in g['params'])) for g in opt.param_groups],
                seconds=time.time()-began))
            w.write(root/f'{arm}_COMPLETE.json',results[-1])
            del net,opt,item,estimates,logits,loss;gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),results=results,
            initial_predictions_exactly_equal=True,heldout_read=False,validation_read=False,weights_discarded=True,time=time.time())
        w.write(root/'COMPLETE.json',result);w.write(public,result)
        w.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
        print(dict(status='COMPLETE',arms=list(ARMS),updates_per_arm=64),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('dependency','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'.run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=register(root,args.dependency.resolve());run(root,p,args.public.resolve())
    except Exception:
        w.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
