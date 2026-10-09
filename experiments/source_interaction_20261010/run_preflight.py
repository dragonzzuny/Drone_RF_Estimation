"""Full retained U-Net vs source head: a bounded GPU TRAIN4 preflight.

Wait for the existing continuation and its independent final audit. Reuse no
fitted weights. This is an implementation/optimization check, not a validation
comparison or a claim that the architecture improves unseen drone recordings.
"""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/architecture_audit_20261009'))
import fit_diagnostic as fit
import gpu_start_guard as guard
import output_fit
import watch_epochs as watch
from models import build, predict
from native_data import NativeMixtures
from drone_rf.losses import pit_waveform_loss
from source_head import augment

ARMS = ('retained_unet', 'source_interaction')
CHECK = ROOT/'reports/2026-10-10/SOURCE_INTERACTION_CPU_CHECK.json'
AUDIT = ROOT/'reports/2026-10-09/LOW_LR_CONTINUATION_FINAL_AUDIT.json'


def register(root, dependency):
    if (root/'PROTOCOL.json').exists():
        raise ValueError('Duplicate preflight registration')
    prior = watch.read(dependency/'PROTOCOL.json')
    checked = watch.read(CHECK)
    if (checked['status'] != 'PASS' or checked['parameters'] != 32_180_747
            or not checked['exact_parent_prediction_at_initialization']
            or checked['recorded_iq_reads'] != 0):
        raise ValueError('Missing full CPU preflight')
    parent_receipt = ROOT/'local/native_frequency_20261009_v1/phase/CHECKPOINT.json'
    parent = watch.read(parent_receipt)
    if parent['selected_epoch'] != 2 or watch.digest(Path(parent['path'])) != parent['sha256']:
        raise ValueError('Retained best parent changed')
    # This registration opens metadata only; no RF samples are read here.
    train = NativeMixtures(prior['preparation'], 'train_pack', 1)
    indices = [int(i) for c in (2, 3) for i in np.flatnonzero(train.rows['count']==c)[:2]]
    sources = dict(prior['source_sha256'])
    for rel, digest in checked['source_sha256'].items():
        if watch.digest(ROOT/rel) != digest:
            raise ValueError('CPU checked source changed')
        sources[rel] = digest
    sources[str(Path(__file__).relative_to(ROOT))] = watch.digest(Path(__file__))
    plan = dict(status='REGISTERED_QUEUED_TRAIN4_PREFLIGHT', source_sha256=sources,
        dependency=str(dependency), dependency_protocol_sha256=watch.digest(dependency/'PROTOCOL.json'),
        dependency_audit=str(AUDIT), cpu_check_sha256=watch.digest(CHECK),
        parent_checkpoint=parent['path'], parent_checkpoint_sha256=parent['sha256'],
        parent_receipt_sha256=watch.digest(parent_receipt), parent_selected_epoch=2,
        preparation=prior['preparation'], preparation_sha256=prior['preparation_sha256'],
        gpu_display_policy=prior['gpu_display_policy'], train_indices=indices, role='train_pack',
        arms=list(ARMS), parameters={'retained_unet':32_142_859,'source_interaction':32_180_747},
        seed=0, steps_per_arm=32, effective_batch=4, microbatch=1, learning_rate=1e-4,
        optimizer='fresh AdamW1e-4 wd1e-4 clip1 FP32 TF32off in BOTH arms',
        loss='unchanged PIT NMSE+coherence+inactive/background plus0.1 count CE',
        input='63872 complex samples and unchanged time-mean long power context',
        initialization='same retained e2 parent; added readout zero; exact same initial predictions',
        change='shared source-axis attention head with zero-sum correction; all original backbone parameters train',
        ordering=list(ARMS), validation_read=False, heldout_read=False, weights_discarded=True,
        acceptance='finite gradients and lower fixed TRAIN NMSE at counts2/3; no architecture selection on TRAIN4',
        limitation='Development-informed, repeated TRAIN examples. Not generalization, convergence, or a SepTDA reproduction.',
        maximum_wait_seconds=24*3600, registered_at=time.time())
    for rel, digest in sources.items():
        if watch.digest(ROOT/rel) != digest:
            raise ValueError('Inherited source changed: '+rel)
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    watch.write(root/'PROTOCOL.json', plan)
    return plan


def verify(root, p):
    for rel, digest in p['source_sha256'].items():
        if watch.digest(ROOT/rel) != digest or watch.digest(root/'source_snapshot'/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)
    for path, digest in ((Path(p['dependency'])/'PROTOCOL.json',p['dependency_protocol_sha256']),
        (CHECK,p['cpu_check_sha256']), (Path(p['parent_checkpoint']),p['parent_checkpoint_sha256']),
        (Path(p['preparation'])/'PREPARATION.json',p['preparation_sha256'])):
        if watch.digest(path) != digest:
            raise ValueError('Preflight dependency changed')


def run(root, p, public):
    dependency = Path(p['dependency']); began = time.time()
    while not (dependency/'COMPLETE.json').exists() or not AUDIT.exists():
        if (dependency/'FAILURE.json').exists():
            raise RuntimeError('Predecessor failed')
        if time.time()-began > p['maximum_wait_seconds']:
            raise TimeoutError('Waiting for predecessor and audit')
        watch.write(root/'STATE.json',dict(status='WAITING_CONTINUATION_AND_AUDIT',
            pid=os.getpid(), model_updates=0, time=time.time()))
        time.sleep(30)
    audit = watch.read(AUDIT)
    if (audit['status'] != 'PASS' or audit['study_protocol_sha256'] != p['dependency_protocol_sha256']
            or audit['complete_sha256'] != watch.digest(dependency/'COMPLETE.json')):
        raise ValueError('Predecessor audit differs')
    with fit.base.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(root,p)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        watch.write(root/'GPU_START_CHECK.json', guard.wait_for_predecessor(
            watch.read(dependency/'STATE.json')['pid'],p['gpu_display_policy']))
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        train = NativeMixtures(p['preparation'],'train_pack',1)
        items = [fit.base.batch([train[i]]) for i in p['train_indices']]
        results = []; initial_predictions = None
        for arm in ARMS:
            verify(root,p)
            net = build('unet_mean')
            saved = torch.load(p['parent_checkpoint'], map_location='cpu', weights_only=False)
            if saved['best']['epoch'] != 2:
                raise ValueError('Wrong retained parent')
            net.load_state_dict(saved['model']); del saved
            if arm == 'source_interaction':
                augment(net)
            net.cuda().eval()
            assert sum(q.numel() for q in net.parameters()) == p['parameters'][arm]
            with torch.no_grad():
                actual = [tuple(t.cpu() for t in predict(net,item)) for item in items]
            if initial_predictions is None:
                initial_predictions = actual
            else:
                for a,b in zip(initial_predictions,actual):
                    for x,y in zip(a,b):
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
            del actual
            torch.manual_seed(0)
            opt = torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=1e-4,foreach=False)
            torch.cuda.reset_peak_memory_stats()
            history = [dict(step=0,**output_fit.score(net,items))]
            started=time.time(); attention_learned=False
            for step in range(1,p['steps_per_arm']+1):
                net.train(); opt.zero_grad(set_to_none=True)
                for item in items:
                    estimates,logits=predict(net,item)
                    loss=pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
                    loss=loss+.1*F.cross_entropy(logits,item['construction_count']-1)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    (loss/len(items)).backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                if not float(norm)>0:
                    raise ValueError('Zero gradient')
                if arm=='source_interaction' and step>1:
                    attention_learned |= float(net.output.qkv.weight.grad.norm())>0
                opt.step()
                if step in (1,8,16,32):
                    history.append(dict(step=step,**output_fit.score(net,items)))
                    watch.write(root/f'{arm}_HISTORY.json',dict(history=history,partial=step<32))
                    watch.write(root/'STATE.json',dict(status='GPU_TRAIN4_PREFLIGHT',arm=arm,
                        step=step,total=32,pid=os.getpid(),seconds=time.time()-started,time=time.time()))
            if arm=='source_interaction' and not attention_learned:
                raise ValueError('Attention branch did not receive gradient after zero initialization')
            if {int(s['step']) for s in opt.state.values()} != {32}:
                raise ValueError('Optimizer steps do not match the budget')
            if not all(torch.isfinite(q).all() for q in net.parameters()):
                raise ValueError('Nonfinite final parameters')
            result=dict(arm=arm,parameters=p['parameters'][arm],updates=32,history=history,
                finite_gradients=True,attention_gradient_after_first_step=attention_learned if arm=='source_interaction' else None,
                both_counts_improved=all(b['mean_nmse']<a['mean_nmse'] for a,b in zip(history[0]['by_count'],history[-1]['by_count'])),
                seconds=time.time()-started,peak_bytes=torch.cuda.max_memory_allocated(),weights_discarded=True)
            results.append(result);watch.write(root/f'{arm}_COMPLETE.json',result)
            del net,opt,estimates,logits,loss,norm
            gc.collect();torch.cuda.empty_cache()
        verify(root,p)
        complete=dict(status='COMPLETE',protocol_sha256=watch.digest(root/'PROTOCOL.json'),
            source_sha256=p['source_sha256'],parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
            initial_gpu_predictions_exactly_equal=True,results=results,validation_read=False,
            heldout_read=False,weights_discarded=True,time=time.time())
        watch.write(root/'COMPLETE.json',complete)
        watch.write(public,complete)
        watch.write(root/'STATE.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time()))
        print({'status':'COMPLETE','results':results},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('run','dependency','public'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();root=args.run.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=register(root,args.dependency.resolve())
            run(root,plan,args.public.resolve())
        except Exception:
            watch.write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
            raise
