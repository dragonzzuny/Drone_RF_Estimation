"""Full retained U-Net, zero adapter equivalence and three-case CPU backward.

Feasibility check, no optimizer or development/held-out data use. Registered
separately from the immutable GPU experiments. Activation checkpointing
includes the scoped input hook on every recomputation.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from conditioning import augment, predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/window_overlap_20261010'))
sys.path.insert(0, str(ROOT/'experiments/recursive_20261010'))
import diagnose as core
from successive import predict as original_predict, strongest_source
from drone_rf.waveform import analyze, synthesize
w = core.w


def state_digest(net):
    h = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'PROTOCOL.json').exists()
    old = w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    algebra = ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_ALGEBRA_CHECK.json'
    check = w.read(algebra); assert check['status'] == 'PASS'
    sources = dict(old['source_sha256']); sources.update(check['sources'])
    for path in (Path(__file__), ROOT/'experiments/recursive_20261010/successive.py',
                 ROOT/'experiments/window_overlap_20261010/diagnose.py'):
        sources[str(path.relative_to(ROOT))] = w.digest(path)
    protocol = dict(status='REGISTERED_CPU_ORIGINAL_CONDITIONING_CHECK', sources=sources,
        indices=[0,4,2],selection='Same pre-existing TRAIN1/2/3 backward-check cases',
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        algebra_sha256=w.digest(algebra),updates=0,gpu_use=False,heldout_read=False,
        architecture='Full shared 32142859 base plus2304 zero-initialized second-pass first-convolution weights',
        registered_at=time.time())
    for rel, sha in sources.items():
        assert w.digest(ROOT/rel) == sha
        dest=root/'source_snapshot'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol); torch.set_num_threads(2)
    net = core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).train()
    augment(net); assert sum(p.numel() for p in net.parameters()) == 32145163
    digest = state_digest(net)
    data = core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    rows=[]; started=time.time()
    bounded = lambda function,*args: checkpoint(function,*args,use_reentrant=False)
    for index in protocol['indices']:
        w.write(root/'STATE.json',dict(status='CPU_FULL_BACKWARD',index=index,cases=len(rows),total_cases=3,pid=os.getpid(),time=time.time()))
        item=core.batch(data[index]); net.zero_grad(set_to_none=True)
        with torch.no_grad(): expected,expected_logits,_=original_predict(net,item,core.base.worker.predict)
        actual,logits,trace = predict(net,item,core.base.worker.predict,analyze,synthesize,strongest_source,bounded)
        assert torch.equal(actual,expected) and torch.equal(logits,expected_logits)
        trace['remaining'].retain_grad()
        loss=core.base.worker.pit_waveform_loss(actual,item['references'],item['active'],item['mixture'])['loss']
        loss=loss+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        assert torch.isfinite(loss)
        with torch.no_grad(): metric,_=core.metrics(actual,item)
        loss.backward()
        grads=[p.grad for p in net.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        adapter_norm=float(net.original_conditioning.weight.grad.norm())
        residual_norm=float(trace['remaining'].grad.abs().square().sum().sqrt())
        assert adapter_norm>0 and residual_norm>0 and torch.isfinite(trace['remaining'].grad).all()
        assert len(net.down[0][0]._forward_hooks)==0 and state_digest(net)==digest
        rows.append(dict(index=index,count=int(item['construction_count']),objective=float(loss.detach()),
            adapter_gradient_norm=adapter_norm,residual_gradient_norm=residual_norm,
            initial_waveforms_exact=True,initial_count_logits_exact=True,all_parameter_gradients_finite=True,
            no_leaked_hooks=True,weights_unchanged=True,metrics=metric))
        del item,actual,expected,expected_logits,logits,trace,loss,grads
        net.zero_grad(set_to_none=True);gc.collect()
    for rel,sha in sources.items(): assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(algebra)==protocol['algebra_sha256'] and state_digest(net)==digest
    result=dict(status='PASS',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,
        parameters=32145163,base_parameters=32142859,extra_parameters=2304,optimizer_steps=0,
        full_model=True,full_input_samples=63872,activation_checkpointing=True,zero_initial_function_exact=True,
        heldout_read=False,gpu_use=False,trained_result=False,seconds=time.time()-started,
        limitation='Three fixed TRAIN cases establish implementation feasibility, not improvement or generalization')
    w.write(root/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-10/ORIGINAL_CONDITIONING_FULL_CHECK.json',result)
    w.write(root/'STATE.json',dict(status='PASS',cases=3,pid=os.getpid(),time=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    try: run(a.run.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
