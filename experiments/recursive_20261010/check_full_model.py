"""Three TRAIN cases, original 32M U-Net, two-pass backward, no optimizer.

Wait for the other CPU inference to finish; checkpoint activations to bound
RAM while the independent GPU trial runs. No GPU or validation data access.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from successive import predict
w=core.w

def state_digest(net):
    h=hashlib.sha256()
    for key,value in sorted(net.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(root,predecessor):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    algebra=ROOT/'reports/2026-10-10/SUCCESSIVE_ALGEBRA_CHECK.json'
    assert w.read(algebra)['status']=='PASS'
    sources=dict(old['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('successive.py'),Path(__file__).with_name('check_algebra.py'),
                 ROOT/'experiments/window_overlap_20261010/diagnose.py'):
        sources[str(path.relative_to(ROOT))]=w.digest(path)
    protocol=dict(status='REGISTERED_CPU_FULL_SUCCESSIVE_CHECK',sources=sources,indices=[0,4,2],
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        algebra_sha256=w.digest(algebra),updates=0,gpu_use=False,heldout_read=False,
        activation_checkpointing='non-reentrant both full U-Net passes; no stop-gradient',registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol)
    while not (predecessor/'COMPLETE.json').exists():
        if (predecessor/'FAILURE.json').exists():raise RuntimeError('CPU predecessor failed')
        w.write(root/'STATE.json',dict(status='WAITING_FOR_CPU_DIAGNOSTIC',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    torch.set_num_threads(2)
    net=core.base.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).train()
    assert sum(p.numel() for p in net.parameters())==32142859
    original_state=state_digest(net)
    data=core.base.worker.NativeMixtures(old['preparation'],'train_pack',1)
    prior={r['index']:r for r in w.read(ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json')['rows'] if r['model']=='parent/e0'}
    def bounded_predict(model,item):
        def forward(mixture,context,position):
            return core.base.worker.predict(model,dict(mixture=mixture,context_features=context,crop_start=position))
        return checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)
    rows=[];started=time.time()
    for index in protocol['indices']:
        net.zero_grad(set_to_none=True);item=core.batch(data[index])
        estimates,logits,trace=predict(net,item,bounded_predict)
        trace['remaining'].retain_grad()
        values=core.base.worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])
        objective=values['loss']+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        assert bool(torch.isfinite(objective))
        with torch.no_grad():
            parent,_=core.metrics(trace['first_predictions'],item)
            measured,_=core.metrics(estimates,item)
            for key in ('nmse','si_sdr'):
                assert np.max(np.abs(np.array(parent[key])-np.array(prior[index][key])))<2e-5
        objective.backward()
        gradients=[p.grad for p in net.parameters() if p.grad is not None]
        assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
        residual_gradient=trace['remaining'].grad
        assert residual_gradient is not None and bool(torch.isfinite(residual_gradient).all()) and float(residual_gradient.abs().sum())>0
        gradnorm=float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
        assert gradnorm>0 and state_digest(net)==original_state
        row=dict(index=index,count=int(item['construction_count']),objective=float(objective.detach()),
            gradient_norm=gradnorm,residual_gradient_norm=float(residual_gradient.abs().square().sum().sqrt()),
            first_slot=int(trace['first_slot']),second_slot=int(trace['second_slot']),
            all_gradients_finite=True,weights_unchanged=True,parent=parent,successive_initial=measured)
        rows.append(row)
        del estimates,logits,trace,values,objective,gradients,residual_gradient,item
        net.zero_grad(set_to_none=True);gc.collect()
        w.write(root/'STATE.json',dict(status='CPU_FULL_BACKWARD_CHECK',cases=len(rows),total_cases=3,pid=os.getpid(),time=time.time()))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(algebra)==protocol['algebra_sha256']
    result=dict(status='PASS',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,parameters=32142859,
        optimizer_steps=0,all_weights_unchanged=True,all_gradients_finite=True,two_stage_gradient_connected=True,
        activation_checkpointing=True,full_scale_geometry=True,seconds=time.time()-started,
        heldout_read=False,gpu_use=False,trained_result=False)
    w.write(root/'COMPLETE.json',result);w.write(ROOT/'reports/2026-10-10/SUCCESSIVE_FULL_MODEL_CHECK.json',result)
    w.write(root/'STATE.json',dict(status='PASS',cases=3,pid=os.getpid(),time=time.time()))
    print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--predecessor',type=Path,required=True)
    a=p.parse_args()
    try:run(a.run.resolve(),a.predecessor.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
