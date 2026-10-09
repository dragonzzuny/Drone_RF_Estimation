"""Full-capacity CPU backward check on fixed TRAIN cases; zero updates."""
import gc
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from affinity import Capture,targets,loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/window_overlap_20261010'))
import diagnose as core
from drone_rf.waveform import analyze


def run():
    torch.set_num_threads(2);w=core.w;worker=core.base.worker
    p=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).train()
    initial={k:v.clone() for k,v in net.state_dict().items()}
    first=core.batch(data[4])
    with torch.no_grad():expected=worker.predict(net,first)
    capture=Capture(net);capture.enabled=True
    with torch.no_grad():actual=worker.predict(net,first);embedding=capture.take()
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert embedding.shape==(1,64000,16)
    parameters=sum(q.numel() for q in net.parameters());assert parameters==32142859+1040
    rows=[];started=time.time()
    for index in (4,2):
        item=core.batch(data[index]);net.zero_grad(set_to_none=True)
        pred,logits=worker.predict(net,item);e=capture.take()
        with torch.no_grad():power=F.avg_pool2d(analyze(item['references'][0]).abs().square()[None],2)
        y,weight=targets(power,'soft');hy,hw=targets(power,'hard')
        assert torch.equal(weight,hw) and bool(torch.allclose(e.norm(dim=-1),torch.ones_like(weight).float(),atol=1e-6))
        main=worker.pit_waveform_loss(pred,item['references'],item['active'],item['mixture'])['loss']
        main=main+.1*F.cross_entropy(logits,item['construction_count']-1)
        soft=loss(e,y,weight);hard=loss(e,hy,weight)
        (main+.1*soft).backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in net.parameters())
        assert float(net.source_affinity.weight.grad.norm())>0
        rows.append(dict(index=index,count=int(item['construction_count']),main_loss=float(main.detach()),
            hard_affinity_loss=float(hard.detach()),soft_affinity_loss=float(soft.detach()),
            weighted_soft_to_main_loss=float((.1*soft/main).detach()),
            gradient_norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)),
            head_gradient_nonzero=True,all_gradients_finite=True,weight_sum=float(weight.sum()),embedding_shape=list(e.shape)))
        w.write(ROOT/'local/source_affinity_cpu_check_20261010_STATE.json',dict(status='CPU_BACKWARD',completed=len(rows),total=2,pid=os.getpid(),time=time.time()))
        del pred,logits,e,power,y,weight,hy,hw,main,soft,hard;gc.collect()
    assert all(torch.equal(initial[k],net.state_dict()[k]) for k in initial)
    hashes=dict(p['source_sha256'])
    for f in ('affinity.py','check.py','check_training.py'):
        q=Path(__file__).with_name(f);hashes[str(q.relative_to(ROOT))]=w.digest(q)
    result=dict(status='PASS',rows=rows,parameters=parameters,base_parameters=32142859,aux_parameters=1040,
        initial_predictions_exact=True,initial_count_logits_exact=True,all_parent_tensors_unchanged=True,
        updates=0,gpu_use=False,heldout_read=False,training_indices=[4,2],seconds=time.time()-started,
        source_sha256=hashes,parent_sha256=p['parent_checkpoint_sha256'],preparation_sha256=p['preparation_sha256'],
        limitation='Two fixed TRAIN cases: full model backward feasibility only, no optimizer step or generalization evidence')
    w.write(ROOT/'reports/2026-10-10/SOURCE_AFFINITY_TRAINING_CHECK.json',result)
    w.write(ROOT/'local/source_affinity_cpu_check_20261010_STATE.json',dict(status='COMPLETE',completed=2,pid=os.getpid(),time=time.time()))
    print({k:v for k,v in result.items() if k!='source_sha256'},flush=True);capture.close()


if __name__=='__main__':run()
