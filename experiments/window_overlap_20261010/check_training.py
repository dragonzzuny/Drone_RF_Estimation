"""Full-capacity CPU paired-window backward feasibility on fixed TRAIN case 4."""
import hashlib
import json
from pathlib import Path
import time
import torch
import diagnose
from paired import paired_items
from training_step import accumulate


def run():
    torch.set_num_threads(2);started=time.time()
    w=diagnose.w;ROOT=diagnose.ROOT;worker=diagnose.base.worker
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    data=worker.NativeMixtures(old['preparation'],'train_pack',1)
    net=worker.make_model('retained_unet',Path(old['parent_checkpoint'])).train()
    assert sum(q.numel() for q in net.parameters())==32142859
    assert not any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in net.modules())
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    a,b,offset=paired_items(data,4)
    a,b=diagnose.batch(a),diagnose.batch(b)
    value=accumulate(net,a,b,offset,worker.predict,worker.pit_waveform_loss,.1,32)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(torch.isfinite(p).all() for p in net.parameters())
    sources=dict(old['source_sha256'])
    for name in ('paired.py','training_step.py','check_training.py','diagnose.py','check_paired.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=w.digest(path)
    result=dict(status='PASS',parameters=32142859,training_index=4,source_count=2,
        offset_samples=offset,loss=value,gradient_norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)),
        all_parameter_gradients_finite=True,no_optimizer_step=True,updates=0,
        no_dropout_or_batchnorm=True,source_sha256=sources,
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        heldout_read=False,gpu_use=False,seconds=time.time()-started,
        limitation='One existing TRAIN case, full-model CPU backward feasibility; no learning or DEV gain demonstrated')
    w.write(ROOT/'reports/2026-10-10/PAIRED_WINDOW_TRAINING_CHECK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':run()
