"""Full retained-model TRAIN2 contract check of auxiliary-gradient subtraction."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
import torch
from wave_update_guard import WaveAccumulator

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('wave_guard_full_check_base',HERE/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
w=base.w;ROOT=base.ROOT


def run(output):
    torch.set_num_threads(2)
    protocol=ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json';p=w.read(protocol)
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    net=base.worker.make_model('retained_unet',Path(p['parent_checkpoint'])).train()
    params=list(net.parameters());acc=WaveAccumulator(net.named_parameters())
    data=base.worker.NativeMixtures(p['preparation'],'train_pack',1)
    rows=[];started=time.time()
    for index in (4,2):
        raw=data[index];count=int(raw['construction_count'])
        item={k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
              ('mixture','references','active','context_features','crop_start','construction_count')}
        estimates,logits=base.worker.predict(net,item)
        wave=base.worker.pit_waveform_loss(estimates,item['references'],item['active'],item['mixture'])['loss']
        ce=.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        target=torch.autograd.grad(wave/32,params,retain_graph=True,allow_unused=True)
        target=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(target,params)])
        extra=torch.autograd.grad(ce/32,acc.ce_parameters,retain_graph=True,allow_unused=True)
        acc.collect_ce(count,extra);del extra
        ((wave+ce)/32).backward();acc.collect(count)
        recovered=acc.waveform_groups()[count-2]
        difference=recovered-target
        error=float(difference.abs().max());relative=float(difference.norm()/target.norm().clamp_min(1e-30))
        assert error<1e-6 and relative<1e-5
        rows.append(dict(index=index,count=count,max_error=error,relative_error_norm=relative))
        acc.reset();net.zero_grad(set_to_none=True)
        del recovered,difference,target,wave,ce,estimates,logits,item
    files=[Path(__file__),HERE/'gradient.py',HERE/'wave_update_guard.py',HERE/'update_projection.py']
    value=dict(status='PASS',full_parameters=sum(p.numel() for p in params),rows=rows,
        parent_checkpoint_sha256=p['parent_checkpoint_sha256'],protocol_sha256=w.digest(protocol),
        source_sha256={str(p.relative_to(ROOT)):w.digest(p) for p in files},
        seconds=time.time()-started,device='cpu',optimizer_updates=0,heldout_read=False,
        limitation='Two fixed TRAIN examples validate implementation, not model performance')
    w.write(output,value);print(value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.output.resolve())
