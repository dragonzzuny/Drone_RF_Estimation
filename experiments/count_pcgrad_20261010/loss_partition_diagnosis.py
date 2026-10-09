"""Disentangle waveform and 0.1 count-CE gradients at the frozen parent.

TRAIN6 are fixed prior indices, not selected from new outcomes. No updates;
CPU only. Count averages are two-example means, not the running GPU batches.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pcgrad_base_diagnostic',HERE/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
w=base.w;ROOT=base.ROOT


def flatten(grads,params):
    return torch.cat([(v if v is not None else torch.zeros_like(p)).detach().flatten() for v,p in zip(grads,params)])


def run(root,study,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    p=w.read(study/'PROTOCOL.json');indices=[0,1,4,5,2,11]
    source=ROOT/'local/magnitude_parameter_gradient_20261010_v1/PROTOCOL.json'
    assert indices==w.read(source)['indices']
    hashes=dict(p['source_sha256']);hashes[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_COUNT_LOSS_PARTITION_DIAGNOSIS',source_sha256=hashes,
        indices=indices,parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
        source_selection_sha256=w.digest(source),study_protocol_sha256=w.digest(study/'PROTOCOL.json'),
        grouping='Mean of two examples per source count at retained parent; not running GPU batches',
        losses=['waveform PIT NMSE+coherence+inactive/background','0.1 count CE'],
        parameter_groups=['waveform_backbone','context_encoder_and_projection','count_head'],
        device='cpu',updates=0,heldout_read=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==sha
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    torch.set_num_threads(2);torch.manual_seed(0)
    net=base.worker.make_model('retained_unet',Path(p['parent_checkpoint'])).train()
    named=list(net.named_parameters());params=tuple(v for _,v in named);size=sum(v.numel() for v in params)
    assert size==32142859
    ranges={'waveform_backbone':[],'context_encoder_and_projection':[],'count_head':[]}
    offset=0
    for name,param in named:
        group=('count_head' if name.startswith('count_head.') else
               'context_encoder_and_projection' if name.startswith(('context_encoder.','context_projection.')) else 'waveform_backbone')
        ranges[group].append((offset,offset+param.numel()));offset+=param.numel()
    means=torch.zeros(6,size);data=base.worker.NativeMixtures(p['preparation'],'train_pack',1)
    rows=[];started=time.time()
    for index in indices:
        raw=data[index];count=int(raw['construction_count'])
        item={k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
              ('mixture','references','active','context_features','crop_start','construction_count')}
        outputs,logits=base.worker.predict(net,item)
        loss=base.worker.pit_waveform_loss(outputs,item['references'],item['active'],item['mixture'])['loss']
        ce=.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        gw=flatten(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True),params)
        gc=flatten(torch.autograd.grad(ce,params,allow_unused=True),params)
        assert torch.isfinite(gw).all() and torch.isfinite(gc).all()
        means[count-1].add_(gw/2);means[count+2].add_(gc/2)
        rows.append(dict(index=index,count=count,waveform_loss=float(loss.detach()),weighted_count_ce=float(ce.detach())))
        del gw,gc,loss,ce,outputs,logits,item
        w.write(root/'STATE.json',dict(status='CPU_GRADIENT_PARTITION',cases=len(rows),total=6,pid=os.getpid(),time=time.time()))
    groups=[]
    for name,intervals in ranges.items():
        gram=torch.zeros(6,6,dtype=torch.float64)
        for lo,hi in intervals:
            for start in range(lo,hi,200000):
                block=means[:,start:min(start+200000,hi)].double();gram.add_(block@block.T)
        groups.append(dict(group=name,parameters=sum(hi-lo for lo,hi in intervals),gram=gram.tolist()))
    full=sum(np.array(g['gram']) for g in groups);groups.append(dict(group='all',parameters=size,gram=full.tolist()))
    derived=[]
    for group in groups:
        a=np.array(group['gram']);main=a[:3,:3]+a[3:,3:]+a[:3,3:]+a[3:,:3]
        metrics={}
        for label,gram in [('waveform',a[:3,:3]),('count_ce',a[3:,3:]),('main',main)]:
            norms=np.sqrt(np.maximum(np.diag(gram),0.));den=norms[:,None]*norms[None,:]
            cosine=np.divide(gram,den,out=np.zeros_like(gram),where=den>0)
            metrics[label]=dict(norms=norms.tolist(),gram=gram.tolist(),cosine=cosine.tolist(),zero_norm=(norms==0).tolist())
        derived.append(dict(group=group['group'],parameters=group['parameters'],**metrics))
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==sha
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,
        gradient_row_order=['waveform_1','waveform_2','waveform_3','count_ce_1','count_ce_2','count_ce_3'],
        blocks=groups,derived=derived,seconds=time.time()-started,updates=0,heldout_read=False,
        limitation='TRAIN6 only, mean before cosine, different examples across counts; no causality or generalization proof')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=6,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds'],all_parameters=derived[-1]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','study','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.study.resolve(),a.public.resolve())
