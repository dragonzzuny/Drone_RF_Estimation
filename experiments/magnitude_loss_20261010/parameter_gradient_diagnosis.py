"""Full parameter-space gradients at frozen parent, two TRAIN48 cases/count.

No optimizer steps. Small, preselected diagnosis, not dataset-wide conflict or
an explanation of AdamW trajectories. Includes the original count CE gradient.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from objective import objective,worker,ROOT

w=worker.watch


def flatten(values,params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(values,params)])


def compare(a,b):
    aa=bb=ab=0.
    for start in range(0,a.numel(),1000000):
        x=a[start:start+1000000].double();y=b[start:start+1000000].double()
        aa+=float(x.dot(x));bb+=float(y.dot(y));ab+=float(x.dot(y))
    return dict(main_norm=aa**.5,weighted_magnitude_norm=bb**.5,
                ratio=(bb/aa)**.5 if aa>0 else None,
                cosine=ab/(aa*bb)**.5 if aa*bb>0 else None,real_inner_product=ab)


def run(study,source,root,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate full gradient diagnosis')
    p=w.read(study/'PROTOCOL.json');prior=w.read(source)
    torch.set_num_threads(2);torch.manual_seed(0)
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    indices=[r['index'] for count in (1,2,3) for r in [x for x in prior['rows'] if x['count']==count][:2]]
    assert len(indices)==6 and len(set(indices))==6
    sources=dict(p['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_CPU_PARAMETER_GRADIENTS',source_sha256=sources,
        study_protocol_sha256=w.digest(study/'PROTOCOL.json'),selection_source_sha256=w.digest(source),
        selection='First two previously registered TRAIN48 cases per count, in original order, before new waveform reads',
        indices=indices,parent_checkpoint_sha256=p['parent_checkpoint_sha256'],updates=0,
        device='cpu',count_CE_in_main=True,magnitude_weight=.1,heldout_read=False,registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).train();params=tuple(net.parameters())
    assert sum(p.numel() for p in params)==32142859
    aggregate={};rows=[];started=time.time()
    for index in indices:
        raw=data[index];count=int(raw['construction_count'])
        item={k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
              ('mixture','references','active','context_features','crop_start','construction_count')}
        output,logits=worker.predict(net,item);v=objective(output,logits,item)
        main=flatten(torch.autograd.grad(v['main'],params,retain_graph=True,allow_unused=True),params)
        mag=flatten(torch.autograd.grad(.1*v['magnitude'],params,allow_unused=True),params)
        assert torch.isfinite(main).all() and torch.isfinite(mag).all()
        values=compare(main,mag)
        rows.append(dict(index=index,count=count,main_loss=float(v['main'].detach()),magnitude_loss=float(v['magnitude'].detach()),**values))
        if count not in aggregate:aggregate[count]=[main/2,mag/2]
        else:aggregate[count][0].add_(main/2);aggregate[count][1].add_(mag/2)
        del main,mag,v,output,logits,item
        w.write(root/'STATE.json',dict(status='CPU_PARAMETER_GRADIENTS',cases=len(rows),total=6,pid=os.getpid(),time=time.time()))
    within=[dict(count=n,**compare(*aggregate[n])) for n in (1,2,3)]
    cross=[dict(main_count=n,magnitude_count=m,**compare(aggregate[n][0],aggregate[m][1])) for n in (1,2,3) for m in (1,2,3)]
    combined=compare((aggregate[2][0]+aggregate[3][0])/2,aggregate[1][1])
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),indices=indices,rows=rows,
        within_count=within,cross_count=cross,multisource_main_vs_single_magnitude=combined,
        full_parameters=32142859,updates=0,heldout_read=False,independent_test=False,seconds=time.time()-started,
        limitation='Only two TRAIN48 examples/count at frozen parent; local gradients, not full-dataset AdamW updates or generalization')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=6,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',within_count=within,multisource_main_vs_single_magnitude=combined),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','source','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();run(a.study.resolve(),a.source.resolve(),a.run.resolve(),a.public.resolve())
