"""CPU TRAIN48 output-space gradient diagnosis; never updates model weights.

Gradients on independent output waveforms differ from network parameter-space
gradients. Also report gradients projected onto the mixture-consistent tangent
space, whose four output perturbations sum to zero.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from objective import objective,worker,ROOT

w=worker.watch


def compare(a,b):
    a=a.to(torch.complex128);b=b.to(torch.complex128)
    an=a.abs().square().sum().sqrt();bn=b.abs().square().sum().sqrt()
    return dict(main_norm=float(an),weighted_magnitude_norm=float(bn),
        ratio=float(bn/an) if an>0 else None,
        cosine=float((a.conj()*b).real.sum()/(an*bn)) if an*bn>0 else None)


def run(study,root,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Duplicate CPU gradient diagnosis')
    p=w.read(study/'PROTOCOL.json');torch.set_num_threads(2)
    data=worker.NativeMixtures(p['preparation'],'train_pack',1)
    used=Counter();indices=[]
    for i,row in enumerate(data.rows):
        n=int(row['count']);names=tuple(data.library.clips[int(j)]['category'] for j in row['indices'][:n])
        key=names+tuple(map(float,row['levels'][:n]))
        if used[key]<2:used[key]+=1;indices.append(i)
    assert len(indices)==48
    sources=dict(p['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_TRAIN48_OUTPUT_GRADIENT',parent_sha256=p['parent_checkpoint_sha256'],
        magnitude_study_protocol_sha256=w.digest(study/'PROTOCOL.json'),source_sha256=sources,
        indices=indices,selection='First two TRAIN epoch1 rows per category tuple and nominal levels, same existing TRAIN48 rule',
        magnitude_weight=.1,updates=0,heldout_read=False,device='cpu',registered_at=time.time())
    w.write(root/'PROTOCOL.json',protocol)
    assert w.digest(Path(p['parent_checkpoint']))==p['parent_checkpoint_sha256']
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    net=worker.make_model('retained_unet',Path(p['parent_checkpoint'])).eval();rows=[];start=time.time()
    for index in indices:
        raw=data[index];item={k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
            ('mixture','references','active','context_features','crop_start','construction_count')}
        with torch.no_grad():pred,logits=worker.predict(net,item)
        output=pred.detach().requires_grad_();value=objective(output,logits,item)
        main=torch.autograd.grad(value['main'],output,retain_graph=True)[0]
        mag=torch.autograd.grad(.1*value['magnitude'],output)[0]
        assert torch.isfinite(main).all() and torch.isfinite(mag).all()
        # Orthogonal projection for the exact four-output mixture-sum constraint.
        pm=main-main.mean(1,keepdim=True);pg=mag-mag.mean(1,keepdim=True)
        n=int(raw['construction_count']);source=data.rows[index]
        clips=[data.library.clips[int(j)] for j in source['indices'][:n]]
        powers=item['references'][0,:n].abs().square().mean(-1).tolist()
        rows.append(dict(index=index,count=n,categories=[c['category'] for c in clips],
            reference_power=powers,weakest_index=int(np.argmin(powers)),main_loss=float(value['main'].detach()),
            magnitude_loss=float(value['magnitude'].detach()),raw_output=compare(main,mag),
            projected_output=compare(pm,pg),
            components=[dict(reference_index=k,slot=int(slot),**compare(main[0,slot],mag[0,slot]))
                for k,slot in enumerate(value['assignment'][0,:n].tolist())]))
        w.write(root/'STATE.json',dict(status='CPU_DIAGNOSIS',cases=len(rows),total=48,pid=os.getpid(),time=time.time()))
    groups=[]
    for count in (1,2,3):
        part=[r for r in rows if r['count']==count]
        for mode in ('raw_output','projected_output'):
            groups.append(dict(count=count,cases=len(part),space=mode,
                mean_ratio=statistics.mean(r[mode]['ratio'] for r in part),
                mean_cosine=statistics.mean(r[mode]['cosine'] for r in part),
                negative_cosines=sum(r[mode]['cosine']<0 for r in part)))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==sha
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,by_count=groups,
        parent_sha256=p['parent_checkpoint_sha256'],model_updates=0,heldout_read=False,
        interpretation='Fixed TRAIN48 output-space directions at retained parent; not full parameter gradients, generalization, or proof of optimization conflict',
        seconds=time.time()-start)
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=48,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',by_count=groups),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('study','run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();run(a.study.resolve(),a.run.resolve(),a.public.resolve())
