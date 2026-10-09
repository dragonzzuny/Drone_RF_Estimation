"""Disentangle overlap matching error from waveform error on the same TRAIN6."""
import argparse
import itertools
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import diagnose as core

w=core.w;ROOT=core.ROOT


def coherence_align(pred,anchor):
    a=anchor[:,:3].to(torch.complex128);b=pred[:,:3].to(torch.complex128)
    a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
    cross=(a[:,:,None]*b[:,None].conj()).sum(-1).abs().square()
    denom=a.abs().square().sum(-1)[:,:,None]*b.abs().square().sum(-1)[:,None]
    score=torch.where(denom>0,cross/denom.clamp_min(1e-300),torch.zeros_like(cross)).clamp(0,1)
    permutations=list(itertools.permutations(range(3)))
    values=[float(score[0,range(3),list(p)].sum()) for p in permutations]
    order=list(permutations[int(np.argmax(values))])+[3]
    return pred[:,order],order,score[0].tolist()


def run(root,public):
    root.mkdir(parents=True,exist_ok=True);assert not (root/'PROTOCOL.json').exists()
    old_root=ROOT/'local/window_overlap_20261010_v1';p=w.read(old_root/'PROTOCOL.json')
    previous=w.read(old_root/'COMPLETE.json')
    assert w.read(ROOT/'reports/2026-10-10/WINDOW_OVERLAP_AUDIT.json')['status']=='PASS'
    hashes=dict(p['source_sha256']);hashes[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_CPU_OVERLAP_ALIGNMENT',parent_protocol_sha256=w.digest(old_root/'PROTOCOL.json'),
        preceding_result_sha256=w.digest(old_root/'COMPLETE.json'),source_sha256=hashes,
        indices=p['indices'],offsets=p['offsets'],updates=0,inferences=18,heldout_read=False,gpu_use=False,
        comparison=['prediction_mse','prediction_coherence','oracle_reference_assignment_diagnostic_only'],
        selection='Same six previously reported TRAIN examples, including all failures; posthoc mechanism follow-up',
        registered_at=time.time())
    for rel,sha in hashes.items():
        assert w.digest(ROOT/rel)==sha
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    w.write(root/'PROTOCOL.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(0)
    synthetic=torch.randn(1,4,257,dtype=torch.complex128)
    changed=synthetic[:,[2,0,1,3]]*torch.tensor([2.,.1,4.,1.])[None,:,None]
    _,order,_=coherence_align(changed,synthetic)
    assert order==[1,2,0,3]
    data=core.base.worker.NativeMixtures(p['preparation'],'train_pack',1)
    net=core.base.worker.make_model('retained_unet',Path(p['parent'])).eval()
    rows=[];details=[];started=time.time()
    with torch.inference_mode():
        for case,index in enumerate(p['indices']):
            start=int(data.rows[index]['crop_start']);length=data.length;lo=16384;hi=length-16384
            original=data[index];common=core.batch(original)
            for key in ('mixture','references'):common[key]=common[key][...,lo:hi]
            predictions=[];assignments=[]
            for j,offset in enumerate(p['offsets']):
                data.rows[index]['crop_start']=start+offset
                try:raw=data[index]
                finally:data.rows[index]['crop_start']=start
                item=core.batch(raw)
                for key in ('mixture','references'):assert torch.equal(item[key][...,lo-offset:hi-offset],common[key])
                full,_=core.base.worker.predict(net,item);pred=full[...,lo-offset:hi-offset]
                _,assignment=core.metrics(pred,common)
                predictions.append(pred);assignments.append(assignment)
                w.write(root/'STATE.json',dict(status='CPU_ALIGNMENT_DIAGNOSIS',inferences=3*case+j+1,total=18,pid=os.getpid(),time=time.time()))
            anchor=predictions[0];fixed=assignments[0]
            central,_=core.metrics(anchor,common,fixed);central.update(index=index,alignment='central',averaging='none')
            rows.append(central)
            for name in protocol['comparison']:
                aligned=[]
                for offset,pred,assignment in zip(p['offsets'],predictions,assignments):
                    if name=='prediction_mse':value,order,_=core.align(pred,anchor)
                    elif name=='prediction_coherence':value,order,_=coherence_align(pred,anchor)
                    else:
                        order=[0,1,2,3]
                        for target in range(3):order[int(fixed[0,target])]=int(assignment[0,target])
                        value=pred[:,order]
                    aligned.append(value)
                    row,_=core.metrics(value,common,fixed)
                    row.update(index=index,alignment=name,offset=offset,order=order)
                    details.append(row)
                weights=torch.stack([torch.sin(torch.pi*(torch.arange(lo,hi)-offset+.5)/length).square() for offset in p['offsets']])
                weights/=weights.sum(0);stack=torch.stack(aligned)
                for average,pred in [('uniform',stack.mean(0)),('center_weighted',(stack*weights[:,None,None]).sum(0))]:
                    row,_=core.metrics(pred,common,fixed)
                    independent,_=core.metrics(pred,common)
                    row.update(index=index,alignment=name,averaging=average,
                        independently_scored_nmse=independent['nmse'],independently_scored_si_sdr=independent['si_sdr'])
                    rows.append(row)
                    if name=='prediction_mse':
                        mode='uniform_average' if average=='uniform' else 'center_weighted_average'
                        earlier=next(q for q in previous['rows'] if q['index']==index and q['mode']==mode)
                        assert max(abs(a-b) for a,b in zip(row['nmse'],earlier['nmse']))<1e-12
            w.write(root/'PARTIAL.json',dict(rows=rows,details=details))
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(Path(p['parent']))==p['parent_sha256']
    summary=[]
    for alignment,average in [('central','none')]+[(a,b) for a in protocol['comparison'] for b in ('uniform','center_weighted')]:
        for count in (1,2,3):
            selected=[q for q in rows if q['alignment']==alignment and q['averaging']==average and q['count']==count]
            summary.append(dict(alignment=alignment,averaging=average,count=count,cases=len(selected),
                nmse=float(np.mean([np.mean(q['nmse']) for q in selected])),
                si_sdr=float(np.mean([np.mean(q['si_sdr']) for q in selected])),
                weakest_nmse=float(np.mean([q['nmse'][q['weakest_index']] for q in selected]))))
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,details=details,
        by_count=summary,seconds=time.time()-started,updates=0,inferences=18,heldout_read=False,gpu_use=False,
        baseline_reproduced=True,synthetic_gain_and_permutation_check=True,
        limitation='Posthoc TRAIN6 diagnosis. Reference-based matching is an oracle diagnostic, never a deployable method. Coherence matching only changes order, not complex gain or phase.')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',inferences=18,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds'],by_count=summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    try:run(a.run.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
