"""CPU TRAIN6 receiver-coordinate sensitivity; all inputs remain mixture-only."""
import argparse
import importlib.util
import itertools
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from coordinates import augment,oscillator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/count_pcgrad_20261010'))
spec=importlib.util.spec_from_file_location('count_for_coordinates',ROOT/'experiments/count_pcgrad_20261010/train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from drone_rf.waveform import waveform_metrics
w=base.w


def batch(raw):
    return {k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
        ('mixture','references','active','context_features','crop_start','construction_count')}


def evaluate(estimates,logits,item):
    m=waveform_metrics(estimates,item['references'],item['active'],item['mixture'])
    count=int(item['construction_count']);p=m['reference_power'][0,:count].tolist()
    row=dict(count=count,nmse=m['nmse'][0,:count].tolist(),si_sdr=m['si_sdr'][0,:count].tolist(),
        reference_power=p,weakest_index=int(np.argmin(p)),predicted_count=int(logits.argmax(-1))+1,
        sum_relative_error=float(m['sum_relative_error'][0]))
    assert np.isfinite(row['nmse']+row['si_sdr']).all() and row['sum_relative_error']<1e-10
    return row


def align_to_anchor(estimates,anchor):
    # Only predicted signals participate. The fourth/background role is fixed.
    perms=list(itertools.permutations(range(3)))
    costs=[float((estimates[:,:3][:,list(p)]-anchor[:,:3]).abs().square().sum()) for p in perms]
    k=int(np.argmin(costs));return estimates[:,list(perms[k])+[3]],list(perms[k])


def run(root,public,wait_for):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    prior_path=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json'
    prior=w.read(prior_path);reference=[r for r in prior['rows'] if r['model']=='parent/e0']
    indices=[r['index'] for c in (1,2,3) for r in [q for q in reference if q['count']==c][:2]]
    source=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    check_path=ROOT/'reports/2026-10-10/RECEIVER_COORDINATE_CPU_CHECK.json';check=w.read(check_path)
    assert check['status']=='PASS' and check['training_implemented'] is False
    hashes=dict(source['source_sha256']);hashes.update(check['source_sha256'])
    hashes[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    p=dict(status='REGISTERED_CPU_COORDINATE_DIAGNOSIS',source_sha256=hashes,
        check_sha256=w.digest(check_path),prior_sha256=w.digest(prior_path),indices=indices,
        parent=source['parent_checkpoint'],parent_sha256=source['parent_checkpoint_sha256'],
        preparation=source['preparation'],preparation_sha256=source['preparation_sha256'],
        steps=[0,-2,-1,1,2],step_hz=100_000_000/64,updates=0,heldout_read=False,
        inference='Same NCO on all components, invert it on predictions; align three estimated source slots only, fixed background',
        selection='Unchanged first two cases per count from previous TRAIN48; no DEV or best-example selection',
        registered_at=time.time())
    w.write(root/'PROTOCOL.json',p)
    for rel,sha in hashes.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    def verify():
        for rel,sha in hashes.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
        assert w.digest(Path(p['parent']))==p['parent_sha256']
        assert w.digest(Path(p['preparation'])/'PREPARATION.json')==p['preparation_sha256']
    verify()
    if wait_for:
        w.write(root/'STATE.json',dict(status='WAITING_FOR_CPU_FIT',pid=os.getpid(),time=time.time()))
        while not (wait_for/'COMPLETE.json').exists():
            if (wait_for/'FAILURE.json').exists():raise RuntimeError('CPU predecessor failed')
            time.sleep(5)
    verify();torch.set_num_threads(2);started=time.time()
    train=base.worker.NativeMixtures(p['preparation'],'train_pack',1)
    net=base.worker.make_model('retained_unet',Path(p['parent'])).eval()
    rows=[];done=0
    with torch.inference_mode():
        for index in indices:
            raw=train[index];item=batch(raw);predictions=[];scores=[];anchor=None
            for steps in p['steps']:
                shifted=augment(raw,steps);pred,logits=base.worker.predict(net,batch(shifted))
                pred=pred*torch.from_numpy(oscillator(pred.shape[-1],raw['crop_start'],steps).conj().copy())[None,None]
                if anchor is None:anchor=pred.clone()
                aligned,order=align_to_anchor(pred,anchor)
                # All four outputs sum to the unshifted mixture after inversion.
                drift=float((aligned-anchor).abs().square().sum()/item['mixture'].abs().square().sum())
                row=evaluate(pred,logits,item);row.update(index=index,mode='coordinate',steps=steps,
                    prediction_only_alignment=order,equivariance_relative_error=drift)
                rows.append(row);predictions.append(aligned);scores.append(logits);done+=1
                if steps==0:
                    prior_row=next(r for r in reference if r['index']==index)
                    assert max(abs(a-b) for a,b in zip(row['nmse'],prior_row['nmse']))<2e-5
                w.write(root/'PARTIAL.json',dict(rows=rows))
                w.write(root/'STATE.json',dict(status='CPU_COORDINATE_DIAGNOSIS',inferences=done,total=30,pid=os.getpid(),time=time.time()))
            average=torch.stack(predictions).mean(0);mean_logits=torch.stack(scores).mean(0)
            row=evaluate(average,mean_logits,item);row.update(index=index,mode='five_coordinate_average',steps=None)
            rows.append(row)
    verify()
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),rows=rows,
        seconds=time.time()-started,indices=indices,inferences=30,updates=0,heldout_read=False,
        inference_alignment_used_references=False,
        limitation='Six fixed TRAIN cases; coordinate sensitivity or averaging is not proof that augmentation training will improve DEV')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',inferences=30,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','public'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wait-for',type=Path);a=parser.parse_args()
    try:run(a.run.resolve(),a.public.resolve(),a.wait_for.resolve() if a.wait_for else None)
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
