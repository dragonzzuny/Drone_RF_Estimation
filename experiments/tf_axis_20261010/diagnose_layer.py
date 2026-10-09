"""CPU TRAIN6 actual e1 adapter effect; paired inference ablation, no training."""
import argparse
import itertools
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
import train as training
from drone_rf.waveform import waveform_metrics

ROOT=training.ROOT;w=training.w


def batch(raw):
    return {k:torch.as_tensor(np.asarray(raw[k])[None]) for k in
        ('mixture','references','active','context_features','crop_start','construction_count')}


def metric(pred,logits,item):
    m=waveform_metrics(pred,item['references'],item['active'],item['mixture']);n=int(item['construction_count'])
    power=m['reference_power'][0,:n].tolist()
    row=dict(count=n,nmse=m['nmse'][0,:n].tolist(),si_sdr=m['si_sdr'][0,:n].tolist(),
        reference_power=power,weakest_index=int(np.argmin(power)),predicted_count=int(logits.argmax(-1))+1,
        sum_relative_error=float(m['sum_relative_error'][0]))
    assert np.isfinite(row['nmse']+row['si_sdr']).all() and row['sum_relative_error']<1e-10
    return row


def run(root,pred,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    original=w.read(pred/'PROTOCOL.json');training.verify(pred,original)
    prior_path=ROOT/'reports/2026-10-10/SOURCE_INTERACTION_TRAIN_FIT.json';prior=w.read(prior_path)
    candidates=[r for r in prior['rows'] if r['model']=='parent/e0']
    indices=[r['index'] for count in (1,2,3) for r in [q for q in candidates if q['count']==count][:2]]
    hashes=dict(original['source_sha256']);hashes[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    protocol=dict(status='REGISTERED_CPU_TF_AXIS_MECHANISM',indices=indices,source_sha256=hashes,
        predecessor_protocol_sha256=w.digest(pred/'PROTOCOL.json'),prior_sha256=w.digest(prior_path),
        selection='Same first two TRAIN48 cases per count, fixed before e1 result',
        intervention='Actual e1 model versus same weights with adapter bypassed during inference; not a retrained ablation',
        updates=0,heldout_read=False,registered_at=time.time())
    for rel,sha in hashes.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    w.write(root/'PROTOCOL.json',protocol)
    w.write(root/'STATE.json',dict(status='WAITING_FOR_TF_AXIS_E1',pid=os.getpid(),time=time.time()))
    while not (pred/'COMPLETE.json').exists():
        if (pred/'FAILURE.json').exists():raise RuntimeError('Predecessor failed')
        time.sleep(5)
    checkpoint=pred/'ACTUAL_001.pt';checkpoint_sha=w.digest(checkpoint)
    w.write(root/'CHECKPOINT.json',dict(checkpoint_sha256=checkpoint_sha,complete_sha256=w.digest(pred/'COMPLETE.json')))
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    torch.set_num_threads(2);started=time.time()
    net=training.make_model('retained_unet').eval()
    initial={k:v.clone() for k,v in net.tf_axes.state_dict().items()}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==1 and saved['protocol_sha256']==protocol['predecessor_protocol_sha256']
    net.load_state_dict(saved['model']);del saved
    params=[]
    for key,value in net.tf_axes.state_dict().items():
        params.append(dict(parameter=key,initial_norm=float(initial[key].norm()),trained_norm=float(value.norm()),
            change_norm=float((value-initial[key]).norm()),changed_elements=int(torch.count_nonzero(value!=initial[key])),elements=value.numel()))
    del initial
    train=training.base.worker.NativeMixtures(original['preparation'],'train_pack',1)
    rows=[r for r in prior['rows'] if r['model'] in ('parent/e0','retained_unet/e1') and r['index'] in indices]
    internal=[];adapter=net.tf_axes;captured={}
    def hook(module,inputs,output):
        x=inputs[0];delta=output-x
        captured.update(input_shape=list(x.shape),input_rms=float(x.square().mean().sqrt()),
            delta_rms=float(delta.square().mean().sqrt()),delta_relative_energy=float(delta.square().sum()/x.square().sum()))
    handle=adapter.register_forward_hook(hook)
    with torch.inference_mode():
        for case,index in enumerate(indices,1):
            item=batch(train[index]);enabled,logits=training.base.worker.predict(net,item)
            measured=dict(captured)
            net.tf_axes=torch.nn.Identity()
            try:disabled,off_logits=training.base.worker.predict(net,item)
            finally:net.tf_axes=adapter
            assert torch.equal(logits,off_logits)
            distances=[float((enabled[:,:3]-disabled[:,list(p)]).abs().square().sum()) for p in itertools.permutations(range(3))]
            measured.update(index=index,prediction_change_relative_energy=min(distances)/float(item['mixture'].abs().square().sum()),count_logits_bitwise_equal_when_bypassed=True)
            internal.append(measured)
            for name,estimate in [('tf_axis/e1',enabled),('tf_axis/e1_bypassed',disabled)]:
                row=metric(estimate,logits,item);row.update(index=index,model=name);rows.append(row)
            w.write(root/'PARTIAL.json',dict(rows=rows,internal=internal))
            w.write(root/'STATE.json',dict(status='CPU_LAYER_DIAGNOSIS',cases=case,total=6,pid=os.getpid(),time=time.time()))
    handle.remove()
    for rel,sha in hashes.items():assert w.digest(ROOT/rel)==sha
    assert w.digest(checkpoint)==checkpoint_sha
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),checkpoint_sha256=checkpoint_sha,
        rows=rows,internal=internal,parameters=params,seconds=time.time()-started,updates=0,heldout_read=False,
        inference_intervention_only=True,
        limitation='Six TRAIN cases; bypass is not an independently retrained ablation; count head has no direct dependency on the added local adapter')
    w.write(root/'COMPLETE.json',result);w.write(public,result)
    w.write(root/'STATE.json',dict(status='COMPLETE',cases=6,pid=os.getpid(),time=time.time()))
    print(dict(status='COMPLETE',seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('run','predecessor','public'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    try:run(a.run.resolve(),a.predecessor.resolve(),a.public.resolve())
    except Exception:
        w.write(a.run/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
