"""TRAIN6: put learned e1 adapter on the unchanged parent, CPU only.

An inference intervention diagnoses backbone drift versus adapter influence.
It is not proof that frozen-backbone adapter training improves unseen data.
"""
import argparse
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
import diagnose_layer as layer

w=layer.w;ROOT=layer.ROOT


def run(root,predecessor,public):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():raise ValueError('Already registered')
    plan=w.read(predecessor/'PROTOCOL.json')
    axis=ROOT/'local/tf_axis_20261010_v1';axis_plan=w.read(axis/'PROTOCOL.json')
    sources=dict(plan['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=w.digest(Path(__file__))
    p=dict(status='REGISTERED_CPU_ADAPTER_TRANSFER',indices=plan['indices'],source_sha256=sources,
        predecessor_protocol_sha256=w.digest(predecessor/'PROTOCOL.json'),axis_protocol_sha256=w.digest(axis/'PROTOCOL.json'),
        intervention='Original parent backbone and context/count weights + trained e1 adapter only',
        updates=0,heldout_read=False,registered_at=time.time())
    for rel,sha in sources.items():
        assert w.digest(ROOT/rel)==sha
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    w.write(root/'PROTOCOL.json',p)
    w.write(root/'STATE.json',dict(status='WAITING_FOR_LAYER_DIAGNOSIS',pid=os.getpid(),time=time.time()))
    while not (predecessor/'COMPLETE.json').exists():
        if (predecessor/'FAILURE.json').exists():raise RuntimeError('Layer diagnosis failed')
        time.sleep(5)
    before=w.read(predecessor/'COMPLETE.json');assert before['protocol_sha256']==p['predecessor_protocol_sha256']
    assert w.digest(axis/'ACTUAL_001.pt')==before['checkpoint_sha256']
    torch.set_num_threads(2);started=time.time()
    net=layer.training.make_model('retained_unet',Path(axis_plan['parent_checkpoint'])).eval()
    parent={k:v.clone() for k,v in net.state_dict().items() if not k.startswith('tf_axes.')}
    checkpoint=torch.load(axis/'ACTUAL_001.pt',map_location='cpu',weights_only=False)
    trained=checkpoint['model'];adapter={k[len('tf_axes.'):]:v for k,v in trained.items() if k.startswith('tf_axes.')}
    net.tf_axes.load_state_dict(adapter)
    for key,value in parent.items():assert torch.equal(net.state_dict()[key],value)
    for key,value in adapter.items():assert torch.equal(net.tf_axes.state_dict()[key],value)
    change={}
    for prefix in ('down.','up.','output.','context_encoder.','context_projection.','count_head.'):
        names=[k for k in parent if k.startswith(prefix)];assert names
        original_energy=sum(float(parent[k].double().square().sum()) for k in names)
        delta_energy=sum(float((trained[k].double()-parent[k].double()).square().sum()) for k in names)
        change[prefix]=dict(parent_norm=original_energy**.5,delta_norm=delta_energy**.5,
            relative_change=(delta_energy/original_energy)**.5)
    del checkpoint,trained,parent,adapter
    train=layer.training.base.worker.NativeMixtures(axis_plan['preparation'],'train_pack',1)
    rows=list(before['rows'])
    with torch.inference_mode():
        for case,index in enumerate(p['indices'],1):
            item=layer.batch(train[index]);estimates,logits=layer.training.base.worker.predict(net,item)
            row=layer.metric(estimates,logits,item);row.update(index=index,model='parent_with_e1_adapter');rows.append(row)
            parent_row=next(r for r in rows if r['model']=='parent/e0' and r['index']==index)
            assert row['predicted_count']==parent_row['predicted_count']
            w.write(root/'PARTIAL.json',dict(rows=rows))
            w.write(root/'STATE.json',dict(status='CPU_TRANSFER_DIAGNOSIS',cases=case,total=6,pid=os.getpid(),time=time.time()))
    for rel,sha in sources.items():assert w.digest(ROOT/rel)==w.digest(root/'source_snapshot'/rel)==sha
    assert w.digest(axis/'ACTUAL_001.pt')==before['checkpoint_sha256']
    result=dict(status='COMPLETE',protocol_sha256=w.digest(root/'PROTOCOL.json'),
        checkpoint_sha256=before['checkpoint_sha256'],predecessor_complete_sha256=w.digest(predecessor/'COMPLETE.json'),
        rows=rows,backbone_parameter_change=change,parent_weights_preserved=True,adapter_weights_exact_e1=True,
        seconds=time.time()-started,updates=0,heldout_read=False,inference_intervention_only=True,
        limitation='Same six TRAIN examples; composite weights were not trained jointly; no frozen-backbone training or DEV superiority claim')
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
