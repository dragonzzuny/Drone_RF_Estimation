"""Real TRAIN context, full parent weights; branch checks without CNN inference."""
from pathlib import Path
import json
import torch
import architecture as model

ROOT,w,worker=model.ROOT,model.w,model.worker


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    old=w.read(ROOT/'local/count_pcgrad_20261010_v1/PROTOCOL.json')
    checkpoint=Path(old['parent_checkpoint'])
    assert w.digest(checkpoint)==old['parent_checkpoint_sha256']
    data=worker.NativeMixtures(old['preparation'],'train_pack',3)
    index=next(i for i,r in enumerate(data.rows) if int(r['count'])==3)
    feature=torch.from_numpy(data.features[index].copy())[None]
    position=torch.tensor([int(data.rows[index]['crop_start'])])
    net=model.build(checkpoint).eval()
    bias,logits,base,correction=net.context(feature,position,500)
    assert torch.equal(bias,base)
    assert bias.shape==(1,1024,31) and logits.shape==(1,3)
    # Constant contexts must produce exactly zero added path at ANY gate.
    with torch.no_grad():
        net.gate.fill_(.1)
        constant=feature.mean(-1,keepdim=True).expand_as(feature)
        b,_,a,c=net.context(constant,position,500)
        assert torch.count_nonzero(c)==0 and torch.equal(a,b)
    first=net.context(feature,position,500)
    reverse=net.context(feature.flip(-1),position,500)
    reverse_delta=float((first[0]-reverse[0]).abs().max().detach())
    assert reverse_delta>1e-7
    first[0].square().mean().backward()
    gradients={}
    for prefix in ('gate','ordered_encoder','ordered_projection'):
        selected=[p for n,p in net.named_parameters() if n==prefix or n.startswith(prefix+'.')]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected)
        gradients[prefix]=sum(float(p.grad.double().square().sum()) for p in selected)**.5
        assert gradients[prefix]>0
    assert all(p.grad is None for p in net.parent.parameters())
    frozen=model.verify_parent(net,checkpoint)
    result=dict(status='PASS',parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        frozen_parent_parameters=sum(p.numel() for p in net.parent.parameters()),
        frozen_parent_tensors=frozen,train_schedule_epoch=3,train_context_index=index,
        zero_gate_preserves_context_exactly=True,constant_context_zero_added_path=True,
        reversed_context_max_bias_difference=reverse_delta,nonzero_gate_gradient_norms=gradients,
        full_waveform_gpu_preflight_pending=True,full_waveform_cpu_inference=False,optimizer_updates=0,
        validation_read=False,heldout_read=False,parent_checkpoint_sha256=old['parent_checkpoint_sha256'],
        source_sha256={str(p.relative_to(ROOT)):w.digest(p) for p in (Path(__file__),Path(model.__file__))})
    w.write(ROOT/'reports/2026-10-11/ORDERED_BRANCH_CPU_CHECK.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
