"""Full-configuration CPU checks on native TRAIN; no optimizer updates."""
import hashlib
import importlib.util
import os
from pathlib import Path
import time
import torch
from architecture import augment,segment,overlap_add,RelativeAttention
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('septda_check_parent',ROOT/'experiments/fresh_schedule_20261010/train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def digest_tensor(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    os.sched_setaffinity(0,{12,13});os.nice(15);torch.set_num_threads(2);torch.manual_seed(0)
    root=ROOT/'local/septda_rf_check_20261010_v1';root.mkdir(exist_ok=True)
    assert not (root/'RESULT.json').exists();started=time.time()
    def state(label,**kwargs):t.w.write(root/'STATE.json',dict(status=label,pid=os.getpid(),time=time.time(),**kwargs))
    state('GEOMETRY_CHECK')
    for length in (1,48,96,97,500):
        x=torch.randn(2,length,128,requires_grad=True)
        parts,g=segment(x);y=overlap_add(parts,g)
        torch.testing.assert_close(y,x,rtol=0,atol=0);y.sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=0,atol=0)
    rel=RelativeAttention();v=rel.bucket(torch.arange(-400,401))
    assert v.min()>=0 and v.max()<32 and v[400]==0 and v[399]!=v[401]
    p=t.w.read(ROOT/'local/fresh_schedule_20261010_v1/PROTOCOL.json');old=p['original_protocol']
    data=t.worker.NativeMixtures(old['preparation'],'train_pack',3)
    indices=[next(i for i,r in enumerate(data.rows) if int(r['count'])==n) for n in (1,2,3)]
    torch.manual_seed(0);net=t.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    parents={k:digest_tensor(v) for k,v in net.state_dict().items()}
    expected=[]
    for index in indices:
        state('PARENT_FORWARD',index=index)
        with torch.no_grad():o,l=t.worker.predict(net,t.core.batch(data[index]))
        expected.append((o.clone(),l.clone()))
    net=augment(net);initial={k:digest_tensor(v) for k,v in net.state_dict().items()};rows=[]
    for i,index in enumerate(indices):
        state('AUGMENTED_FORWARD',index=index)
        item=t.core.batch(data[index])
        with torch.no_grad():o,l=t.worker.predict(net,item)
        assert torch.equal(o,expected[i][0]) and torch.equal(l,expected[i][1])
        error=float((o.sum(1)-item['mixture']).abs().square().mean()/item['mixture'].abs().square().mean())
        assert error<1e-9
        rows.append(dict(index=index,count=i+1,initial_output_bitwise_equal=True,sum_relative_error=error))
    # Genuine, full 63,872-sample TRAIN triple, with the complete 8-block branch.
    item=t.core.batch(data[indices[-1]]);net.train();net.zero_grad(set_to_none=True)
    state('ZERO_READOUT_BACKWARD')
    o,l=t.worker.predict(net,item)
    loss=t.worker.pit_waveform_loss(o,item['references'],item['active'],item['mixture'])['loss']
    loss=loss+.1*torch.nn.functional.cross_entropy(l,item['construction_count']-1);loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in net.parameters())
    readout_norm=float(net.septda.source_readout.weight.grad.norm());assert readout_norm>0
    zero_query_norm=float(net.septda.queries.grad.norm());assert zero_query_norm==0
    del o,l,loss;net.zero_grad(set_to_none=True)
    state('ENABLED_READOUT_BACKWARD')
    with torch.no_grad():
        torch.manual_seed(17);torch.nn.init.normal_(net.septda.source_readout.weight,std=1e-4)
    o,l=t.worker.predict(net,item)
    loss=t.worker.pit_waveform_loss(o,item['references'],item['active'],item['mixture'])['loss']
    loss=loss+.1*torch.nn.functional.cross_entropy(l,item['construction_count']-1);loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in net.parameters())
    norms={name:float(q.grad.norm()) for name,q in net.septda.named_parameters() if name in (
        'queries','encoder.weight','attractors.0.cross_attention.in_proj_weight',
        'triple.0.temporal.intra.recurrent.weight_ih_l0','triple.7.temporal.inter.recurrent.weight_ih_l0',
        'triple.7.source.self_attn.in_proj_weight')}
    assert len(norms)==6 and all(v>0 for v in norms.values())
    with torch.no_grad():net.septda.source_readout.weight.zero_()
    assert all(digest_tensor(v)==initial[k] for k,v in net.state_dict().items())
    assert all(digest_tensor(net.state_dict()[k])==sha for k,sha in parents.items())
    result=dict(status='PASS',parameters=sum(q.numel() for q in net.parameters()),
        branch_parameters=sum(q.numel() for q in net.septda.parameters()),parent_parameters=32142859,
        full_configuration=dict(width=128,encoder_width=256,lstm_hidden_per_direction=256,attention_heads=4,
            attractor_layers=2,triple_blocks=8,chunk_frames=96,hop_frames=48,complex_samples=63872),
        rows=rows,zero_readout_gradient_norm=readout_norm,initial_query_gradient_norm=zero_query_norm,
        enabled_internal_gradient_norms=norms,initial_state_restored=True,optimizer_updates=0,gpu_use=False,heldout_read=False,
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        source_sha256={str(q.relative_to(ROOT)):t.w.digest(q) for q in (Path(__file__),Path(__file__).with_name('architecture.py'))},
        seconds=time.time()-started,limitation='Full configuration numerical/gradient checks on three TRAIN mixtures, not performance evidence.')
    t.w.write(root/'RESULT.json',result);t.w.write(ROOT/'reports/2026-10-10/SEPTDA_RF_CPU_CHECK.json',result)
    state('COMPLETE');print(result,flush=True)


if __name__=='__main__':main()
