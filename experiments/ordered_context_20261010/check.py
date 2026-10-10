"""Three full-size CPU checks; no optimizer or held-out records."""
import importlib.util
import os
from pathlib import Path
import time
import torch
from model import augment
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fresh_context_check',ROOT/'experiments/fresh_schedule_20261010/train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def main():
    os.sched_setaffinity(0,{12,13});os.nice(15);torch.set_num_threads(2)
    root=ROOT/'local/ordered_context_check_20261010_v1';root.mkdir(exist_ok=True)
    assert not (root/'RESULT.json').exists()
    fp=ROOT/'local/fresh_schedule_20261010_v1';p=t.w.read(fp/'PROTOCOL.json');t.verify(fp,p)
    old=p['original_protocol'];data=t.worker.NativeMixtures(old['preparation'],'train_pack',3)
    indices=[next(i for i,r in enumerate(data.rows) if int(r['count'])==n) for n in (1,2,3)]
    net=t.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).eval()
    original={k:v.clone() for k,v in net.state_dict().items()}
    rows=[];started=time.time()
    for index in indices:
        item=t.core.batch(data[index])
        if not hasattr(net.context_encoder,'order_gate'):
            with torch.no_grad(): initial,logits=t.worker.predict(net,item)
            net=augment(net)
        else:
            wrapper=net.context_encoder
            net.context_encoder=wrapper.base;net.context_mode='mean'
            with torch.no_grad(): initial,logits=t.worker.predict(net,item)
            net.context_encoder=wrapper;net.context_mode='ordered'
        net.zero_grad(set_to_none=True)
        out,count=t.worker.predict(net,item)
        assert torch.equal(out,initial) and torch.equal(count,logits)
        loss=t.worker.pit_waveform_loss(out,item['references'],item['active'],item['mixture'])['loss']
        loss=loss+.1*torch.nn.functional.cross_entropy(count,item['construction_count']-1)
        loss.backward()
        gate=net.context_encoder.order_gate
        grad=gate.grad.detach();assert torch.isfinite(grad).all() and float(grad.norm())>0
        with torch.no_grad():
            features=item['context_features'];reverse=features.flip(-1)
            # Reverse after recomputing the mean: roundoff in reduction is allowed.
            zero_a=net.context_encoder(features);zero_b=net.context_encoder(reverse)
            mean_gap=float((zero_a-zero_b).abs().max())
            assert torch.allclose(zero_a,zero_b,rtol=2e-5,atol=2e-5)
            gate.fill_(.25)
            encoded_a=net.context_encoder(features);encoded_b=net.context_encoder(reverse)
            gap=float((encoded_a-encoded_b).square().mean().sqrt());assert gap>1e-4
            enabled,_=t.worker.predict(net,item)
            permuted=dict(item,context_features=reverse)
            changed,_=t.worker.predict(net,permuted)
            waveform_gap=float((enabled-changed).abs().square().mean()/item['mixture'].abs().square().mean())
            assert waveform_gap>1e-12
            sum_error=float((enabled.sum(1)-item['mixture']).abs().square().mean()/item['mixture'].abs().square().mean())
            assert sum_error<1e-9
            gate.zero_()
        rows.append(dict(index=index,count=int(item['construction_count']),initial_output_bitwise_equal=True,
            gate_gradient_norm=float(grad.norm()),mean_reverse_max_difference=mean_gap,
            enabled_context_reverse_rms=gap,enabled_waveform_reverse_relative_difference=waveform_gap,
            mixture_sum_relative_error=sum_error))
        t.w.write(root/'STATE.json',dict(status='CHECKING',completed=len(rows),total=3,pid=os.getpid(),time=time.time()))
        del out,count,loss,initial,logits,item
    for key,value in original.items():
        target=key.replace('context_encoder.','context_encoder.base.',1) if key.startswith('context_encoder.') else key
        assert torch.equal(value,net.state_dict()[target])
    assert torch.count_nonzero(net.context_encoder.order_gate)==0
    result=dict(status='PASS',parameters=sum(q.numel() for q in net.parameters()),new_parameters=65,
        rows=rows,optimizer_updates=0,parent_weights_unchanged=True,gate_reset_zero=True,gpu_use=False,heldout_read=False,
        parent_sha256=old['parent_checkpoint_sha256'],preparation_sha256=old['preparation_sha256'],
        source_sha256={str(q.relative_to(ROOT)):t.w.digest(q) for q in (Path(__file__),Path(__file__).with_name('model.py'))},
        seconds=time.time()-started,limitation='TRAIN 3 information-flow checks, not performance or generalization evidence.')
    t.w.write(root/'RESULT.json',result);t.w.write(root/'STATE.json',dict(status='COMPLETE',completed=3,total=3,pid=os.getpid(),time=time.time()))
    t.w.write(ROOT/'reports/2026-10-10/ORDERED_CONTEXT_CHECK.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
