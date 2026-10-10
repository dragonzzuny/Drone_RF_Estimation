"""Full-capacity CPU comparison of direct and checkpointed backward, no steps."""
import importlib.util
import os
from pathlib import Path
import time
import torch
from torch.utils.checkpoint import checkpoint

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresh_train_check',HERE/'train.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def main():
    os.sched_setaffinity(0,{12,13});os.nice(15);torch.set_num_threads(2)
    root=t.ROOT/'local/fresh_checkpoint_check_20261010_v1';root.mkdir(exist_ok=True)
    assert not (root/'RESULT.json').exists()
    active=t.ROOT/'local/fresh_schedule_20261010_v1';p=t.w.read(active/'PROTOCOL.json');t.verify(active,p)
    old=p['original_protocol'];start=time.time()
    data=t.worker.NativeMixtures(old['preparation'],'train_pack',3)
    indices=[next(i for i,r in enumerate(data.rows) if int(r['count'])==n) for n in (2,3)]
    net=t.worker.make_model('retained_unet',Path(old['parent_checkpoint'])).train()
    initial={k:v.clone() for k,v in net.state_dict().items()}
    rows=[]
    for index in indices:
        item=t.core.batch(data[index]);net.zero_grad(set_to_none=True)
        def forward(mix,context,position):
            return t.worker.predict(net,dict(mixture=mix,context_features=context,crop_start=position))
        def objective(output,logits):
            return t.worker.pit_waveform_loss(output,item['references'],item['active'],item['mixture'])['loss']+.1*torch.nn.functional.cross_entropy(logits,item['construction_count']-1)
        direct,logits=forward(item['mixture'],item['context_features'],item['crop_start'])
        loss=objective(direct,logits);loss.backward()
        outputs=direct.detach().clone();logits_direct=logits.detach().clone();loss_direct=float(loss.detach())
        grads={k:v.grad.clone() for k,v in net.named_parameters()}
        del direct,logits,loss
        net.zero_grad(set_to_none=True)
        checked,logits=checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)
        loss=objective(checked,logits);loss.backward()
        assert torch.equal(checked,outputs) and torch.equal(logits,logits_direct)
        square_error=square_ref=0.;maximum=0.;exact=True
        for key,value in net.named_parameters():
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert torch.allclose(value.grad,grads[key],rtol=1e-6,atol=1e-8),key
            delta=(value.grad-grads[key]).double();square_error+=float(delta.square().sum())
            square_ref+=float(grads[key].double().square().sum());maximum=max(maximum,float(delta.abs().max()))
            exact=exact and torch.equal(value.grad,grads[key])
        relative=(square_error/max(square_ref,1e-30))**.5
        assert relative<1e-6 and float(loss)==loss_direct
        rows.append(dict(index=index,count=int(item['construction_count']),loss=loss_direct,
            outputs_bitwise_equal=True,gradients_bitwise_equal=exact,gradient_relative_l2=relative,max_abs_gradient_error=maximum))
        t.w.write(root/'STATE.json',dict(status='CPU_CHECKPOINT_CHECK',completed=len(rows),total=2,pid=os.getpid(),time=time.time()))
        del checked,logits,loss,grads,outputs,logits_direct,item
    assert all(torch.equal(value,initial[k]) for k,value in net.state_dict().items())
    t.verify(active,p)
    result=dict(status='PASS',parameters=sum(q.numel() for q in net.parameters()),rows=rows,
        optimizer_steps=0,weights_unchanged=True,gpu_use=False,heldout_read=False,
        training_protocol_sha256=t.w.digest(active/'PROTOCOL.json'),checker_sha256=t.w.digest(Path(__file__)),
        seconds=time.time()-start,limitation='Two complete TRAIN examples on CPU; not all75 GPU updates or equal-FLOP guarantee.')
    t.w.write(root/'RESULT.json',result);t.w.write(t.ROOT/'reports/2026-10-10/FRESH_CHECKPOINT_CHECK.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
