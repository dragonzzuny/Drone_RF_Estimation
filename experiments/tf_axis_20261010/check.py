"""Capacity, exact initialization and axis-order/gradient checks; no RF data."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
from adapter import augment,AxisBlock,DualAxisAdapter

ROOT=Path(__file__).resolve().parents[2]


def run():
    torch.set_num_threads(2);torch.manual_seed(0)
    path=ROOT/'experiments/count_pcgrad_20261010/train.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('count_training_for_axis_check',path)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    net=base.worker.make_model('retained_unet').eval()
    original={k:v.clone() for k,v in net.state_dict().items()}
    # Small synthetic STFT checks geometry, not a reduced-capacity model.
    z=torch.randn(1,32,48,dtype=torch.complex64)
    context=torch.randn(1,65,256);crop=torch.tensor([0])
    with torch.no_grad():parent=net(z,context,crop)
    torch.manual_seed(0);augment(net)
    assert sum(p.numel() for p in net.parameters())>32142859
    for key,value in original.items():assert torch.equal(value,net.state_dict()[key])
    with torch.no_grad():candidate=net(z,context,crop)
    assert torch.equal(parent['estimates'],candidate['estimates'])
    assert torch.equal(parent['count_logits'],candidate['count_logits'])
    relative=float((candidate['estimates'].sum(1)-z).abs().square().sum()/z.abs().square().sum())
    assert relative<1e-12
    # Distinct batch/frequency/time sizes expose wrong reshape orders. Compare
    # vectorized forward to explicit per-frame and per-frequency loop processing.
    block=AxisBlock().eval();value=torch.randn(2,256,3,5)
    with torch.no_grad():
        vectorized=block(value);reference=value.clone()
        for b in range(2):
            for t in range(5):
                x=value[b,:,:,t].T[None];h,_=block.frequency(block.frequency_norm(x))
                reference[b,:,:,t]=(x+block.frequency_output(h))[0].T
        expected=reference.clone()
        for b in range(2):
            for f in range(3):
                x=reference[b,:,f,:].T[None];h,_=block.time(block.time_norm(x))
                expected[b,:,f,:]=(x+block.time_output(h))[0].T
    error=float((vectorized-expected).abs().max());assert error<2e-6
    adapter=DualAxisAdapter();x=torch.randn(1,1024,3,5);target=torch.randn_like(x)
    opt=torch.optim.AdamW(adapter.parameters(),lr=1e-5)
    for step in range(2):
        opt.zero_grad(set_to_none=True);loss=(adapter(x)-target).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        if step==1:assert float(adapter.blocks[0].frequency.weight_ih_l0.grad.norm())>0
        opt.step()
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path(__file__).with_name('adapter.py'))}
    return dict(status='PASS',full_parameters=sum(p.numel() for p in net.parameters()),
        added_parameters=sum(p.numel() for p in net.tf_axes.parameters()),
        full_parent_parameters_preserved=True,initial_predictions_bitwise_equal=True,
        mixture_sum_relative_error=relative,axis_loop_max_error=error,
        recurrence_gradient_nonzero_after_readout_step=True,source_sha256=hashes,
        synthetic_only=True,recorded_iq_reads=0,gpu_use=False,trained_separation_result=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
