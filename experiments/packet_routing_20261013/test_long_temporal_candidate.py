"""Full-capacity synthetic algebra only; no RF training or performance claim."""
import json
from pathlib import Path
import torch
from long_temporal_candidate import LongTemporalUNet,ROOT
from models import build
from checkpoint_blocks import enable_checkpointing


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    parent=build('unet_mean').eval()
    net=LongTemporalUNet().initialize_parent(parent.state_dict()).train()
    z=torch.randn(1,32,96,dtype=torch.complex64);features=torch.randn(1,65,255);start=torch.tensor([40000])
    with torch.no_grad():original=parent(z,features,start)
    del parent
    enable_checkpointing(net)
    predicted=net(z,features,start)
    for key in original:torch.testing.assert_close(predicted[key],original[key],rtol=0,atol=0)
    assert float((predicted['estimates'].sum(1)-z).abs().square().mean()/z.abs().square().mean())<1e-12
    target=torch.randn_like(predicted['estimates'])
    loss=(predicted['estimates']-target).abs().square().mean()+predicted['count_logits'].square().mean()
    loss.backward()
    projection_norm=float(net.gates.expand.weight.grad.norm())
    assert projection_norm>0 and float(net.gates.reduce.weight.grad.norm())==0
    assert float(net.down[0][0].weight.grad.norm())>0
    with torch.no_grad():net.gates.expand.weight.add_(net.gates.expand.weight.grad,alpha=-1e-3)
    net.zero_grad(set_to_none=True);second=net(z,features,start)
    (second['estimates']-target).abs().square().mean().backward()
    encoder_norm=float(net.gates.reduce.weight.grad.norm());assert encoder_norm>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    result=dict(status='PASS',parameters=sum(p.numel() for p in net.parameters()),
        added_parameters=sum(p.numel() for p in net.gates.parameters()),parent_output_max_difference=0.,
        first_projection_gradient_norm=projection_norm,second_adapter_encoder_gradient_norm=encoder_norm,
        checkpointed=True,dilations=[1,2,4,8,16,32],backbone_widths=[64,128,256,512,1024],
        gpu_preflight='NOT_RUN',rf_training_updates=0,synthetic_weights_discarded=True,
        note='Only synthetic identity/gradient checks; no candidate training registered or queued')
    (ROOT/'reports/2026-10-13/LONG_TEMPORAL_CANDIDATE_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
