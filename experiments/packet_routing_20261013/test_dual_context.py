"""Full model algebra and frozen-parent checks; synthetic weights are discarded."""
import gc
from pathlib import Path
import torch
from dual_context_candidate import DualContextUNet,ADDED_PREFIXES
from long_temporal_candidate import ROOT
from models import build
from checkpoint_blocks import enable_checkpointing
from sequence_data import write_json,sha256


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    parent=build('unet_mean').eval();original=parent.state_dict()
    net=DualContextUNet().initialize_parent(original).eval();enable_checkpointing(net)
    spectra=[torch.randn(1,32,48,dtype=torch.complex64) for _ in range(3)]
    starts=[torch.tensor([40000+i*63872]) for i in range(3)]
    features=torch.randn(1,65,255);targets=[torch.randn(1,4,32,48,dtype=torch.complex64) for _ in range(3)]
    with torch.no_grad():
        out=net(spectra,features,starts)
        for z,start,y in zip(spectra,starts,out['estimates']):
            old=parent(z,features,start)
            torch.testing.assert_close(y,old['estimates'],rtol=0,atol=0)
            torch.testing.assert_close(out['count_logits'],old['count_logits'],rtol=0,atol=0)
        net.ordered_projection.weight.normal_(std=1e-3)
        constant=features[...,:1].expand_as(features)
        assert int(net.variation(constant).count_nonzero())==0
        reversal=float((net.variation(features)-net.variation(features.flip(-1))).abs().max())
        assert reversal>1e-7
        net.ordered_projection.weight.zero_()
    del parent,out,old;gc.collect()
    net.train()
    for name,p in net.named_parameters():p.requires_grad_(name.startswith(ADDED_PREFIXES))
    assert sum(p.numel() for p in net.parameters())==36554507
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==4411648
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4,foreach=False)
    rows=[]
    for step in (1,2):
        opt.zero_grad(set_to_none=True);out=net(spectra,features,starts)
        loss=sum((x-y).abs().square().mean() for x,y in zip(out['estimates'],targets))/3;loss.backward()
        assert all(p.grad is None for n,p in net.named_parameters() if not n.startswith(ADDED_PREFIXES))
        norms={prefix:sum(float(p.grad.double().square().sum()) for n,p in net.named_parameters()
                         if n.startswith(prefix) and p.grad is not None)**.5 for prefix in ADDED_PREFIXES}
        assert norms['gates.']>0 and norms['ordered_projection.']>0
        if step==2:assert norms['ordered_encoder.']>0 and float(net.gates.reduce.weight.grad.norm())>0
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        rows.append(dict(step=step,gradient_norms=norms))
        for estimate,z in zip(out['estimates'],spectra):
            assert float((estimate.detach().sum(1)-z).abs().square().mean()/z.abs().square().mean())<1e-12
    for name,value in original.items():assert torch.equal(net.state_dict()[name],value),name
    assert all(int(v['step'])==2 for v in opt.state.values())
    result=dict(status='PASS',total_parameters=36554507,trainable_added_parameters=4411648,
        preserved_parent_tensors=len(original),initial_parent_output_difference=0,
        constant_context_addition_zero=True,nonzero_projection_reversal_effect=reversal,
        gradient_checks=rows,rf_training_updates=0,synthetic_weights_discarded=True,gpu_preflight='NOT_RUN',
        source_sha256={Path(p).name:sha256(p) for p in (__file__,Path(__file__).with_name('dual_context_candidate.py'))})
    write_json(ROOT/'reports/2026-10-13/DUAL_CONTEXT_CPU_CHECK.json',result);print(result,flush=True)


if __name__=='__main__':main()
