"""An optional frozen-backbone adaptation mode, tested without RF training."""
import gc
from pathlib import Path
import torch
from chunk_temporal_candidate import ChunkTemporalUNet
from long_temporal_candidate import ROOT
from models import build
from checkpoint_blocks import enable_checkpointing
from sequence_data import write_json, sha256


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    parent=build('unet_mean');original=parent.state_dict()
    net=ChunkTemporalUNet().initialize_parent(original).train();del parent;gc.collect()
    for name,p in net.named_parameters():p.requires_grad_(name.startswith('gates.'))
    enable_checkpointing(net)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==3677440
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4,foreach=False)
    spectra=[torch.randn(1,32,48,dtype=torch.complex64) for _ in range(3)]
    starts=[torch.tensor([40000+i*63872]) for i in range(3)]
    features=torch.randn(1,65,255);targets=[torch.randn(1,4,32,48,dtype=torch.complex64) for _ in range(3)]
    rows=[]
    for step in (1,2):
        opt.zero_grad(set_to_none=True);out=net(spectra,features,starts)
        loss=sum((x-y).abs().square().mean() for x,y in zip(out['estimates'],targets))/3
        loss.backward()
        assert all(p.grad is None for n,p in net.named_parameters() if not n.startswith('gates.'))
        reduce=float(net.gates.reduce.weight.grad.norm());expand=float(net.gates.expand.weight.grad.norm())
        assert expand>0
        if step==2:assert reduce>0
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));opt.step()
        rows.append(dict(step=step,reduce_gradient=reduce,expand_gradient=expand,gradient_norm=norm))
    for name,value in original.items():assert torch.equal(net.state_dict()[name],value),name
    assert all(int(v['step'])==2 for v in opt.state.values())
    receipt=dict(status='PASS',synthetic_steps=2,rf_training_updates=0,weights_discarded=True,
        total_parameters=35820299,trainable_parameters=3677440,frozen_tensors=len(original),rows=rows,
        checks=['Every parent tensor exactly unchanged','Only adapter optimizer states','Adapter input gradient nonzero after first step'],
        source_sha256={Path(p).name:sha256(p) for p in (__file__,Path(__file__).with_name('chunk_temporal_candidate.py'))})
    write_json(ROOT/'reports/2026-10-13/CHUNK_FROZEN_CPU_CHECK.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
