"""Full-capacity synthetic initialization, gradient, and cross-chunk checks."""
import json
import torch
from long_temporal_candidate import ROOT
from models import build
from chunk_temporal_candidate import ChunkTemporalUNet
from checkpoint_blocks import enable_checkpointing


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    parent=build('unet_mean').train()
    net=ChunkTemporalUNet().initialize_parent(parent.state_dict()).train()
    enable_checkpointing(net)
    features=torch.randn(1,65,255)
    zs=[torch.randn(1,32,48,dtype=torch.complex64) for _ in range(3)]
    starts=[torch.tensor([40000+i*63872]) for i in range(3)]
    targets=[torch.randn(1,4,32,48,dtype=torch.complex64) for _ in range(3)]
    expected=[];old_loss=0
    for z,start,target in zip(zs,starts,targets):
        result=parent(z,features,start);expected.append({k:v.detach().clone() for k,v in result.items()})
        loss=((result['estimates']-target).abs().square().mean()+result['count_logits'].square().mean())/3
        old_loss+=float(loss.detach());loss.backward()
    old_grads={k:p.grad.clone() for k,p in parent.named_parameters()}
    del parent
    output=net(zs,features,starts)
    for estimate,original in zip(output['estimates'],expected):
        torch.testing.assert_close(estimate,original['estimates'],rtol=0,atol=0)
        torch.testing.assert_close(output['count_logits'],original['count_logits'],rtol=0,atol=0)
    loss=sum((estimate-target).abs().square().mean() for estimate,target in zip(output['estimates'],targets))/3
    loss=loss+output['count_logits'].square().mean();loss.backward()
    max_gradient=0.
    for name,p in net.named_parameters():
        if name in old_grads:
            torch.testing.assert_close(p.grad,old_grads[name],rtol=2e-5,atol=2e-7)
            max_gradient=max(max_gradient,float((p.grad-old_grads[name]).abs().max()))
    assert float(net.gates.expand.weight.grad.norm())>0
    # An initially zero adapter has no cross-chunk effect. A nonzero synthetic
    # projection must create one, showing this path is connected rather than a
    # cosmetically named long-context module. No RF training or score involved.
    net.eval()
    with torch.no_grad():
        before=[v.clone() for v in net(zs,features,starts)['estimates']]
        changed=[zs[0],zs[1],zs[2]*1j]
        zero=net(changed,features,starts)['estimates']
        torch.testing.assert_close(zero[0],before[0],rtol=0,atol=0)
        net.gates.expand.weight.normal_(std=1e-3)
        original=net(zs,features,starts)['estimates']
        altered=net(changed,features,starts)['estimates']
        cross=float((original[0]-altered[0]).abs().max())
        assert cross>1e-7
        for estimate,z in zip(original,zs):
            assert float((estimate.sum(1)-z).abs().square().mean()/z.abs().square().mean())<1e-12
    result=dict(status='PASS',parameters=sum(p.numel() for p in net.parameters()),
        initial_chunk_output_difference=0.,parent_gradient_max_difference=max_gradient,
        synthetic_cross_chunk_effect=cross,backbone_widths=[64,128,256,512,1024],
        synthetic_weights_discarded=True,rf_training_updates=0,gpu_preflight='NOT_RUN',
        note='Full parameter capacity, small synthetic STFTs for algebra; not a downsized training experiment')
    (ROOT/'reports/2026-10-13/CHUNK_TEMPORAL_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
