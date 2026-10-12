"""Full-parameter CPU algebra check on small synthetic STFT tensors, no I/Q data."""
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008'))
from models import build
from checkpoint_blocks import enable_checkpointing,combine_items


def main():
    torch.set_num_threads(2);torch.manual_seed(12013)
    net=build('unet_mean').train()
    count=sum(v.numel() for v in net.parameters());assert count==32142859
    keys=list(net.state_dict())
    z=torch.randn(1,32,48,dtype=torch.complex64)
    features=torch.randn(1,65,255);start=torch.tensor([12345])
    target=torch.randn(1,4,32,48,dtype=torch.complex64)
    def calculate():
        result=net(z,features,start)
        loss=(result['estimates']-target).abs().square().mean()+result['count_logits'].square().mean()
        loss.backward()
        return {k:v.detach().clone() for k,v in result.items()}
    expected=calculate();grads={n:p.grad.clone() for n,p in net.named_parameters()}
    net.zero_grad(set_to_none=True);enable_checkpointing(net)
    assert list(net.state_dict())==keys
    actual=calculate();max_output=0.;max_gradient=0.
    for k in expected:
        torch.testing.assert_close(actual[k],expected[k],rtol=1e-6,atol=1e-7)
        max_output=max(max_output,float((actual[k]-expected[k]).abs().max()))
    for n,p in net.named_parameters():
        torch.testing.assert_close(p.grad,grads[n],rtol=1e-5,atol=1e-7)
        max_gradient=max(max_gradient,float((p.grad-grads[n]).abs().max()))
    # Exact observed-sample and coordinate preservation through concatenation.
    ref=torch.randn(1,3,291,dtype=torch.complex64);mix=ref.sum(1)
    items=[dict(references=ref[...,i*97:(i+1)*97],mixture=mix[...,i*97:(i+1)*97],
        active=torch.ones(1,3,dtype=torch.bool),context_features=features,
        crop_start=start+i*97,construction_count=torch.tensor([3])) for i in range(3)]
    joined=combine_items(items);assert torch.equal(joined['mixture'],mix) and torch.equal(joined['references'],ref)
    try:enable_checkpointing(net)
    except ValueError:pass
    else:raise AssertionError('Double wrapping should fail')
    result=dict(status='PASS',parameters=count,tensor_count=len(grads),max_output_error=max_output,
        max_gradient_error=max_gradient,training_data_used=False,gpu_preflight='NOT_RUN',
        note='Small synthetic tensors verify full model algebra, not training or long-window capacity',time=time.time())
    p=ROOT/'local/long_context_cpu_check_20261013.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
