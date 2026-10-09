"""Full-capacity synthetic check: frozen tensors, nonzero adapter learning."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import train as training
from adaptation import configure,ARMS

ROOT=training.ROOT


def run():
    torch.set_num_threads(2);rows=[]
    for arm in ARMS:
        torch.manual_seed(0);net=training.make_model('retained_unet').train()
        original={k:v.clone() for k,v in net.state_dict().items()}
        opt,params=configure(net,arm)
        z=torch.randn(1,32,48,dtype=torch.complex64);features=torch.randn(1,65,255);crop=torch.tensor([0])
        target=torch.randn(1,4,32,48,dtype=torch.complex64)
        with torch.no_grad():initial=net(z,features,crop)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);value=net(z,features,crop)
            loss=(value['estimates']-target).abs().square().mean()+.1*torch.nn.functional.cross_entropy(value['count_logits'],torch.tensor([2]))
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        state=net.state_dict()
        changed_base=sum(int(torch.count_nonzero(state[k]!=v)) for k,v in original.items() if not k.startswith('tf_axes.'))
        changed_adapter=sum(int(torch.count_nonzero(state[k]!=v)) for k,v in original.items() if k.startswith('tf_axes.'))
        assert changed_adapter>0 and (changed_base==0 if arm=='frozen_backbone' else changed_base>0)
        assert float(net.tf_axes.blocks[0].frequency.weight_ih_l0.grad.norm())>0
        if arm=='frozen_backbone':
            assert all(p.grad is None for n,p in net.named_parameters() if not n.startswith('tf_axes.'))
            with torch.no_grad():assert torch.equal(net(z,features,crop)['count_logits'],initial['count_logits'])
        assert {int(s['step']) for s in opt.state.values()}=={2}
        rows.append(dict(arm=arm,full_parameters=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in params),changed_backbone_elements=changed_base,
            changed_adapter_elements=changed_adapter,optimizer_learning_rates=[g['lr'] for g in opt.param_groups],
            recurrent_gradient_nonzero=True,frozen_count_logits_equal=arm=='frozen_backbone'))
        del net,opt,params,original,state,initial,value,loss
    files=[Path(__file__),Path(__file__).with_name('adaptation.py')]
    return dict(status='PASS',rows=rows,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        synthetic_only=True,gpu_use=False,recorded_iq_reads=0,trained_separation_result=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
