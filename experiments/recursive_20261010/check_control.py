"""Verify duplicated-control gradients equal one original supervised example."""
import hashlib
import json
from pathlib import Path
import torch
import train as training

def run():
    torch.manual_seed(0)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(4,dtype=torch.float64))
            self.logits=torch.nn.Parameter(torch.randn(1,3,dtype=torch.float64))
    def predictor(net,item):
        raw=net.weight[None,:,None]*item['mixture'][:,None]
        return raw+(item['mixture']-raw.sum(1))[:,None]/4,net.logits
    refs=torch.randn(1,3,47,dtype=torch.complex128)
    item=dict(mixture=refs.sum(1),references=refs,active=torch.ones(1,3,dtype=torch.bool),
        context_features=torch.zeros(1,65,255),crop_start=torch.zeros(1,dtype=torch.long),construction_count=torch.tensor([3]))
    net=Toy();original_predict=training.worker.predict
    try:
        training.worker.predict=predictor
        value=training.accumulate(net,item,training.ARMS[0]);duplicated=[p.grad.clone() for p in net.parameters()]
        net.zero_grad(set_to_none=True)
        output,logits=predictor(net,item);loss=training.main_loss(output,logits,item);(loss/32).backward()
        errors=[float((g-p.grad).abs().max()) for g,p in zip(duplicated,net.parameters())]
        assert max(errors)<1e-12 and abs(value-float(loss.detach()))<1e-12
    finally:training.worker.predict=original_predict
    paths=[Path(__file__),Path(__file__).with_name('train.py'),Path(__file__).with_name('successive.py'),Path(__file__).with_name('validation.py')]
    result=dict(status='PASS',duplicate_vs_single_gradient_max_error=max(errors),
        no_new_examples_in_control=True,optimizer_steps=0,synthetic_only=True,
        sources={str(p.relative_to(training.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    path=training.ROOT/'reports/2026-10-10/SUCCESSIVE_CONTROL_CHECK.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':run()
