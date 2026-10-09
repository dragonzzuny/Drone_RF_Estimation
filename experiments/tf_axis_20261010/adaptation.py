"""Same full dual-axis model; one controlled choice of backbone trainability."""
import torch

ARMS=('joint_adapter_lr','frozen_backbone')


def configure(net,arm):
    if arm not in ARMS:raise ValueError(arm)
    backbone=[];adapter=[]
    for name,param in net.named_parameters():
        is_adapter=name.startswith('tf_axes.')
        param.requires_grad_(is_adapter or arm=='joint_adapter_lr')
        (adapter if is_adapter else backbone).append(param)
    assert sum(p.numel() for p in backbone)==32142859
    assert sum(p.numel() for p in adapter)==5263616
    groups=[dict(params=adapter,lr=1e-4,name='adapter')]
    if arm=='joint_adapter_lr':groups.insert(0,dict(params=backbone,lr=1e-5,name='backbone'))
    opt=torch.optim.AdamW(groups,weight_decay=1e-4,foreach=False)
    return opt,[p for p in net.parameters() if p.requires_grad]
