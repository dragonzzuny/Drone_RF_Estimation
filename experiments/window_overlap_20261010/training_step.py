"""Exact two-view gradient with at most one trainable forward graph retained."""
import torch
from paired import canonical,consistency,regions


def accumulate(net,first,second,offset,predict,loss_fn,weight,divisor=32):
    left,right=regions(first['mixture'].shape[-1],offset)
    for key in ('mixture','references'):
        assert torch.equal(first[key][...,left],second[key][...,right])
    assert torch.equal(first['context_features'],second['context_features'])
    assert torch.equal(first['active'],second['active'])
    # The architecture must be deterministic within an update: no dropout or
    # mutable normalization statistics. Our retained TCN/U-Net meets this.
    with torch.no_grad():
        target_b,_=predict(net,second)
        loss_b=loss_fn(target_b,second['references'],second['active'],second['mixture'])
        canonical_b=canonical(target_b,loss_b['assignment'])[...,right]
    pred_a,logits_a=predict(net,first)
    loss_a=loss_fn(pred_a,first['references'],first['active'],first['mixture'])
    canonical_a=canonical(pred_a,loss_a['assignment'])[...,left]
    saved_a=canonical_a.detach()
    common=(first['references'][...,left],first['active'],first['mixture'][...,left])
    consistency_a=consistency(canonical_a,canonical_b,*common)
    main_a=loss_a['loss']+.1*torch.nn.functional.cross_entropy(logits_a,first['construction_count']-1)
    assert torch.isfinite(main_a+consistency_a)
    ((.5*main_a+weight*consistency_a)/divisor).backward()
    main_a_value=float(main_a.detach());consistency_value=float(consistency_a.detach())
    del pred_a,logits_a,loss_a,canonical_a,main_a,consistency_a
    pred_b,logits_b=predict(net,second)
    repeat_error=float((pred_b.detach()-target_b).abs().max())
    assert torch.equal(pred_b.detach(),target_b),'Forward changed within a shared-weight update'
    loss_b=loss_fn(pred_b,second['references'],second['active'],second['mixture'])
    canonical_b=canonical(pred_b,loss_b['assignment'])[...,right]
    consistency_b=consistency(saved_a,canonical_b,*common)
    main_b=loss_b['loss']+.1*torch.nn.functional.cross_entropy(logits_b,second['construction_count']-1)
    assert torch.isfinite(main_b+consistency_b)
    ((.5*main_b+weight*consistency_b)/divisor).backward()
    return dict(loss=.5*(main_a_value+float(main_b.detach()))+weight*consistency_value,
        main=.5*(main_a_value+float(main_b.detach())),consistency=consistency_value,
        applied_weight=float(weight),second_forward_max_difference=repeat_error)
