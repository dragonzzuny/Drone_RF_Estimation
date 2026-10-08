"""Reference-free four-phase inference; no model updates or gain fitting."""
import itertools
import torch

FACTORS=(1+0j,1j,-1+0j,-1j)


def align_predictions(predicted,anchor):
    if predicted.shape!=anchor.shape or predicted.shape[:2]!=(1,4):
        raise ValueError('Expected one window with three source slots and background')
    permutations=list(itertools.permutations(range(3)))
    costs=torch.stack([(predicted[:,p]-anchor[:,:3]).abs().square().mean() for p in permutations])
    order=permutations[int(costs.argmin())]
    return torch.cat((predicted[:,order],predicted[:,3:]),1),list(order)


@torch.no_grad()
def phase_predictions(model,predict,item):
    # Only these mixture-derived fields enter prediction. Scoring is separate.
    inputs={k:item[k] for k in ('mixture','context_features','crop_start','long_mixture','long_start') if k in item}
    full,logits=predict(model,inputs)
    views=[full];orders=[[0,1,2]];changes=[0.]
    for factor in FACTORS[1:]:
        transformed=dict(inputs,mixture=inputs['mixture']*factor)
        if 'long_mixture' in inputs:
            transformed['long_mixture']=inputs['long_mixture']*factor
        estimate,count=predict(model,transformed)
        if not torch.equal(count,logits):
            raise RuntimeError('Power-only count output changed under pure global phase')
        estimate,order=align_predictions(estimate/factor,full)
        changes.append(float((estimate-full).abs().square().sum()/full.abs().square().sum().clamp_min(1e-20)))
        views.append(estimate);orders.append(order)
    return full,torch.stack(views).mean(0),logits,dict(prediction_only_orders=orders,
        inverse_rotated_prediction_relative_changes=changes,forward_passes=4)
