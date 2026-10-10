"""Nested C4/C8 prediction averaging, with prediction-only slot matching."""
import importlib.util
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
FOUR_SOURCE=ROOT/'experiments/rfuav_phase_average_20261009/phase_core.py'
spec=importlib.util.spec_from_file_location('phase_eight_existing_four',FOUR_SOURCE)
four=importlib.util.module_from_spec(spec);spec.loader.exec_module(four)


@torch.no_grad()
def predict_eight(model,predict,item):
    anchor,mean4,logits,diagnostic=four.phase_predictions(model,predict,item)
    inputs={k:item[k] for k in ('mixture','context_features','crop_start','long_mixture','long_start') if k in item}
    extra=[];orders=[]
    for degrees in (45,135,225,315):
        theta=math.radians(degrees);factor=complex(math.cos(theta),math.sin(theta))
        view=dict(inputs,mixture=inputs['mixture']*factor)
        if 'long_mixture' in inputs:view['long_mixture']=inputs['long_mixture']*factor
        pred,count=predict(model,view)
        assert torch.equal(count,logits)
        aligned,order=four.align_predictions(pred/factor,anchor)
        extra.append(aligned);orders.append(order)
    mean8=.5*(mean4+torch.stack(extra).mean(0))
    return mean4,mean8,logits,dict(four_phase=diagnostic,additional_prediction_only_orders=orders,
        forward_passes=8,additional_angles=[45,135,225,315])
