"""Protect two/three-source waveform objectives in actual AdamW update space.

This is a GEM-inspired two-halfspace correction, not GEM reproduction. It
retains ordinary AdamW moments and changes the proposed parameter displacement.
The constraints use waveform loss only, explicitly subtracting count-CE grads.
"""
import torch
from gradient import CountAccumulator
from update_projection import solve


class WaveAccumulator(CountAccumulator):
    def __init__(self,named_parameters):
        named=list(named_parameters)
        super().__init__([p for _,p in named])
        self.ce_parameters=[];self.ce_indices=[]
        for i,(name,param) in enumerate(named):
            if name.startswith(('context_encoder.','count_head.')):
                self.ce_indices.append(i);self.ce_parameters.append(param)
        self.ce_buffers=[[torch.zeros_like(p) for p in self.ce_parameters] for _ in range(2)]

    @torch.no_grad()
    def collect_ce(self,count,values):
        assert count in (2,3) and len(values)==len(self.ce_parameters)
        for buffer,value in zip(self.ce_buffers[count-2],values):
            if value is not None:buffer.add_(value)

    @torch.no_grad()
    def waveform_groups(self):
        result=self.groups[1:].clone()
        for row,buffers in zip(result,self.ce_buffers):
            views=row.split(self.sizes)
            for i,value in zip(self.ce_indices,buffers):views[i].sub_(value.flatten())
        return result

    @torch.no_grad()
    def reset(self):
        super().reset()
        for group in self.ce_buffers:
            for value in group:value.zero_()


@torch.no_grad()
def correct(parameters,before,protected):
    parameters=list(parameters)
    proposed=torch.cat([p.detach().flatten() for p in parameters])-before
    gram=torch.stack([torch.stack([torch.dot(a,b) for b in protected]) for a in protected])
    dots=torch.stack([torch.dot(g,proposed) for g in protected])
    solution=solve(gram.cpu().double().numpy(),dots.cpu().double().numpy())
    delta=proposed.clone()
    for alpha,g in zip(solution['coefficients'],protected):delta.add_(g,alpha=-alpha)
    sizes=[p.numel() for p in parameters]
    for param,part in zip(parameters,(before+delta).split(sizes)):param.copy_(part.view_as(param))
    actual=torch.cat([p.detach().flatten() for p in parameters])-before
    after=[float(torch.dot(g,actual)) for g in protected]
    tolerance=[1e-5*float(g.norm())*float(actual.norm())+1e-8 for g in protected]
    if any(q>tol for q,tol in zip(after,tolerance)):
        raise ValueError('Applied FP32 displacement violated waveform guard tolerance')
    stats=dict(protected_gram=gram.tolist(),projection=solution,
        proposed_delta_norm=float(proposed.norm()),corrected_delta_norm=float(actual.norm()),
        correction_norm=float((actual-proposed).norm()),
        protected_dot_after_fp32=after,fp32_constraint_tolerance=tolerance)
    return actual,stats
