"""Full-model, overlapping frequency-tile inference with prediction alignment.

All 512 complex FFT bins and the full time window are retained. This is a
test-time distribution change, not a trained band-split RNN reproduction.
"""
import itertools
import torch


def geometry(total, width=128, step=64, device='cpu'):
    if total % step or width != 2*step or width>total or width<16:
        raise ValueError('Use two-fold circular overlap on all frequency bins')
    indices=[(torch.arange(width,device=device)+start)%total for start in range(0,total,step)]
    weights=torch.sin(torch.pi*(torch.arange(width,device=device,dtype=torch.float64)+.5)/width).square()
    denominator=torch.zeros(total,device=device,dtype=torch.float64)
    for index in indices:denominator[index]+=weights
    assert torch.allclose(denominator,torch.ones_like(denominator),atol=1e-12,rtol=0)
    return indices,weights,denominator


def align(prediction, anchor, weights):
    if prediction.shape!=anchor.shape or prediction.shape[1]!=4:
        raise ValueError('Three source slots plus fixed background required')
    permutations=list(itertools.permutations(range(3)))
    costs=torch.stack([((prediction[:,list(p)].to(torch.complex128)-anchor[:,:3].to(torch.complex128)).abs().square()
                        *weights[None,None,:,None]).sum((1,2,3)) for p in permutations],-1)
    chosen=costs.argmin(-1)
    order=torch.tensor(permutations,device=prediction.device)[chosen]
    order=torch.cat((order,torch.full_like(order[:,:1],3)),1)
    result=prediction.gather(1,order[:,:,None,None].expand_as(prediction))
    return result,order,costs


def predict(net,z,context,position,width=128,step=64):
    # Inputs contain only mixture spectrum/context/position, never references.
    original=net(z,context,position)
    anchor=original['estimates']
    indices,weights,denominator=geometry(z.shape[1],width,step,z.device)
    combined=torch.zeros_like(anchor);trace=[]
    for index in indices:
        tile=net(z[:,index],context,position)['estimates']
        matched,order,costs=align(tile,anchor[:,:,index],weights)
        combined[:,:,index]+=matched*weights[None,None,:,None].to(matched.real.dtype)
        trace.append(dict(indices=index.tolist(),order=order.tolist(),costs=costs.tolist()))
    combined=combined/denominator[None,None,:,None].to(combined.real.dtype)
    return dict(parent=anchor,tiles=combined,blend50=.5*(anchor+combined)),original['count_logits'],trace
