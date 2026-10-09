"""Two physically overlapping native crops and supervised source correspondence.

Training-only helpers. Reference waveforms never enter the separator forward.
"""
import torch

SHIFT=16384


def paired_items(data,index):
    first=data[index]
    original=int(data.rows[index]['crop_start'])
    source_index=int(data.rows[index]['indices'][0])
    size=data.library.clips[source_index]['samples']
    offset=SHIFT if index%2==0 else -SHIFT
    if original+offset<0 or original+offset+data.length>size:
        offset=-offset
    assert 0<=original+offset and original+offset+data.length<=size
    data.rows[index]['crop_start']=original+offset
    try:
        second=data[index]
    finally:
        data.rows[index]['crop_start']=original
    return first,second,offset


def regions(length,offset):
    if not 0<abs(offset)<length:
        raise ValueError('A nonzero overlapping crop offset is required')
    lo=max(0,offset);hi=min(length,length+offset)
    return slice(lo,hi),slice(lo-offset,hi-offset)


def canonical(estimates,assignment):
    k=assignment.shape[1]
    ordered=estimates[:,:k].gather(1,assignment[...,None].expand(-1,-1,estimates.shape[-1]))
    return torch.cat([ordered,estimates[:,-1:]],1)


def consistency(a,b,reference,active,mixture):
    """Same true-source order and same common samples on both sides.

    All three padded source slots plus background are included. Active source
    denominators use their common-segment power with the existing numerical
    floor; inactive/background denominators use common mixture power.
    """
    if a.shape!=b.shape or a.shape[1]!=reference.shape[1]+1:
        raise ValueError('Wrong aligned source geometry')
    mix_power=mixture.abs().square().mean(-1).clamp_min(1e-8)
    ref_power=reference.abs().square().mean(-1)
    denominator=torch.where(active,torch.maximum(ref_power,1e-6*mix_power[:,None]),mix_power[:,None])
    err=(a-b).abs().square().mean(-1)
    return (err[:,:-1]/denominator).mean()+(err[:,-1]/mix_power).mean()
