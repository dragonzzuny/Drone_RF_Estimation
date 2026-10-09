"""Common receiver-coordinate augmentation; relative RF spacing is unchanged.

Multiply ALL components and the mixture by the SAME oscillator. The physical
scene is unchanged when the receiver reference is changed from f0 to f0-delta.
This is not independent carrier jitter or cross-band mixture construction.
Only integer steps fs/64 admit an exact roll of cached 64-band context features.
"""
import numpy as np


FS=100_000_000
STEP_HZ=FS/64


def oscillator(samples,start,steps):
    if not isinstance(steps,(int,np.integer)) or steps not in (-2,-1,0,1,2):raise ValueError('Registered steps only')
    if not isinstance(start,(int,np.integer)) or start<0 or samples<1:raise ValueError('Invalid crop')
    phase=((np.arange(samples,dtype=np.int64)+int(start))%64)*int(steps)
    return np.exp(2j*np.pi*(phase%64)/64).astype(np.complex64)


def augment(item,steps):
    result=dict(item)
    if item['context_features'].shape[0]!=65:raise ValueError('Expected 64 spectral bands + envelope')
    carrier=oscillator(item['mixture'].shape[-1],int(item['crop_start']),steps)
    result['mixture']=(item['mixture']*carrier).astype(np.complex64)
    result['references']=(item['references']*carrier[None]).astype(np.complex64)
    features=item['context_features'].copy()
    features[:64]=np.roll(features[:64],steps,axis=0)
    result['context_features']=features
    result['receiver_coordinate_shift_hz']=steps*STEP_HZ
    return result
