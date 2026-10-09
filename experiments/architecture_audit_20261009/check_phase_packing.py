"""Full-capacity CPU candidate checks on one existing TRAIN mixture; not training."""
import argparse
import gc
import inspect
from pathlib import Path
import sys
import time

import numpy as np
import torch

import phase_packing as pp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_native_frequency_20261009'))
from native_data import NativeMixtures, sha256, write_json
from drone_rf.context_data import component_gains


def check(preparation, output):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    x=torch.randn(2,pp.SAMPLES,dtype=torch.complex64)
    packed=pp.pack_iq(x)
    restored=pp.unpack_iq(packed)[:,0]
    assert torch.equal(x,restored)
    # Check the real/imaginary channel permutation's adjoint, not just values.
    short=torch.randn(2,320,dtype=torch.complex64,requires_grad=True)
    pp.unpack_iq(pp.pack_iq(short)).abs().square().sum().backward()
    torch.testing.assert_close(short.grad,2*short.detach(),rtol=1e-6,atol=1e-6)
    del x,packed,restored,short
    gc.collect()
    local=pp.PhasePackedWaveNet('local')
    long=pp.PhasePackedWaveNet('long')
    for key,value in local.state_dict().items():
        assert torch.equal(value,long.state_dict()[key]),key
    assert sum(p.numel() for p in long.parameters())==pp.PARAMETERS
    assert len(long.net.blocks)==30
    assert long.net.blocks[0].filter_gate.in_channels==128
    convolution_samples=(1+sum(2*b.filter_gate.dilation[0] for b in long.net.blocks))*pp.PACK
    assert convolution_samples==4_194_208
    # A reference full output decoding must agree with decoding only selected
    # features, including nonaligned and both endpoint crop coordinates.
    h=torch.randn(1,128,pp.SAMPLES//pp.PACK)
    starts=[0,1,31,32,pp.SAMPLES-pp.FINE-1,pp.SAMPLES-pp.FINE]
    crop_errors=[]
    with torch.no_grad():
        full=pp.unpack_iq(long.net.output(h),sources=4)
        for start in starts:
            small=pp.decode_crop(long.net.output,h,torch.tensor([start]))
            ref=full[:,:,start:start+pp.FINE]
            torch.testing.assert_close(small,ref,rtol=1e-5,atol=1e-6)
            crop_errors.append(float((small-ref).abs().max()))
    # Check the new output head remains differentiable.
    pp.decode_crop(long.net.output,h,torch.tensor([31])).abs().square().mean().backward()
    assert torch.isfinite(long.net.output.weight.grad).all()
    assert long.net.output.weight.grad.abs().sum()>0
    long.zero_grad(set_to_none=True)
    del full,h,ref,small
    gc.collect()
    data=NativeMixtures(preparation,'train_pack',1)
    index=4
    item=data[index]
    row=data.rows[index]
    count=int(row['count'])
    indices=row['indices'][:count]
    gains=component_gains([data.library.clips[int(i)]['mean_power'] for i in indices],
                          row['levels'][:count],row['phases'][:count])
    mixture=np.zeros(pp.SAMPLES,np.complex64)
    for i,gain in zip(indices,gains):
        mixture+=(data.library._array(int(i))*gain).astype(np.complex64)
    start=int(item['crop_start'])
    np.testing.assert_array_equal(mixture[start:start+pp.FINE],item['mixture'])
    mix=torch.from_numpy(mixture)[None]
    position=torch.tensor([start])
    feature=torch.from_numpy(item['context_features'])[None]
    # Both scopes must be identical when all out-of-crop complex samples are
    # zero. Shared mean context/RMS and all parameter tensors are held fixed.
    masked=pp.visible_iq(mix,position,pp.FINE,'local')
    torch.testing.assert_close(masked[:,start:start+pp.FINE],mix[:,start:start+pp.FINE],rtol=0,atol=0)
    assert torch.count_nonzero(masked[:,:start])==0
    assert torch.count_nonzero(masked[:,start+pp.FINE:])==0
    begin=time.time()
    local.eval();long.eval()
    with torch.no_grad():
        a=local(masked,feature,position)
        b=long(masked,feature,position)
        torch.testing.assert_close(a['estimates'],b['estimates'],rtol=0,atol=0)
        c=long(mix,feature,position)
        for result in (a,b,c):
            assert result['estimates'].shape==(1,4,pp.FINE)
            assert torch.isfinite(result['estimates']).all()
        reference=torch.from_numpy(item['mixture'])[None]
        sum_error=float(((c['estimates'].sum(1)-reference).abs().square().mean()/reference.abs().square().mean()))
        assert sum_error<1e-9
        context_sensitivity=float((c['estimates']-b['estimates']).abs().square().mean())
    assert list(inspect.signature(long.forward).parameters)==['long_mixture','context_features','crop_start','fine_samples']
    result=dict(status='PASS',device='cpu',trained=False,gpu_preflight_completed=False,
        train_index=index,validation_iq_read=False,heldout_read=False,
        native_crop_matches_original=True,packing_exact=True,packing_gradient_checked=True,
        crop_starts_checked=starts,crop_max_abs_errors=crop_errors,output_head_gradient_checked=True,
        initial_parameter_tensors_identical=True,parameters=pp.PARAMETERS,
        long_samples=pp.SAMPLES,long_milliseconds=pp.SAMPLES/100_000,
        packed_tokens=pp.SAMPLES//pp.PACK,packing=pp.PACK,input_real_channels=2*pp.PACK,
        convolutional_receptive_field_samples=convolution_samples,
        theoretical_receptive_field_ms=convolution_samples/100_000,
        input_limits_actual_information=True,fine_samples=pp.FINE,
        identical_output_for_identical_visible_iq=True,mixture_sum_relative_error=sum_error,
        full_vs_masked_input_output_difference=context_sensitivity,
        difference_is_not_quality_improvement=True,full_forwards_cpu_seconds=time.time()-begin,
        forward_inputs='observed long complex mixture, its own power features, crop coordinates',
        source_sha256={p.name:sha256(p) for p in (Path(__file__),Path(pp.__file__))},
        preparation_sha256=sha256(preparation/'PREPARATION.json'),time=time.time())
    write_json(output,result)
    print(__import__('json').dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    check(a.preparation,a.output)
