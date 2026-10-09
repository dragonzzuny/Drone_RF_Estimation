"""Synthetic-only proof checks for IQ/context agreement and RF interval invariance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from coordinates import augment,oscillator,FS,STEP_HZ

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/rfuav_dense_gated_20261008/vendor'))
from drone_rf.context_data import mixture_context_features


def run():
    rng=np.random.default_rng(0);length=8192*8;start=4097;crop=8192
    n=np.arange(length);carriers=np.array([-176,-48,208])/1024
    envelope=(.5+.4*np.sin(n/700))[None]
    symbols=np.repeat(np.exp(.5j*np.pi*rng.integers(0,4,size=(3,length//32))),32,axis=1)
    sources=(symbols*envelope*np.exp(2j*np.pi*carriers[:,None]*n)).astype(np.complex64)
    mixture=sources.sum(0);features=mixture_context_features(mixture)['context_features']
    item=dict(mixture=mixture[start:start+crop],references=sources[:,start:start+crop],
        context_features=features,crop_start=start)
    errors=[]
    for steps in (-2,-1,0,1,2):
        changed=augment(item,steps);long=mixture*oscillator(length,0,steps)
        direct=mixture_context_features(long)['context_features']
        context_error=float(np.max(np.abs(direct-changed['context_features'])))
        assert context_error<2e-6
        # Crop-aligned oscillators must be identical. Complex64 products can
        # differ by a final rounding bit for aligned/unaligned array kernels;
        # assess that separately instead of requiring product bit identity.
        assert np.array_equal(oscillator(crop,start,steps),oscillator(length,0,steps)[start:start+crop])
        crop_error=float(np.mean(np.abs(changed['mixture']-long[start:start+crop])**2)/np.mean(np.abs(long[start:start+crop])**2))
        assert crop_error<1e-12
        assert np.array_equal(changed['context_features'][64],features[64])
        sum_error=float(np.mean(np.abs(changed['references'].sum(0)-changed['mixture'])**2)/np.mean(np.abs(changed['mixture'])**2))
        assert sum_error<1e-12
        power=np.mean(np.abs(item['references'].astype(np.complex128))**2,axis=-1)
        new_power=np.mean(np.abs(changed['references'].astype(np.complex128))**2,axis=-1)
        power_error=float(np.max(np.abs(new_power/power-1)));assert power_error<3e-7
        # Analytic absolute frequencies: new reference + shifted baseband.
        original=5.78e9+carriers*FS
        reexpressed=(5.78e9-steps*STEP_HZ)+(carriers*FS+steps*STEP_HZ)
        assert np.array_equal(original,reexpressed)
        assert np.array_equal(np.diff(carriers*FS),np.diff(carriers*FS+steps*STEP_HZ))
        errors.append(dict(steps=steps,context_max_error=context_error,crop_relative_error=crop_error,
            crop_oscillator_bitwise_equal=True,sum_relative_error=sum_error,power_relative_error=power_error))
    assert 39e6+2*STEP_HZ<FS/2 and 29e6+2*STEP_HZ<FS/2
    files=(Path(__file__),Path(__file__).with_name('coordinates.py'),
        ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/context_data.py',
        ROOT/'experiments/rfuav_dense_gated_20261008/vendor/drone_rf/temporal.py')
    return dict(status='PASS',steps=[-2,-1,0,1,2],step_hz=STEP_HZ,rows=errors,
        physical_centers_and_relative_spacing_preserved=True,passband_nyquist_guard_checked=True,
        limitation='Coordinate identity only; residual FIR stopband leakage is not identically zero; no trained improvement',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        synthetic_only=True,recorded_iq_reads=0,gpu_use=False,training_implemented=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
