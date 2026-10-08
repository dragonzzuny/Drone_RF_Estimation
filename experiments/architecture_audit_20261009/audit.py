"""CPU-only structural checks; synthetic inputs, no held-out recordings.

Receptive-field calculation excludes spatial normalization/global conditioning.
It describes convolutional STFT paths, not the full functional dependency.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DENSE = ROOT / 'experiments/rfuav_dense_gated_20261008'
sys.path.insert(0, str(DENSE))
from models import build
from drone_rf.waveform import analyze, synthesize


def union(items):
    return (min(x[0] for x in items), max(x[1] for x in items))


def conv(intervals):
    return [union(intervals[max(0, i-1):i+2]) for i in range(len(intervals))]


def unet_support(length):
    """Exact temporal dependency intervals, positive bilinear coefficients only."""
    x = [(i, i) for i in range(length)]
    skips = []
    for level in range(5):
        if level:
            x = [union(x[2*i:2*i+2]) for i in range(len(x)//2)]
        x = conv(conv(x))
        skips.append(x)
    for skip in reversed(skips[:-1]):
        coordinates = (np.arange(len(skip))+.5)*len(x)/len(skip)-.5
        up = []
        for value, direct in zip(coordinates, skip):
            lower = int(np.floor(value))
            support = [x[min(max(lower, 0), len(x)-1)]]
            if value-lower > 1e-12:
                support.append(x[min(max(lower+1, 0), len(x)-1)])
            up.append(union(support+[direct]))
        x = conv(conv(up))
    return x


def run():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    result = dict(scope='synthetic CPU structure audit, not trained quality evaluation',
                  heldout_read=False, sample_rate_hz=100_000_000)
    wave = torch.complex(torch.randn(2, 63872), torch.randn(2, 63872))
    z = analyze(wave)
    restored = synthesize(z, wave.shape[-1])
    relative = float(((restored-wave).abs().square().sum()/wave.abs().square().sum()).item())
    assert relative < 1e-10
    result['stft'] = dict(n_fft=512, hop=128, complex_input=True,
                         complex_output=True, shape=list(z.shape),
                         window_microseconds=5.12, hop_microseconds=1.28,
                         roundtrip_nmse=relative, fftshift=False)
    net = build('unet_mean').eval()
    result['current_model_parameters'] = sum(p.numel() for p in net.parameters())
    # Binary-valued floats sum exactly: avoid attributing summation round-off
    # to sensitivity to the time order which has explicitly been discarded.
    feature = torch.randint(0, 8, (1, 65, 255)).float()
    shuffled = feature[..., torch.randperm(feature.shape[-1])]
    a = feature.mean(-1, keepdim=True).expand_as(feature)
    b = shuffled.mean(-1, keepdim=True).expand_as(feature)
    assert torch.equal(a, b)
    with torch.no_grad():
        ea = net.context_encoder(a)
        eb = net.context_encoder(b)
        ordered_a = net.context_encoder(feature)
        ordered_b = net.context_encoder(shuffled)
    assert torch.equal(ea, eb)
    result['context'] = dict(tokens=255, duration_ms=20.8896, token_step_ms=.08192,
        averaging_before_encoder=True, shuffled_input_max_difference=float((feature-shuffled).abs().max()),
        mean_context_encoded_max_difference=float((ea-eb).abs().max()),
        ordered_control_encoded_rms_difference=float((ordered_a-ordered_b).square().mean().sqrt()),
        interpretation='mean branch cannot distinguish input time permutations; ordered control is sensitive, not proof of better separation',
        frequency_conditioning='128-channel context projected to 1024 channels and broadcast over bottleneck frequency cells')
    # Full-capacity U-Net, small synthetic map only to check algebraic properties.
    with torch.no_grad():
        test_z = z[:1, :32, :32]
        out = net(test_z, feature, torch.zeros(1, dtype=torch.long))['estimates']
        rotated = net(1j*test_z, feature, torch.zeros(1, dtype=torch.long))['estimates']
    sum_error = float((out.sum(1)-test_z).abs().square().sum()/test_z.abs().square().sum())
    assert sum_error < 1e-10
    result['output'] = dict(complex_slots=4, source_slots=3, background_slots=1,
        mixture_sum_relative_error=sum_error,
        phase_rotation_difference_relative_energy=float((rotated-1j*out).abs().square().sum()/out.abs().square().sum()),
        phase_note='fresh weights, fixed-slot diagnostic only; not a trained phase-invariance performance estimate')
    del net, ea, eb, out, rotated
    gc.collect()
    wave_net = build('wavenet').eval()
    result['prepared_wavenet_parameters'] = sum(p.numel() for p in wave_net.parameters())
    try:
        wave_net(wave[:1], feature, torch.zeros(1, dtype=torch.long))
    except ValueError as exc:
        result['prepared_wavenet_native_compatibility'] = dict(passed=False, error=str(exc),
            reason='prepared historical adapter requires 256 context tokens; native run has 255')
    else:
        raise AssertionError('Expected legacy native-shape incompatibility no longer present')
    dilation = [m.filter_gate.dilation[0] for m in wave_net.blocks]
    receptive = 1 + sum((m.filter_gate.kernel_size[0]-1)*d for m,d in zip(wave_net.blocks, dilation))
    assert receptive == 6139
    result['wavenet_convolutional_path'] = dict(layers=30, residual_channels=128,
        dilations=dilation, samples=receptive, milliseconds=receptive/100_000,
        input_crop_ms=.63872, note='global RMS/mean power context add global summary dependence, not ordered phase-preserving context')
    intervals = unet_support(z.shape[-1])
    widths = [b-a+1 for a,b in intervals]
    central = intervals[len(intervals)//2]
    max_samples = 512+(max(widths)-1)*128
    result['unet_convolutional_path'] = dict(input_stft_frames=z.shape[-1],
        max_dependency_frames=max(widths), central_dependency_frames=central,
        max_input_sample_span=max_samples, max_span_ms=max_samples/100_000,
        note='pre-iSTFT STFT-bin convolutional paths only; excludes global RMS, GroupNorm spatial statistics, context, and iSTFT overlap')
    source_files = [Path(__file__), DENSE/'models.py', DENSE/'vendor/drone_rf/model.py',
                    DENSE/'vendor/drone_rf/context_model.py', DENSE/'vendor/drone_rf/waveform.py']
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
