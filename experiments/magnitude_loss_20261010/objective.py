"""Native complex-I/Q waveform objective plus non-log STFT magnitude L1.

Source assignment remains the original whole-window waveform PIT assignment.
Magnitude supervision never changes slots independently across time/frequency.
"""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/source_interaction_20261010'))
import train_comparison as worker
from drone_rf.waveform import analyze


def spectral_terms(estimates, references, active, mixture, assignment):
    b, k, t = references.shape
    aligned = estimates[:, :k].gather(1, assignment[..., None].expand(-1, -1, t))
    e = analyze(aligned.reshape(b*k, t)).reshape(b, k, 512, -1)
    r = analyze(references.reshape(b*k, t)).reshape(b, k, 512, -1)
    m = analyze(mixture)
    em, rm = e.abs(), r.abs()
    denom = torch.maximum(rm.mean((-2, -1)), 1e-3*m.abs().mean((-2, -1))[:, None]).clamp_min(1e-12)
    relative_l1 = (em-rm).abs().mean((-2, -1))/denom
    energy = rm.square().sum((-2, -1)).clamp_min(1e-24)
    mag_nmse = (em-rm).square().sum((-2, -1))/energy
    phase_part = (2*(em*rm-(e*r.conj()).real)).sum((-2, -1))/energy
    complex_nmse = (e-r).abs().square().sum((-2, -1))/energy
    rms_ratio = (em.square().sum((-2, -1))/energy).sqrt()
    # Average active sources within each mixture; each mixture has equal weight.
    loss = ((relative_l1*active).sum(-1)/active.sum(-1).clamp_min(1)).mean()
    return dict(loss=loss, relative_l1=relative_l1, magnitude_nmse=mag_nmse,
                phase_interaction=phase_part, spectral_complex_nmse=complex_nmse,
                spectral_rms_ratio=rms_ratio)


def objective(estimates, logits, item, weight=.1):
    base = worker.pit_waveform_loss(estimates, item['references'], item['active'], item['mixture'])
    ce = torch.nn.functional.cross_entropy(logits, item['construction_count']-1)
    main = base['loss']+.1*ce
    terms = spectral_terms(estimates, item['references'], item['active'], item['mixture'], base['assignment'])
    return dict(loss=main+weight*terms['loss'], main=main, magnitude=terms['loss'], assignment=base['assignment'])
