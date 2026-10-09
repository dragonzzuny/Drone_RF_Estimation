"""Prospective loss ablation; preserve the original waveform PIT matching.

This is log1p of each already-normalized source error, not log-STFT,
SA-SDR, or a new separation architecture. Inactive-slot error is transformed
too, exactly as in the preceding head-gradient diagnostic. Background and
coherence terms stay unchanged. Evaluation must use untransformed metrics.
"""
import head_gradient_probe as diagnostic
from drone_rf.losses import pit_waveform_loss
import torch

MODES = ('original', 'log1p_nmse')


def objective(estimates, references, active, mixture, mode):
    if mode not in MODES:
        raise ValueError('Unknown registered loss')
    original = pit_waveform_loss(estimates, references, active, mixture)
    if mode == 'original':
        return original
    nmse, coherence = diagnostic.aligned_terms(
        estimates, references, active, mixture, original['assignment'])
    source = (torch.log1p(nmse) + coherence).mean()
    return dict(loss=source + original['background_loss'], source_loss=source,
                background_loss=original['background_loss'], assignment=original['assignment'])
