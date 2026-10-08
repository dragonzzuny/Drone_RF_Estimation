"""Exploratory temporal recurrence diagnostics, not a protocol decoder.

Correlation of power/occupancy traces is not a full cyclic spectral analysis.
Block-shuffle ranks are descriptive controls, not calibrated significance tests.
"""
import numpy as np


def lag_similarity(features, max_lag):
    """Demean each feature; normalize each lag by its overlap energies."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or not np.isfinite(x).all() or not 0 <= max_lag < x.shape[1]:
        raise ValueError('Finite [features, time] input and valid lag required')
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    transform = np.fft.rfft(x, n=1 << (2 * n - 1).bit_length(), axis=1)
    correlation = np.fft.irfft(np.sum(np.abs(transform) ** 2, axis=0))[:max_lag + 1]
    energies = np.sum(x ** 2, axis=0)
    prefix = np.r_[0., np.cumsum(energies)]
    lags = np.arange(max_lag + 1)
    denominator = np.sqrt(prefix[n - lags] * (prefix[n] - prefix[lags]))
    result = np.zeros(max_lag + 1)
    np.divide(correlation, denominator, out=result, where=denominator > 1e-30)
    return np.clip(result, -1., 1.)


def local_peak(curve, min_lag, max_lag):
    """Require a local peak with two neighbors; ties choose the earlier lag."""
    indices = [i for i in range(max(1, min_lag), min(max_lag, len(curve) - 2) + 1)
               if curve[i] > 0 and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]]
    if not indices:
        return None
    index = min(indices, key=lambda i: (-round(float(curve[i]), 10), i))
    return dict(lag_frames=index, correlation=float(curve[index]))


def block_shuffle_rank(trace, peak_value, min_lag, max_lag, seed, repeats=63, block=64):
    """Compare selected peak with each control's maximum over the whole lag range."""
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 1 or len(x) % block or repeats < 1:
        raise ValueError('Trace must comprise complete blocks')
    rng = np.random.default_rng(seed)
    blocks = x.reshape(-1, block)
    maxima = []
    for _ in range(repeats):
        shuffled = blocks[rng.permutation(len(blocks))].reshape(-1)
        curve = lag_similarity(shuffled, max_lag)
        maxima.append(float(curve[min_lag:max_lag + 1].max()))
    return dict(control_max_q95=float(np.quantile(maxima, .95)),
        exceedance_fraction=float((1 + np.count_nonzero(np.array(maxima) >= peak_value)) / (repeats + 1)),
        repeats=repeats, block_frames=block)


def power_features(iq, fft_size=1024, bands=64):
    """Nonoverlapping periodic-Hann FFTs; coarse profiles retain frequency bins.

    Returned envelope uses unwindowed power. Spectral profiles sum to one per
    nonzero frame, removing common amplitude variation from the shape trace.
    """
    if iq.ndim != 1 or not np.iscomplexobj(iq) or len(iq) % fft_size or fft_size % bands:
        raise ValueError('Complex whole FFT blocks and divisible band count required')
    if not np.isfinite(iq).all():
        raise ValueError('Nonfinite I/Q')
    blocks = np.asarray(iq).reshape(-1, fft_size)
    envelope = np.mean(np.abs(blocks.astype(np.complex128)) ** 2, axis=1)
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(fft_size) / fft_size)
    spectrum = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = (np.abs(spectrum) ** 2).reshape(len(blocks), bands, fft_size // bands).sum(-1)
    total = power.sum(1, keepdims=True)
    profile = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    return envelope, profile.T
