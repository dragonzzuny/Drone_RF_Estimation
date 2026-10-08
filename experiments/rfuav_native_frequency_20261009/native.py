"""Guarded, native-rate RF-coordinate replay of admitted complex recordings.

Receiver center is coordinate metadata, not an estimated transmitter carrier.
The output is a band-limited recorded contribution, including receiver noise.
"""
from functools import lru_cache
import numpy as np
from scipy.signal import firwin, oaconvolve

FS = 100_000_000
INPUT_LENGTH = 2**21
GUARD = 4096
LENGTH = INPUT_LENGTH - 2 * GUARD
WINDOW = 63872
TAPS = 513
BETA = 8.6
BANDS = {
    '2.4GHz': dict(center_hz=2_460_000_000, pass_half_hz=37_000_000,
                   stop_half_hz=39_000_000, admitted_centers=[2450000000, 2470000000]),
    '5.8GHz': dict(center_hz=5_780_000_000, pass_half_hz=27_000_000,
                   stop_half_hz=29_000_000, admitted_centers=[5760000000, 5765000000, 5800000000]),
}


def band_for(center_hz):
    for name, spec in BANDS.items():
        if center_hz in spec['admitted_centers']:
            return name
    raise ValueError('Receiver center not admitted to this experiment')


@lru_cache(maxsize=8)
def kernel(center_hz):
    spec = BANDS[band_for(center_hz)]
    shift = center_hz - spec['center_hz']
    if abs(shift) + spec['stop_half_hz'] >= FS / 2:
        raise ValueError('No digital transition guard before translation')
    low = firwin(TAPS, (spec['pass_half_hz'] + spec['stop_half_hz']) / 2,
                 window=('kaiser', BETA), fs=FS)
    n = np.arange(TAPS) - TAPS // 2
    # Pre-translation RF band center in the ORIGINAL baseband coordinates.
    return low * np.exp(-2j * np.pi * shift * n / FS)


def oscillator(length, shift_hz, start_sample):
    if not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
        raise ValueError('Nonnegative absolute integer sample origin required')
    if not isinstance(shift_hz, (int, np.integer)):
        raise ValueError('This protocol uses integer-Hz receiver metadata')
    # Integer modulo before float conversion prevents large-offset phase loss.
    n = np.arange(length, dtype=np.int64) + int(start_sample)
    phase = ((n % FS) * int(shift_hz)) % FS
    return np.exp((2j * np.pi / FS) * phase)


def transform(iq, center_hz, start_sample=0, guard=GUARD):
    x = np.asarray(iq)
    if (x.ndim != 1 or not np.iscomplexobj(x) or len(x) <= 2 * guard
            or guard < TAPS // 2 or not np.isfinite(x).all()):
        raise ValueError('Finite complex data and a sufficient edge guard required')
    spec = BANDS[band_for(center_hz)]
    # Linear convolution, delay compensated by mode=same; no circular wrap.
    filtered = oaconvolve(x, kernel(center_hz), mode='same')
    region = filtered[guard:-guard]
    shifted = region * oscillator(len(region), center_hz - spec['center_hz'],
                                  int(start_sample) + guard)
    return shifted.astype(np.complex64)


def crop_start(original_start):
    if not 0 <= original_start <= INPUT_LENGTH - WINDOW:
        raise ValueError('Original crop outside admitted context')
    # Preserve original physical samples except the explicitly logged edge cases.
    return min(max(int(original_start), GUARD), INPUT_LENGTH - GUARD - WINDOW) - GUARD


def mean_power(x):
    return float(np.mean(np.abs(np.asarray(x, dtype=np.complex128)) ** 2))


def spectral_overlap(first, second, nfft=1024):
    """Histogram intersection of normalized average PSDs; not active occupancy.

    Includes receiver noise. A value of one means identical mean PSD shape,
    not that the waveforms are equal or provably impossible to separate.
    """
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    profiles = []
    for x in (first, second):
        frames = x[:len(x) // nfft * nfft].reshape(-1, nfft)
        p = np.mean(np.abs(np.fft.fft(frames * window, axis=-1))**2, axis=0)
        profiles.append(p / p.sum())
    return float(np.minimum(*profiles).sum())
