"""Mixture-only long-context features and one-to-three-recording construction.

Construction count is bookkeeping, not an automatically valid aircraft count.
No short-window renormalization or reference-derived input features are used.
"""
import numpy as np

from .temporal import power_features


def component_gains(powers, levels_db, phases):
    """Shared long-context gains for both feature preparation and short crops."""
    powers, levels, phases = (np.asarray(v, dtype=np.float64) for v in (powers, levels_db, phases))
    if powers.ndim != 1 or not len(powers) or levels.shape != powers.shape or phases.shape != powers.shape:
        raise ValueError('Matching one-dimensional mixing parameters required')
    if not np.isfinite(np.r_[powers, levels, phases]).all() or np.any(powers <= 0):
        raise ValueError('Finite positive powers and finite levels/phases required')
    weights = 10. ** ((levels - levels.max()) / 10.)
    return np.sqrt(weights / weights.sum() / powers) * np.exp(1j * phases)


def mixture_context_features(mixture, fft_size=1024, bands=64, pool_frames=8):
    """[65,256] features for 2**21 samples with the default geometry.

    Band occupancy and relative envelope are averaged over contiguous frames.
    All normalization is computed from the observed mixture. Time coordinates
    are sample-index centers, including the half-sample offset of even blocks.
    This auxiliary view discards phase; the fine complex branch retains it.
    """
    block_samples = fft_size * pool_frames
    if pool_frames < 1 or bands < 1 or fft_size < 2 or len(mixture) < block_samples:
        raise ValueError('Invalid context geometry')
    if len(mixture) % block_samples:
        raise ValueError('Use complete context tokens, without silent truncation')
    envelope, profile = power_features(mixture, fft_size, bands)
    envelope_scale = max(float(envelope.mean()), 1e-30)
    feature = np.concatenate([np.log1p(bands * profile),
                              np.log1p(envelope[None] / envelope_scale)], axis=0)
    feature = feature.reshape(bands + 1, -1, pool_frames).mean(-1).astype(np.float32)
    return dict(context_features=feature,
                context_first_center=(block_samples - 1) / 2.,
                context_step_samples=block_samples,
                context_samples=len(mixture))


def contextual_mixture(recordings, powers, levels_db, phases, crop_start,
                       window_samples=63872, fft_size=1024, bands=64, pool_frames=8):
    """Create 1/2/3 source targets padded to three slots, plus a zero background.

    Gains are fixed for the full context; the short input is literally cropped
    from that same complex mixture. Recording noise stays in each source.
    ``count_eligible`` stays False pending independent activity/provenance review.
    """
    count = len(recordings)
    if count not in (1, 2, 3) or any(len(v) != count for v in (powers, levels_db, phases)):
        raise ValueError('One to three recordings and matching mixing parameters required')
    shape = np.shape(recordings[0])
    if len(shape) != 1 or not shape[0] or any(np.shape(x) != shape or not np.iscomplexobj(x) for x in recordings):
        raise ValueError('Equal nonempty one-dimensional complex contexts required')
    if not isinstance(crop_start, (int, np.integer)) or not isinstance(window_samples, (int, np.integer)):
        raise ValueError('Crop coordinates must be integer sample counts')
    if crop_start < 0 or window_samples < 1 or crop_start + window_samples > shape[0]:
        raise ValueError('Crop outside long mixture')
    gains = component_gains(powers, levels_db, phases)
    long_mix = np.zeros(shape, dtype=np.complex64)
    references = np.zeros((3, window_samples), dtype=np.complex64)
    region = slice(crop_start, crop_start + window_samples)
    for i, (record, gain) in enumerate(zip(recordings, gains)):
        component = (record * gain).astype(np.complex64)
        if not np.isfinite(component).all():
            raise ValueError('Nonfinite source recording')
        long_mix += component
        references[i] = component[region]
    result = mixture_context_features(long_mix, fft_size, bands, pool_frames)
    mixture = long_mix[region].copy()
    if not np.array_equal(mixture, references.sum(0)):
        raise ValueError('Long/short mixture mismatch')
    result.update(mixture=mixture, references=references,
                  active=np.any(references != 0, axis=1),
                  background_reference=np.zeros_like(mixture), crop_start=int(crop_start),
                  construction_count=count, count_eligible=False)
    return result
