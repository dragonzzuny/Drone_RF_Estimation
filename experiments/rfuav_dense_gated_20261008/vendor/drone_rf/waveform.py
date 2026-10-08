"""Complex STFT round trip, mixture-only prediction and explicit waveform metrics."""
import torch

from .losses import pit_waveform_loss


def sqrt_hann(device):
    return torch.hann_window(512, periodic=True, device=device).sqrt()


def analyze(waveform):
    return torch.stft(waveform, n_fft=512, hop_length=128, window=sqrt_hann(waveform.device),
                      center=True, pad_mode='reflect', onesided=False, return_complex=True)


def synthesize(spectrum, length):
    shape = spectrum.shape
    wave = torch.istft(spectrum.reshape(-1, *shape[-2:]), n_fft=512, hop_length=128,
        window=sqrt_hann(spectrum.device), center=True, onesided=False, length=length, return_complex=True)
    return wave.reshape(*shape[:-2], length)


def predict(model, batch):
    """Never pass source references, identities or a ground-truth count forward."""
    result = model(analyze(batch['mixture']), batch['context_features'], batch['crop_start'])
    return synthesize(result['estimates'], batch['mixture'].shape[-1]), result['count_logits']


def complex_si_sdr(estimate, reference):
    """Mean removed; invariant to a constant COMPLEX gain, including phase.

    This convention is disclosed explicitly, not assumed identical to a real
    audio implementation. It complements unscaled complex I/Q NMSE.
    """
    e = estimate - estimate.mean(-1, keepdim=True)
    r = reference - reference.mean(-1, keepdim=True)
    energy = r.abs().square().sum(-1, keepdim=True)
    valid = (energy[..., 0] > 0) & (e.abs().square().sum(-1) > 0)
    alpha = (e * r.conj()).sum(-1, keepdim=True) / torch.where(energy > 0, energy, 1.)
    projection = alpha * r
    numerator = projection.abs().square().sum(-1)
    denominator = (e - projection).abs().square().sum(-1)
    score = 10 * torch.log10(numerator / denominator)
    # Never turn an absent/constant output into a plausible finite dB score.
    return torch.where(valid, score, torch.full_like(score, float('nan')))


def waveform_metrics(estimates, references, active, mixture):
    value = pit_waveform_loss(estimates, references, active, mixture)
    indices = value['assignment']
    aligned = estimates[:, :3].gather(1, indices[..., None].expand(-1, -1, references.shape[-1])).to(torch.complex128)
    references = references.to(torch.complex128)
    mixture = mixture.to(torch.complex128)
    power = references.abs().square().mean(-1)
    mix_power = mixture.abs().square().mean(-1).clamp_min(1e-8)
    nmse = (aligned - references).abs().square().mean(-1) / torch.where(power > 0, power, 1.)
    return dict(assignment=indices, nmse=nmse, reference_power=power,
        si_sdr=complex_si_sdr(aligned, references),
        input_si_sdr=complex_si_sdr(mixture[:, None].expand_as(references), references),
        inactive_leak=(aligned.abs().square().mean(-1) / mix_power[:, None]) * ~active,
        background_nmse=estimates[:, -1].abs().square().mean(-1) / mix_power,
        sum_relative_error=(estimates.sum(1) - mixture).abs().square().mean(-1) / mix_power)
