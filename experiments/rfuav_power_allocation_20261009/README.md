# Supervised relative spectral power allocation — 2026-10-09

This is a prospective, matched **supervised** fine-tuning trial. It follows the
four-phase inference evaluation in the GPU queue and does not alter that study.
Before main training, both arms must pass the same full-capacity GPU diagnostic.
No improvement is claimed from implementation or unit tests.

## Why this trial

On all 420 existing two/three-source development mixtures, a diagnostic using
true reference time-frequency powers produced NMSE 0.180521/0.302237, compared
with the incumbent U-Net's 0.494315/0.660468. Reference frequency-wise time-mean
powers produced 0.294618/0.440128. These are **reference-assisted diagnostics**,
not deployable models, proof that these powers are inferable from a mixture, or
nonlinear separation performance bounds. No reference powers enter inference.

Hypothesis: explicit supervision of relative time-frequency power allocation can
improve unscaled I/Q restoration in the existing full STFT U-Net. This is a new
development hypothesis, not a verified drone-specific loss or novelty claim.

## Fixed intervention

The model remains the 32,142,859-parameter STFT U-Net, including its mean power
context and count head. Its complex outputs are inverse-STFT reconstructed I/Q.
The source signals, identities and counts are used only during supervised loss
calculation and subsequent scoring; model prediction receives mixture inputs.

Use the current waveform PIT assignment, one permutation per complete scored
window, to align source outputs; background remains fixed. Compute output and
reference STFT powers using the existing 512-point FFT, hop128, sqrt-Hann setup.

At each time-frequency bin, let `q_k = |S_k|² / sum_j |S_j|²`. Define the predicted
fractions from output powers with relative `1e-8` numerical smoothing. Compute
`KL(q || p)` across three source slots plus background. The background target
is zero under this synthesis protocol. Weight bins by their total reference
power, normalized within each example, then average examples. This prevents
nearly empty bins from dominating this particular auxiliary term. It does not
remove any example from the original waveform loss or evaluation.

Power fractions use the **sum of component powers**, not the mixture power:
complex interference cross terms mean those are generally different. The
auxiliary term alone does not specify phase or absolute scale. Both arms retain
the unchanged waveform NMSE, coherence, inactive/background and count objectives.

- `waveform_only`: original objective; no auxiliary gradient.
- `waveform_allocation`: original objective + **0.1 × allocation KL**.
- One prespecified weight. No validation-angle, weight, FFT-resolution or case
  filtering search is part of this trial.
- This is relative power supervision, not the earlier log-STFT loss, magnitude
  image reconstruction, a new network architecture, or unsupervised learning.

## Matched resources and evaluation

Both arms restart at the identical incumbent: dense same-band U-Net selected
additional e1 after the earlier common e22 history; checkpoint hash
`86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356`.
Both use fresh AdamW, lr1e-5, weight decay1e-4, gradient clip1, seed0, float32,
TF32 disabled, batch32 with microbatch2. GroupNorm prediction consistency and GPU
memory are checked before the comparison. Per-arm budget: **5 epochs, 12,000
mixtures, 375 updates**. Arm order alternates each epoch. Wall-clock cost is
recorded; matching updates does not mean the auxiliary backward pass is free.

Before training: four fixed training cases [4,5,2,11], 32 updates per arm. Check
finite gradients, memory, microbatch predictions and improving fit relative to
initialization. Discard those weights. This diagnoses implementation, not
generalization, and does not require the candidate to beat the control there.

Use the same RFUAV 3,902 training contexts and 144 validation contexts; same
original-record group split, same-band center-aligned synthesis, 100MS/s. Preserve
long-region power scaling before local crops. No new recordings or reserved
heldout files. Original center-frequency spacings are not preserved. References
include receiver noise, and aircraft folder labels do not prove airframe-only
emission in every sample.

Validate all 630 mixtures every epoch; reconstruct initial incumbent scores.
Select by mean NMSE over counts2/3, including epoch0. Report both counts' NMSE and
complex SI-SDR, weakest source, local power gap, single-source and count results.
Adoption requires lower NMSE and higher complex SI-SDR for both counts2/3 against
the matched control. Repeated development selection is not an independent test
or a statistical significance claim.

The rationale and verified sources are in
[the method evidence report](../../reports/2026-10-09/METHOD_EVIDENCE_AND_DECISIONS_KO.md).
Wang et al. (2021) warn that magnitude and waveform objectives can trade off;
their speech result does not establish the effectiveness of this RF KL term.
