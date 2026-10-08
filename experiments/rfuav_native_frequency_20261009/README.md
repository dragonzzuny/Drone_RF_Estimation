# Native receiver-frequency-coordinate RFUAV study

This study implements the user's requested preservation of original receiver
frequency spacings. It is a **new observation geometry**, not a new U-Net
architecture or a performance improvement over historical center-aligned scores.
Only the already admitted five-category RFUAV development cohort is used.
Reserved recordings and Autel remain unopened. Controller-only and explicitly
combined labels remain excluded; transmitter provenance inside aircraft-named
recordings is still unresolved. Outputs represent **recorded contributions**,
not a verified physical aircraft count.

## Signal construction

| Original RF band | Common receiver center | Filter passband | Stopband begins | Native rate |
|---|---:|---:|---:|---:|
| 2.4 GHz | 2460 MHz | 2423–2497 MHz | below2421 / above2499 MHz | 100 MS/s |
| 5.8 GHz | 5780 MHz | 5753–5807 MHz | below5751 / above5809 MHz | 100 MS/s |

These bands lie within the intersection of the admitted receivers' theoretical
digital Nyquist intervals, with transition/edge margins. Actual analog receiver
passband response has not been calibrated. Receiver center metadata is not an
estimate of the source carrier. No cross-band or cross-dataset mixture is used.
The I+jQ convention and per-recording XML acquisition settings follow
[RFUAV section4.2](https://arxiv.org/html/2503.09033v2), rechecked2026-10-09.

For original receiver center `fc` and common center `f0`, first apply a
513-tap, beta8.6 Kaiser FIR centered at `f0-fc` in the original baseband. Then
multiply the filtered I/Q by `exp(j*2*pi*(fc-f0)*absolute_sample/fs)`.
The FIR is applied as delay-compensated **linear** convolution, not circular
FFT wrapping. Filtering precedes translation. Integer-modulo oscillator phase
keeps sample-origin/crop consistency. The rate remains100 MS/s.

Exclude4096samples from each edge: valid long contexts contain2,088,960samples
(20.8896ms), yielding255context tokens instead of256. Fine windows remain
63,872samples (0.63872ms). Original crop locations are retained unless they
intersect an excluded edge; those locations are clamped and their row indices
logged. The unchanged context network accepts255tokens without parameter changes.

Scale each **filtered valid long contribution** to its scheduled relative power,
apply the original random phases, and then crop. Fine windows are not individually
equalized. Features derive only from the exact same long mixture. Both mixture
and targets use the same filtered contributions. Source receiver noise remains.
The reconstruction target is the common-band-limited contribution, **not the
complete original wideband recording**. Filtering can remove substantial energy;
per-context retained-power fractions are saved, not used to exclude hard cases.

## Frozen budget and comparisons

- Original record-pack roles and all12,000TRAIN /630development-validation rows
  retained. TRAIN has3,902contexts from83files/8packs; validation144contexts from
  48files/5packs. Window counts are not independent recording counts.
- Current split still confounds VTSBW20 validation with recording-group changes.
  Repeated validation is development evidence, not an untouched test.
- Full32,142,859-parameter complex STFT U-Net;3unordered source slots and a
  separate background slot, mixture-sum constraint, count head. Parent is the
  frozen pre-allocation incumbent, SHA256
  `86470af0de460d6b71072e08fb19ec77bfb9d3545a41e7da728fad084870a356`.
- Existing supervised waveform PIT NMSE/coherence/inactive/background objective
  plus0.1construction-count cross entropy. The unsuccessful allocation KL is
  not added. References, identities and true count never enter prediction.
- Seed0, five epochs,2,400mixtures/75updates per epoch,375updates total,
  fresh AdamW lr1e-5 wd1e-4, clip1, batch32/microbatch2, float32/TF32off.
- Four fixed TRAIN cases,32discarded diagnostic updates, must improve fit and
  pass finite-gradient/memory/sum checks before restarting the parent for training.
- Evaluate frozen parent and every epoch on the **same native630mixtures**.
  Select minimum mean NMSE over counts2/3, including e0. Report NMSE, complex
  gain-invariant SI-SDR, weakest source, local power gap and construction count.
  Joint improvement requires lower NMSE and higher SI-SDR at both counts2/3.
  Parent versus adapted model explicitly has a different training-update budget.

## Simple spectral comparator

Build category mean PSD templates using TRAIN only: equalize context PSD sums,
average within each original file, then within pack, then within category.
At inference, fit nonnegative coefficients to the **observed long-mixture PSD**
against all templates in its receiver band. Use the fitted powers for constant
frequency masks, keep mixture phase, and synthesize I/Q. No true mixture category
subset or count is supplied. A fixed1%coefficient threshold produces a separate
construction-count diagnostic; no threshold search. Positive1e-8coefficient
smoothing is fixed before evaluation. This comparator uses known training
category signatures; it does not establish unknown-type generalization.

The comparator tests whether a full learned model outperforms simple spectral
partitioning under the revised geometry. It is not a reference-assisted ideal
mask, and component PSD additivity is a statistical approximation rather than
an exact complex-power identity.

Before observing this baseline's validation result, a second CPU comparison was
registered in `FRAME_PROTOCOL.json`: keep exactly the same TRAIN templates, but
fit nonnegative coefficients independently in each observed mixture STFT frame.
With at most3templates, enumerate all active sets and choose the feasible
least-squares minimum. Independent tests match SciPy NNLS and verify zero/single
active cases. The1e-8coefficient floor and mixture phase are unchanged. Carry over
the constant-mask baseline's **predicted** count only for count scoring; it never
controls waveform outputs. This tests local power adaptation, not hopping-period
learning or a new deep architecture. Both baseline results remain reported.

## Additional development diagnoses

After seeing the early count-head failure, `count_probe.py` registered two
mixture-only CPU classifiers: mean65 and mean+population-standard-deviation130
of the long-context sequence. StandardScaler and C=1 multinomial logistic
regression fit all12,000TRAIN mixtures and evaluate the same630development
mixtures without a hyperparameter search. A TRAIN-derived receiver-band majority
is also reported. These probes do not retain time order, replace the waveform
network, or establish physical-aircraft counting. They are adaptively chosen
diagnoses, not prospective independent tests.

`reference_diagnostic.py` separately examines all420development count2/3
mixtures with **reference-derived** frequency-mean and time-frequency power masks.
It requires unavailable reference I/Q and true count, retains mixture phase, and
is neither deployable separation nor an optimal performance bound. This result
does not establish that its side information can be inferred from a mixture.
`audit_auxiliary.py` reaggregates count predictions and all reference-diagnostic
rows, and verifies input/source identities without reopening raw recordings.

During the fifth native training epoch, `phase_native.py` registers a fixed
four-phase inference check on the eventual `SELECTED_005.pt`. Its weights and
epoch are selected by the original one-pass NMSE rule; there is no phase-based
reselection. Angles are0/90/180/270degrees; invert phase, align three source slots
using predictions alone, and take an equal complex mean with background fixed.
This reuses the previously tested inference implementation, with four forward
passes and zero training updates. All630native development mixtures are scored;
the one-pass outputs must reproduce the prior scores. Same-model NMSE and
complex SI-SDR must both improve at counts2/3, with weak-source metrics inspected
separately. It is an adaptive development comparison, not exact set equivariance
or new-type generalization. Its state and results reside in the run's `phase/`.

`fit_gap.py` follows this with the same selected one-pass model and no updates.
It fixes210examples from TRAIN epoch1 and210from epoch5 before inference,
matching the validation category-tuple proportions exactly (70cases per count).
The chosen model may not have seen later-epoch native mixtures: exact mixture
records are compared against every native epoch up to the selected checkpoint,
and exposure is reported, not assumed. Original TRAIN recordings are shared.
Saved630development-validation results provide a third comparison. This adaptive
diagnosis separates fit on seen mixtures, other mixtures of known recordings,
and changed recording/BW conditions; it cannot by itself prove absence of
memorization. Native source identities and CPU feature receipts are sealed.

`gradient_probe.py` fixes96TRAIN cases (32per count) from the registered epoch1
fit-gap subset and computes the waveform-loss and0.1count-CE gradients on the
603,136shared context-encoder parameters. No model update is made. Per-case and
mean-vector geometry are retained. The prospective follow-up trigger was a
negative overall cosine and a weighted count-gradient norm at least as large as
the waveform gradient. This diagnosis is local to a development-selected model,
not a causal estimate of validation degradation.

The trigger was met. `detach_count.py` runs a matched five-epoch ablation in a
separate run folder using the same pre-native parent, seed, prepared mixtures,
fresh optimizer and375updates as the completed native control. A forward pre-hook
detaches only the encoded input to `count_head`: the count head still learns;
the encoder learns from waveform loss. Full capacity and numerical forward values
are unchanged. GPU preflight requires bitwise identical predictions, no count
gradient on the encoder, a nonzero count-head gradient, and retained waveform
encoder gradients. Four-case fitting is then discarded and the common parent
reloaded. Every epoch is reaggregated against the control at the same prefix
budget; select by the existing mean count2/3 NMSE rule includinge0. No four-phase
inference is included in this training comparison. The run automatically writes
`COMPARISON_001` through`005` and completion/failure receipts.

The first phase-inference attempt failed before any scored case because the
reused raw-IQ batch adapter requested a nonexistent`long_mixture` field. That
attempt and its dependent cancelled fit-gap attempt are archived locally as
`phase_attempt1`/`fit_gap_attempt1`. The corrected native six-field batch adapter
passed all630one-pass reproduction checks. Model weights, inputs and budgets
were unchanged; failure records were retained.

## Implementation and checks

`native.py` implements filtering/translation; `native_data.py` is a separate
adapter. Old experiment code remains unchanged. Six numerical tests cover tone
placement, absolute phase/delay, alias rejection, guarded chunk consistency,
linearity/no circular wrap, filter pass/stop response and valid crop positions.
A separate spectral-baseline test checks two-tone reconstruction and mixture sum.
`diagnose.py` registers the first two TRAIN examples per category/level stratum
before reading I/Q; it audits48mixtures/100contexts without validation access.

`prepare.py` runs on CPU12/13 at low priority and seals code/data hashes, guarded
native caches, and mixture features. `run_native.py` waits for validation and
epoch1 readiness, then runs on GPU with the shared process lock; remaining CPU
feature preparation continues concurrently. `spectral_baseline.py` runs on CPU.
The protocols, full rows and checkpoints are retained locally. Local progress
files do not automatically send chat notifications after an assistant turn ends.

Experiment-design procedure follows the project's already cited Scientific
Agent Skills workflow: [Kassis et al.,2026](https://doi.org/10.48550/arXiv.2609.00065).
This is a software/procedure acknowledgment, not evidence of RF performance.
