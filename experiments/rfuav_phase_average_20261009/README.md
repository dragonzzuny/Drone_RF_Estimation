# Four-phase inference development experiment — 2026-10-09

Status at registration: CPU probes and three unit checks complete; full GPU
evaluation queued after the already frozen decoder-fusion training. No new
training, held-out recording access, or checkpoint selection by ensemble scores.

## Hypothesis and intervention

An I/Q separator should respond consistently when the complete mixture is
multiplied by a unit complex scalar. Imperfect phase consistency may contribute
to reconstruction error. This is a candidate mechanism, not an established cause
of our weak-source errors.

Evaluate the fixed angles 0, 90, 180, 270 degrees. Undo each input rotation in its
output. Match the three predicted source slots to the original-angle prediction
by the minimum unscaled whole-window squared difference over all 3! orders.
Keep the background slot fixed, then average all four complex outputs equally.
References, source identities and true source count never enter this operation.
There is no fitted gain, phase, angle search or selective case removal.

This costs four forward passes versus one. With ordered outputs, group averaging
has a useful symmetry interpretation; our prediction-anchored unordered-slot
matching is **not asserted to provide exact set equivariance**. It is also not
the group-convolution neural receiver in Gansekoele et al.

## Fixed evaluation

- Models: existing full STFT U-Net incumbent; late-fusion local-only control;
  late-fusion local/global candidate. No reduced-capacity replacement.
- Use their existing base-NMSE-selected checkpoints, including epoch 0 where the
  original protocol permits it. The late models finish five epochs first.
- Same 630 development mixtures: 210 each with 1, 2 and 3 recorded contributions.
  RFUAV only, native 100 MS/s, one original RF band within a mixture, source-record
  groups split before mixture generation. Center-aligned baseband synthesis does
  **not** retain original center-frequency spacing.
- Reproduce every baseline row against its prior evaluation before comparison.
- Primary endpoints: unscaled complex I/Q NMSE and explicitly complex-gain
  invariant SI-SDR, separately for two and three sources. Also retain one-source,
  weakest-source, local power-gap and count diagnostics.
- Directional acceptance: NMSE decreases and SI-SDR increases for **both** two
  and three sources. This is a development decision, not a significance test.
- A one-source input may equal its reference, giving infinite input SI-SDR and
  undefined/infinite improvement. Preserve its status; do not omit that case.
  Absolute output NMSE and SI-SDR remain evaluated for every case.
- Keep all model results. Comparisons within one checkpoint isolate inference
  averaging. STFT versus waveform architecture comparisons have different prior
  training histories and cannot be described as equal-budget architecture tests.

## Prior evidence and limits

Four **training** mixtures, fixed indices 4, 5, 2, 11, were inspected without
updates. The late full-context e1 model's case-average NMSE was 0.604353 versus
0.584527 with averaging; SI-SDR was −0.0116 versus 1.2749 dB. In the incumbent,
two-source NMSE slightly worsened (0.285732 → 0.287023); three-source NMSE slightly
improved (0.632487 → 0.632032). These probes motivated the candidate and are not
independent/generalization results. All four cases and modes are retained in the
local diagnostic receipts.

The development split has been used repeatedly for earlier model decisions.
Autel and reserved confirmation recordings stay sealed. No physical aircraft
count, whole-record identity tracking, genuine simultaneous RF capture or
general-purpose separation claim follows from this experiment.

## Reproducibility and execution

`run_phase.py --run <new-local-directory> --parent <decoder-comparison-run>`
waits for completion and the shared GPU lock. It checks source/data/checkpoint
hashes, preserves a source snapshot, evaluates all rows and audits aggregates.
It refuses silent reuse of completed model folders. Per-model completion and
queue/run/failure state are saved separately. It neither preempts GPU jobs nor
silently falls back to CPU.

The three unit tests verify prediction-only slot matching, invariance of an
already equivariant separator with reference fields blocked, and preservation of
a one-source case with infinite input SI-SDR.

## Methodological sources

Gansekoele, A., Bhulai, S., Hoogendoorn, M., and van der Mei, R. (2025).
*Relative Phase Equivariant Deep Neural Systems for Physical Layer
Communications*. Transactions on Machine Learning Research.
[Published record](https://ir.cwi.nl/pub/36105/),
[methods §§3.2–3.3](https://arxiv.org/html/2501.04730v2).
The source supports global phase symmetry as an RF inductive bias; its task is
neural reception/demodulation, not multi-drone waveform recovery.

Experimental-design and evidence-recording procedure assisted by Scientific
Agent Skills: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M.
(2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. [arXiv](https://doi.org/10.48550/arXiv.2609.00065).
