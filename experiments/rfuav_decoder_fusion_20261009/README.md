# Late decoder fusion of local waveform and long complex context

This is a new, bounded development experiment. The preceding fine/coarse FiLM
trial did not jointly improve NMSE and complex SI-SDR. Four training examples
showed very small output sensitivity to its added coarse conditioning. This
motivates testing a shorter path from context features to the I/Q output; it
does not prove the mechanism of the earlier validation failures.

## Intervention and controlled comparison

Keep the full 32,355,203-parameter waveform U-Net from the selected short-context
e5 checkpoint. Both arms contain 33,494,851 parameters and start identically.
The common parent's 1,500 updates are distinct from the 1,500 additional updates
per arm in this comparison. Earlier FiLM trial weights are not reused.

Local input remains the same 63,872-sample scored window inside a masked
1,048,576-sample array. The added branch sees either the masked array
(`local_only`) or the complete 10.48576 ms complex mixture (`local_global`).
Both retain the existing mean power features from approximately 21 ms.

The context encoder is unchanged: lossless pack16, four learned stride4
convolutions, 256 ordered 128-channel tokens, and seven dilated TCN blocks.
The learned compression is lossy. The context token spacing is 40.96 us;
interpolation does not restore phase information discarded by compression.

At the last decoder output, align context to native feature centers, concatenate
local64 and context128, and apply Conv1d(192,64,k3), SiLU, Conv1d(64,64,k1).
Add this correction to the local features at scored-region centers, immediately
before the unchanged I/Q head. The last correction projection is initialized
with small nonzero weights (std 1e-3). There are no fine/coarse FiLM modules.
Outputs remain three unordered recorded-source contributions and background,
followed by the existing mixture-sum projection. No references, source identity,
or true source count are inference inputs.

Two arms, seed0, five additional epochs each, 2,400 mixtures and 300 optimizer
updates per epoch; effective batch8, microbatch4. Same admitted schedule and
fresh AdamW optimizers, local cosine LR 1e-5→1e-6, added branch 10×, float32,
gradient clip1, unchanged PIT NMSE/coherence, background and count objectives.
The full-context arm runs first in odd epochs and second in even epochs.

The new information contrast is controlled within this architecture. Comparison
to the preceding FiLM trial is descriptive because connection type, location,
and total parameters differ. It is not an isolated location-only experiment.

## Gates, selection and interpretation

CPU tests check native time alignment, exact preservation of parent weights,
identical initial parameters and optimizer coverage. CUDA checks use training
data only: full shape, exact parent bypass, label independence, mixture sum,
microbatch agreement, context perturbation and nonzero finite branch gradients.
Each arm then performs 32 discarded updates on the first two count2 and first
two count3 training examples. Both objective and mean unscaled NMSE must fall
before development validation starts. This fit is implementation evidence only.

Choose checkpoints by minimum validation mean NMSE across counts2/3, including
epoch0. Directional acceptance requires lower NMSE and higher complex SI-SDR
for both counts at matched total budget. Tiny directional differences are not
evidence of statistical or practical superiority. Report all epochs, weak
components and local power strata, retaining failures and low-power examples.

RFUAV only, same original band, center-aligned synthetic mixtures; native
center-frequency offsets are not preserved. Grouped recordings remain frozen.
Heldout Autel is unopened. The count means 1–3 recorded contributions, not proven
physical aircraft count. Targets include receiver noise. Validation VTSBW20
differs from training VTSBW10/40/60. Reusing development cases for architecture
selection does not create an independent test.

The paired design follows the previously applied experimental-design skill;
its procedural reference is Kassis et al. (2026), *Scientific Agent Skills*,
https://doi.org/10.48550/arXiv.2609.00065. It is not RF performance evidence.
