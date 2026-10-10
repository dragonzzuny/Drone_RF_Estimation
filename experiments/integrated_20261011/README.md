# Integrated RF I/Q experiment

The full retained U-Net combines ordered long-context features, a dual-axis
adapter, and a balanced source-interaction head. All parent parameters remain
frozen; the waveform update guard applies to the new trainable parameters only.
The original same-band RFUAV data, source-recording split, native sampling rate,
loss, and waveform scoring remain in use. This is a performance experiment,
not a factorial attribution of each component's contribution.

The first launch (`integrated_20261011_v1`) failed in input-hash verification:
a string was passed to a Path-only hashing helper. No CUDA preflight or optimizer
update occurred. The corrected source was sealed in `integrated_20261011_v2`.
The failed registration and traceback remain preserved locally.

All GPU checks and epoch results are under the v2 run and
`reports/2026-10-11/INTEGRATED_*`. Temporary preflight weights are discarded.
No raw data or checkpoints belong in Git. The previous SepTDA run and its
individual-candidate queue were stopped by the user's explicit instruction.

Procedural design guidance: the local `experimental-design` skill was used for
predeclared comparison/stop rules and recording-level replication limits. Its
software reference, verified against the current arXiv record on 2026-10-11, is:
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner
(2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
This is procedural software attribution, not evidence of RF separation accuracy.
