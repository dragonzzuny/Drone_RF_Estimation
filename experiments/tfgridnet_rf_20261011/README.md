# Full TF-GridNetV2 blocks for complex RF

This is an RF adaptation, not a reproduction of published speech performance.
The author implementation is pinned to ESPnet commit
`3826362c7313d69d92eca721ec49441b6908c0b0`; Apache-2.0 license and attribution
are in `vendor/`. Six blocks, embedding128, BiLSTM hidden192 and four attention
heads follow the complete published V2 recipe, not a reduced bottleneck adapter.
The resulting RF model has9,004,297 parameters, including its context/count path.
Its parameter count differs from the existing32,142,859-parameter U-Net.

Each block performs full-band recurrence, per-frequency temporal recurrence,
and cross-frame attention. The entire512×500 fine grid is preserved for a
63,872-sample crop. There is no frequency/time pooling to the old32×31 U-Net
bottleneck. This sees0.63872ms of complex waveform at100MS/s; it does **not**
preserve20.8896ms of complex phase. The original mean long-context power
features remain as a separate conditioning/count path.

RF changes from the speech recipe are full two-sided512-bin complex STFT
(hop128, existing square-root Hann), ascending negative-to-positive frequency
order, three unordered source streams plus fixed background, existing mixture
STFT RMS scaling and sum projection, and the unchanged RF waveform/count loss.
The speech recipe uses128-point one-sided STFT and SI-SNR training. Therefore
this is a transfer candidate with explicit changes, not an exact paper replica.

For memory, independent LSTM batch sequences are processed in chunks32 with
activation recomputation. Every sequence, time step and parameter remains.
CPU float64 forward agreement with the unmodified author block was6.66e-15
maximum absolute difference. FP32 batching differed by3.34e-6 maximum and
2.85e-7 relative L2. The first overly strict elementwise CPU check failed near
zero; its receipt remains under `local/tfgridnet_cpu_20261011_v1`.
The corrected check adds an independent float64 comparison and full-model
backpropagation, rather than claiming bitwise FP32 identity.

CPU checks use all model parameters and2048 synthetic complex samples only.
The separate GPU preflight uses an original-length TRAIN three-source example,
one temporary optimizer update, and records all six block gradients and memory.
Preflight weights are discarded. It does not use DEV/heldout and is not a
performance evaluation. A prospective training protocol must be registered
after the full-length preflight passes.

References:

- [Wang et al., TF-GridNet, TASLP2023](https://arxiv.org/abs/2211.12433).
- [Author implementation and V2 integration](https://github.com/espnet/espnet/pull/5395).
- [Pinned block source](https://github.com/espnet/espnet/blob/3826362c7313d69d92eca721ec49441b6908c0b0/espnet2/enh/separator/tfgridnetv2_separator.py).

Current results and limitations: `reports/2026-10-11/TFGRIDNET_CPU_CHECK.json`
and the GPU preflight receipt when completed. No validation improvement has
been established by the implementation checks.
