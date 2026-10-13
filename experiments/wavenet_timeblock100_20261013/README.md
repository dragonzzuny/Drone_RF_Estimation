# WaveNet on the fixed B100 comparison

This run trains a fresh 30-layer, 128-channel WaveNet with three complex source
outputs and a background output. It reuses the exact 100 training schedules and
630 validation cases of the completed U-Net control. See the frozen
[protocol](../../reports/2026-10-13/WAVENET100_PLAN_KO.md).

`launch_v2.py` prepares CPU features, registers immutable source/file hashes,
checks full-input GPU forward/gradient equivalence for microbatch 1 versus 4,
evaluates the random initialization, then trains 100 epochs / 7,500 updates.
The preflight update is discarded. Actual training uses a new seed-0 model.
The two models have different parameter counts and compute costs.

The original `prepare.py` / `launch.py` failed before registration on a
cross-device hard link. The v2 preparation uses verified byte-identical copies;
the original sources and failure receipt are preserved.

## Run and recovery

The current service is `rf-wavenet-b100-20261013-v2.service`. Large artifacts
are on the external disk via `local/wavenet_timeblock100_20261013_v1`.
Read `STATE.json`, the process state, and epoch audit receipts together.
`VALIDATING` with epoch 0 means no actual training updates have occurred.

Do not rerun the launcher once `PROTOCOL.json` exists. After a genuine failure,
first verify the process has stopped, inspect `FAILURE.json`, and preserve the
registered sources. `train.py --resume` restores the most recent audited LAST
or optimizer RECOVERY, including optimizer/scheduler/RNG state. A failure before
either checkpoint exists requires separate examination; it is not resumable
from trained weights. Do not start duplicate GPU jobs. The shared GPU lock is
acquired by the trainer.

The CPU worker prepares the following epoch concurrently. Only completed,
regenerable feature arrays are deleted, with their hashes and reason recorded.
Schedules, checkpoints, recovery files, source snapshots, and audits remain.

## Reporting

Every epoch is evaluated and audited. The separate
`publication_followup_20261013/watch_wavenet_notify.py` worker generates reports
at completed epochs 5, 10, ..., 100 and sends local desktop notifications. It
also announces the first actual training update and terminal states locally.
The notification receipt records command completion, not whether the user read
the notice. These workers do not send chat messages.

The five-epoch report compares the actual weights at the same epoch and lists
each model's selected epoch separately. It reports NMSE, complex SI-SDR,
weakest-source NMSE, and count accuracy. B is a temporal split within recordings;
its results are not an independent-recording test. C and Autel are not used in
training or model selection for this run.
