# Matched phase inference after power-allocation training

Registered while the five-epoch parent comparison is still running. Evaluate
both `waveform_only` and `waveform_allocation` using their parent's existing
`SELECTED_005.pt` rule (minimum count2/3 base NMSE, including e0). Do not select
epochs, angles, examples or weights using the new phase scores.

Reuse the verified fixed 0/90/180/270-degree inference implementation. Undo each
input rotation, align the three outputs using predicted whole-window waveforms
against the zero-degree output, keep background fixed, then average equally.
References/count/identities enter scoring only. No gradient updates; both models
have 32,142,859 parameters and the same five-epoch fine-tuning budget. Four-phase
inference requires four forward passes instead of one.

Evaluate all 630 existing development cases. Reproduce each model's saved
one-pass scores before reporting phase results. The main comparison is candidate
four-phase versus control four-phase, requiring lower NMSE and higher complex
SI-SDR for counts2/3 separately. Also compare both against the already evaluated
incumbent four-phase output, with identical input identities checked. This
second comparison is an operational incumbent check; additional training makes
it a different budget from the incumbent. Neither comparison is an independent
test or a significance claim. Preserve weakest-source and power-gap diagnostics.

Wait for completed parent receipts, verify the parent checkpoints/sources/data,
acquire the shared GPU lock and check native CUDA. Keep original group splits,
same-band center-aligned synthesis, recorded noise, and the unopened heldouts.
This does not implement native RF center offsets or validate physical aircraft
count, whole-record tracking, identification, or exact set equivariance.

Implementation reuses `rfuav_phase_average_20261009` unchanged. See its README
and the method evidence report for the motivation and primary literature.
