# Research working rules

## Authorization and reporting

- User authorized completion of100epochs and four follow-ups, then autonomous evidence-driven research/experiments through the morning of2026-10-13 (08:00KST). See `reports/2026-10-13/OVERNIGHT_PLAN_KO.md`. Mentioning95 was NOT a stop instruction. All100epochs are complete. Do not revive old stopped queues or auto-extend to200.
- Report **every completed actual epoch in chat**: two/three-source NMSE, complex SI-SDR, weakest NMSE, changes from preceding epoch, selected model and next actual state. For short tracking epochs, a table containing every completed epoch is acceptable. Files/desktop notifications alone do not satisfy this. Verify STATE, process and AUDIT before claiming training is active; distinguish prepared/registered/waiting/running/evaluated/adopted. Detached workers cannot send chat messages themselves.
- Performance takes priority over real-time speed. Preserve full waveform model capacity. Initial comparisons use seed0; reuse completed controls rather than retrain them repeatedly.

## Live state:2026-10-13 06:20KST

- Parent B100 COMPLETE_AUDITED,7500updates. Original B630 macro-NMSE selects99: NMSE2/3=.2574137238/.3815573738, complexSI8.7230138/3.7962867, weakest.4401105690/.7223533810. `local/timeblock_extend100_20261012_v1`.
- Matched shared/independent-PIT studies each completed3epochs/288updates from99. Shared failed all8adoption checks. Independent3 modestly improves means but worsens triple tails. No global best replacement. `local/sequence_assignment_20261013_v1`.
- DualContextUNet completed3epochs/288updates,full36,554,507params,4,411,648added trainable,131parent tensors frozen. Final3 NMSE2/3=.3621748415/.5581097557, complexSI5.616089/1.286300, weakest.610741000/.821806219. All auditsPASS; worse than parent, not adopted or extended. `local/dual_context_20261013_v1`. Protocol ec715d7e89001fb2468991884e0917b2c6cdb62848b13b3c3aff497956adbbb5.
- Its registered fixed final3 full5context/known-packet/C630 evaluation is RUNNING, unit `rf-dual-context-followup-v1` PID602366 (verify), `local/dual_context_followup_20261013_v1`. Sources/plans PINNED.
- NEW registered supervised output tracker: frozen full U-Net99/C8 waveform teacher plus875,521param Transformer embedding on observed PSD/relative power. FIT48contexts,50epochs,12updates/epoch,600total,AdamW2.5e-4,seed0. B24everyepoch,C24final50only. Permutes existing waveform blocks; does not generate new waveforms or identify devices.
- Teacher unit `rf-tracking-teacher-v1` PID637156 is WAITING_DUAL_FOLLOWUP, `local/tracking_teacher_20261013_v1`. Tracker unit `rf-learned-tracking-v1` PID637220 is WAITING_FIT_TEACHER, `local/learned_tracking_20261013_v1`. Verify actual states; do not say RF tracking training started yet.
- Final CPU replay unit `rf-tracking-direct-audit-v1` PID637221 waits for50, `local/tracking_direct_audit_20261013_v1`. Recomputes observed features, embedding, orders and direct I/Q scores forB/C48. No references in forward/routing. Teacher/trainer/audit sources and two plans are PINNED; do not edit.
- GPU jobs serialize with existing `neural_queue.lock`. CPU diagnostics use2cores,low priority,bounded memory. New large runs reside on external Elements via local symlinks. Preserve source snapshots and data manifests.

## Evidence and selection boundaries

- Full20.8896ms B24/C24 routing studies completed. Parent waveforms fixed; only0.63872ms output-block orders change. Bv2 NMSE2/3=.224988/.512759 vs parent.358720/1.370470;6of19multi cases worsen. Constrained B=.229268/.489582 improves triple but worsens pair means/weakness versusv2. No universally best routing rule.
- C constrained NMSE2/3=.532317/.858304 but weak3=1.738647 exceeds parent1.604018. Count correctness1/2/3=.6/.333333/.714286 on24only. Oracle count sometimes worsens waveforms; always3clusters harms singles. Neither adopted.
- Reference-assisted optimal block permutations give B2/3=.086358/.183042,C2/3=.299677/.529332. Quarter blocks giveB.076386/.168218,C.269772/.508748. These are NONDEPLOYABLE diagnostics, not learned results,clean-noise bounds or SI-SDR optima. Keep existing block length for the first learned tracker.
- Existing C630 CURRENT_BEST stays native e2 U-Net+C8: NMSE2/3=.4446011356/.6889476042,complexSI4.249089/−2.735627,weakest.7343158631/.9486435775. `local/CURRENT_BEST.json`, `docs/CURRENT_BEST_KO.md`. Full24 and short630 are different evaluations; never rank their means together.
- Known FPV packet decodes in multiple outputs/background; mixture also passesCRC. Better NMSE/desired-output CRC does not prove exclusive separation or detector gain. No reference-based gain/phase fitting in actual inference.
- These are U-Net+C8/optional tracking results, NOT Deep NMF improvements. Do not claim perfect/general-purpose separation,first novelty or publication readiness without evidence.

## Data policy

- Read `docs/PLAN_KO.md` and `docs/DATA_POLICY_KO.md` before extending. Goal: complex I/Q reconstruction,generalization to unseen recordings/types,variable1–3source separation/count. Four-slot compatibility code does not schedule four-source experiments.
- Use one dataset and one original RF band per primary mixture. RFUAV stays100MS/s with native RF-coordinate offsets. Do not pool RFUAV with DroneDetect; future DroneDetect studies are separate at60MS/s. Old cross-band trials are exploratory history.
- Exclude RC and explicit combined aircraft/controller categories. Aircraft-origin telemetry/status/identification/video all qualify; lack of video is not exclusion. Product folders do not prove airframe-only emission. FPV recordings can contain uplink/downlink/ID intervals. Construction-source count is not verified physical drone count.
- B is VTSBW=10 same-record time separation with≥1s support gap, NOT independent-record validation. C changes recording andVTSBW=20 together. VTSBW means video/image transmission signal bandwidth,NOT receiver bandwidth/sample rate. Preserve misnamed pinned historical variables and clarify prose.
- Current5training labels:Avata2,FPV COMBO,Mavic3Pro,Mini3,Mini4Pro; same-band4pairs/1triple. Autel stays held out/unopened. RFUAV37categories do not imply37verified airframe-only I/Q classes. Other sources remain under provenance review; do not silently enlarge frozen cohorts.
- Split recording groups before mixtures/windows in independent A/C studies. B is explicitly limited. Never rebrand repeatedly viewed development data as untouched confirmation. No references,true count,categories or validation labels in deployed forward paths.
- Report dataset/split,budget,waveform metrics,limits and adverse cases. Reference I/Q includes receiver noise. Preserve metric tradeoffs.

## Operations and history

- User authorizes commits/pushes to `dragonzzuny/Drone_RF_Estimation`. Commit reviewed code/plans/verified reports only. Exclude raw I/Q,checkpoints,credentials,unrelated changes. Never `git add -A` or rewrite remote history.
- Pushed5e26484 includes paired success/adverse STFT PDF/SVG,export hashes,routing limits and prepared tracker code. Existing askpass uses active socket `/run/user/1000/vscode-git-3140698939.sock`; old socket was stale. Verify socket availability; do not expose credentials.
- Preserve unrelated dirty `PHASE_CONSISTENT_EPOCH_REPORTS_KO.md` and SepTDA diagnostics. Generated PNG previews need not be committed when PDF/SVG exports are recorded.
- User authorizes cleanup of obsolete regenerable artifacts after dependency checks,retained summaries/evidence and archive/deletion ledger. Preserve active recovery/caches and evidence needed for comparisons.
- Detector work,DR-NMF D2,SepTDA and prior stopped candidate queues remain stopped. Do not automatically restart integrated,TF-GridNet,phase-consistency or other historical runs.
- Earlier detailed rules/results are archived verbatim in `reports/2026-10-13/AGENTS_HISTORY_BEFORE_TRACKING.md`, hash ledger `local/AGENTS_HISTORY_ARCHIVE_20261013.json`. Consult for older studies; latest user instructions and current registered protocols govern current work.
