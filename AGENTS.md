# Research working rules

## Authorization and reporting

- User authorized completion of100epochs and four follow-ups, then autonomous evidence-driven research/experiments through the morning of2026-10-13 (08:00KST). See `reports/2026-10-13/OVERNIGHT_PLAN_KO.md`. Mentioning95 was NOT a stop instruction. All100epochs are complete. Do not revive old stopped queues or auto-extend to200.
- Report **every completed actual epoch in chat**: two/three-source NMSE, complex SI-SDR, weakest NMSE, changes from preceding epoch, selected model and next actual state. For short tracking epochs, a table containing every completed epoch is acceptable. Files/desktop notifications alone do not satisfy this. Verify STATE, process and AUDIT before claiming training is active; distinguish prepared/registered/waiting/running/evaluated/adopted. Detached workers cannot send chat messages themselves.
- Performance takes priority over real-time speed. Preserve full waveform model capacity. Initial comparisons use seed0; reuse completed controls rather than retrain them repeatedly.

## Live state:2026-10-13 08:05KST

- Overnight work and latest steering completed this experiment batch. All scheduled GPU/CPU jobs COMPLETE_AUDITED. No training is running or queued. Publication/generalization goal is NOT marked achieved. Read `reports/2026-10-13/MORNING_REPORT_KO.md`.
- Parent B100 completed7500updates; B630 selects99. Shared/independent PIT3each and DualContext3 plus full5/packet/C630 followups finished. DualContext not adopted.
- Learned tracker50/600updates,875521params,full U-Net99 frozen: B24 NMSE2/3=.143027/.242095,SI10.417/7.130,weak.228390/.439353. C24=.517603/.710601,SI−.952/−1.812,weak.754059/1.319718. FIT48 teacher and independent B/C48 CPU I/Q auditsPASS. All50epochs reported in chat.
- Matched positionless tracker50/600 completed,875520trainable parameters,time_gain fixed0. B=.148457/.472485,SI9.865/5.001,weak.213375/1.098440; C=.488258/.815615,SI.263/−2.207,weak.724358/1.218208. Independent CPU48PASS. All50epochs reported in chat. `local/positionless_tracking_20261013_v2`; cancelled v1 performed0updates, exact sources archived before cleanup. All executed sources/plans remain pinned.
- Existing native e2+C8 full C24 evaluation complete. Parent NMSE2/3=.655874/.761105; fixed PSD=.426044/.674980,SI3.448/−1.521,weak.595265/.955783; transferred B99-learned50=.458999/.753204,SI2.926/−2.825,weak.635950/1.041373. Fixed PSD is preferable for this parent on multicomponent means, with adverse cases retained. `local/incumbent_tracking_c24_20261013_v1`,24auditsPASS.
- B/C long evaluations do not replace short C630 CURRENT_BEST. Position has conditional tradeoffs; never claim universal temporal benefit. Oracle-count and label-objective diagnostics do not solve all errors.
- Next proposed study NOT IMPLEMENTED/REGISTERED/RUNNING: train a native e2-specific tracker from FIT48 outputs, matched600update budget and existing PSD comparator. Check native parent training provenance before treating any B curve as heldout; C24 remains exposed development. No Autel access.
- GPU jobs use existing neural_queue.lock; large artifacts on external Elements. Keep final checkpoints/feature caches/source snapshots. Preserve user changes and unrelated stopped queues.

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
