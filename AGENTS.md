# Research working rules

- This repository is the user-authorized destination for future commits and pushes of drone RF separation work. Commit only reviewed research code, plans, and verified reports. Do not add raw datasets, checkpoints, credentials, or unrelated work from the previous repository.
- Exclude remote-controller classes and explicit combined aircraft/controller labels from the aircraft cohort. Keep ambiguous transmitter provenance visible; a product folder name is not proof of airframe-only emission.
- Primary objective: complex I/Q reconstruction, generalization to new source recordings/types, then variable 1–4 source separation and count estimation. Performance takes priority over real-time speed. Preserve full model capacity. Use seed 0 for the initial matched comparison.
- Split original recording groups before making windows or synthetic mixtures. Do not claim new windows are independent recordings. Keep held-out data out of model/threshold selection, and keep any previously reserved confirmation files unopened under their existing protocol.
- Existing detector training and DR-NMF D2 remain on user-requested hold. Legacy N power/magnitude trials run in the previous workspace; do not move their files or change frozen source while running. A user message can steer the research but does not itself cancel unrelated authorized trials.
- Report planned, prepared, queued, running, and evaluated states separately. Check the actual process and saved progress before claiming training is active. Never claim perfect separation, general-purpose operation, or publication readiness without evidence.
- Every published result must identify the dataset/split, comparison budget, waveform metrics, and limitations. Keep failed conditions alongside successful ones. A NMF-free U-Net result is not a Deep NMF improvement.
- Read `docs/PLAN_KO.md` and `docs/DATA_POLICY_KO.md` before extending this experiment.
