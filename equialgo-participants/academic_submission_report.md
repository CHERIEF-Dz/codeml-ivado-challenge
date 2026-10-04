# Current submission: academic-only ranking

**File to test: `predictions.csv`.** Reproduce with `python model_corrige.py` from this directory. The identical archived candidate is `results_academic/predictions.csv`; its audit and SHA-256 are in `results_academic/submission_audit.json`.

The previous geography-neutralized logistic submission received a user-reported 90%, below the user's earlier result. The exact scorer metric and earlier score were not provided. This feedback does not identify which changed decisions were wrong. It is not a measured score for this new submission.

## Why this candidate

Both English and French challenge instructions explicitly state that historical committee decisions are not the target. They identify academic R score as explaining part of the legitimate regional selection difference, while income, hours worked, distance, and postal code carry regional information. The earlier candidate removed geographic effects but retained positive learned income and work-hour effects. That still allowed the audited committee's preferences to influence the allocation.

The new candidate therefore ranks solely by R score and selects the global top 40%. This is a transparent academic-merit hypothesis, not a reconstruction of the unknown reference. It keeps genuine differences in academic profiles instead of forcing equal regional award rates. It removes program and first-generation effects as well; the README does not specify how the independent reference weights either of them. Consequently this policy can miss legitimate financial-need or effort criteria if those are part of the hidden reference.

There is no empirical evidence that this new candidate maximizes hidden accuracy. Nor can we establish hidden equal opportunity using historical labels. The technical score combines equity (20 points) and utility (15 points), conditional on meeting the budget. Both require the unavailable independent reference. The user's external test is the necessary next measurement.

## Allocation and verification

- 4,000 unique candidate IDs, in the original evaluation-file order.
- Exactly 1,600 awards: 40%, within the mandatory 36–44% interval.
- Only binary decisions and the two required output columns.
- Centre selection: 42.875%; remote selection: 35.811%.
- 286 decisions differ from the previous 90% attempt.
- Lowest selected R score: 28.54. Candidate ID resolves exact R-score ties deterministically; ID is never used as a predictive feature.
- Tests cover budget, permutation-invariant tie handling, invariance to proxy changes, and invalid input rejection.

The earlier exploratory comparison used input-order ties and found 284 changed decisions. The delivered version uses ID-based ties to preserve decisions when input rows are reordered, yielding 286 changes.

Previous submissions and benchmarks are preserved under `results_improved/`. The earlier report describes those earlier results; its 88.80% historical accuracy does not apply to this academic-only candidate. Hidden-reference accuracy for the new file is **unknown**.
