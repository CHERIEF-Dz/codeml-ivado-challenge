# Deep accuracy investigation

## Decision

**Baseline-research snapshot:** the externally tested 94.6% submission is archived at `results_effort/predictions.csv`. The active file on the family-income branch is described in `family_income_report.md`. Its archive is `results_effort/predictions.csv`, SHA-256 `0120f838852cf652c01bcee39ee0ef453d2591a4b5394d033a506b4d48de18c6`. The prior 92.23% file remains in `results_academic/`. The original README is unchanged from commit `ef7047a`.

A separate **unscored** candidate is ready at `predictions_next_test.csv`. It modestly favors financial need, preserves 40% allocation, and changes 112 decisions from the 94.6% file. It must be evaluated externally before replacing the current best. No measured result above 94.6% is available.

## What actually produced the gain

The 92.23% candidate ranked only academic R score. The 94.6% candidate ranks:

```
R score + 0.14381742782570264 * weekly work hours
```

The hours weight was learned conditionally from historical decisions while controlling for income, geography, and other variables. Only R score and working hours enter the final rank. This exchanged 112 selected applicants for 112 others while keeping 1,600 awards. It improved the reported accuracy by approximately 2.37 percentage points. The later 94.6% report was supplied as text, without its exact evaluator row or macro F1.

A candidate-ID tie break is deterministic and independent of input order. Identifiers do not enter any predictive model.

## Challenge interpretation

The README says historical `decision_octroi` is not the judging target. Improving agreement with those decisions does not establish improved independent-reference accuracy. Geography and socioeconomic variables can reproduce committee bias.

The official English and French PDFs add an important detail on page 1: HxBuddy displays indicative accuracy and F1, while IVADO assesses the official rubric separately. The official technical rubric weights equity 20 and utility 15, with zero outside the 36–44% budget. The reported 94.6 is an indicative accuracy result, not94.6% of that rubric.

The original README was restored exactly and was not changed during this investigation. Its requirements remain authoritative; these notes live separately.

## Investigation completed

### Data integrity and distribution

There are no missing cells, overlapping train/evaluation IDs, duplicate feature rows, or identical feature rows across training and evaluation. Numeric two-sample distribution checks give KS p-values from approximately 0.346 to 1.000. These checks found no obvious dataset mismatch; they cannot rule out every type of shift. See `research/results/data_and_score_audit.json`.

Accuracy 92.23 and macro F1 91.9, combined with 1,600 predicted positives, are compatible with a true positive prevalence near 40%. The exact range depends on rounding precision and whether the evaluator uses all 4,000 rows. With macro F1 shown to two percentage decimals, the compatible range is approximately 1,593–1,603 positives; with one decimal it is wider. This does not identify individual hidden labels. It supports retaining a 40% budget rather than increasing grants merely to improve recall.

### Model flexibility: 340 fits

Twelve configurations were compared using common region/label-stratified five-fold outer validation and three-fold inner model selection. They cover raw/log numeric terms, logistic versus probit links, splines in R score and/or work hours, and interactions. One hundred additional bootstrap fits assess coefficient stability. All fits converged in the completed run.

| Historical-label diagnostic | Out-of-fold accuracy | Log loss |
|---|---:|---:|
| Linear logistic, mixed numeric terms |88.74%|0.260084|
| Logistic with work-hours spline |88.75%|0.259930|
| Logistic with interactions |88.78%|0.259952|
| Probit with flexible numeric splines |88.86%|0.260976|
| Nested model selection by inner log loss |88.79%|0.260112|

The model-specific figures are descriptive; the nested-selection estimate accounts for choosing a model within each training fold. These small differences provide no persuasive evidence for replacing the simple ranking with a more flexible predictor of committee decisions. They do not set a ceiling on the independent reference.

The earlier random-forest and gradient-boosting benchmark remains available in `results_improved/`.

### Work-hours stability

For the richer logistic diagnostic, outer-fold work-hour weights were 0.1402–0.1461 R points/hour. Across 100 bootstrap fits, the mean was 0.14344 and the 2.5th–97.5th percentiles were 0.13119–0.15912. The tested coefficient 0.143817 is near the centre. These quantify historical coefficient variation, not uncertainty about the hidden reference definition.

Changing the coefficient to 0.15 would change only four decisions, limiting its absolute possible accuracy change to 0.10 percentage points on 4,000 evaluated rows. That adjustment alone cannot move 94.6% to 95%. Large coefficient changes are also not supported by the coefficient-stability evidence.

### Public fairness baseline

The README's baseline EOP gap 0.270 is a scoring normalization constant, not a set of labels. Reproducing the original forest and evaluating it under different hypothetical merit definitions produces different gaps; baseline training population, reference noise, and evaluation provenance remain unknown. We therefore did not use 0.270 as a target to reverse engineer a purported true reference. The current nearly equal regional selection rates do not establish hidden equal opportunity.

## Why financial need is the next informative trial

The earlier geographically neutral model still rewarded higher income and received a rough 90% report. Removing income and hours together improved to 92.23%; restoring hours alone improved further to 94.6%. This gives a reason to investigate income separately, but does not prove that a negative income effect belongs in the reference.

We checked a simple family of hypothetical reference mechanisms:

```
latent merit = R + a * work_hours + b * ln(income) + independent noise
```

Under Gaussian and logistic noise assumptions, fitting three aggregate feedback values (92.23%,94.6%, and an assumed 90% for the first candidate) gives work-hour weights near 0.134–0.135 and log-income weights near−0.66 to−0.68. This motivates the rounded trial score:

```
R + 0.135 * weekly_hours - 0.67 * ln(household_income / 60000)
```

At equal R score and working hours, lower income receives a modest preference. Doubling income decreases the score by approximately0.46 R points. The candidate exchanges 56 awardees for 56 others; selection rates are 39.21% centres and 41.15% remote.

**This is hypothesis generation using leaderboard feedback, not validation.** The first 90% report was approximate and its metric breakdown was not supplied. If it instead meant 91.5–92%, the inferred need preference shrinks markedly. Independent-noise and additive-score assumptions may be wrong. First-generation status, program, unobserved criteria, or different scoring provenance could also explain the residual errors.

The resulting scenario calculations imply roughly 95.4% agreement for their own assumed mechanism when the old score is 90%, but this is **not measured accuracy and not a reliable forecast**. Other mechanisms fit the same aggregate information. These values are explicitly labeled hypothetical in `research/results/score_sensitivity.csv`; they are not attached to the trial submission as an achieved score. Repeated leaderboard-guided choices also reduce the independence of that leaderboard as an evaluation set.

## Reproduction and artifacts

Run from the challenge directory using the recorded dependencies:

```bash
python model_corrige.py                 # Reproduce the tested effort submission
python model_need.py                    # Write ONLY the separate unscored next trial
python research/compare_models.py       # 340 historical fits; nested CV and bootstrap
python research/data_and_score_audit.py # Data audit and illustrative two-score sensitivity
python research/score_sensitivity.py    # Illustrative three-score sensitivity
python -m unittest discover -p 'test_model_*.py' -v
```

Model-analysis seeds: outer 147, inner 247+fold, bootstrap 341. `research/results/stats.json` contains per-model and per-fold metrics. External feedback is recorded separately in `external_evaluation_log.json`, keyed by file hash. `best_submission.json` identifies the externally tested best.

Sixteen targeted tests pass. Both current and next-trial files have exactly the required two columns, 4,000 matching unique candidate IDs, binary decisions, and 1,600 awards. The README and both scored submission hashes were verified unchanged. The next useful evidence is the evaluator's accuracy and macro F1 for `predictions_next_test.csv`; an IVADO equity result would also clarify the official-rubric tradeoff.
