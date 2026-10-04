# Family-income priority policy

## Branch and submission

This option is implemented on **`feature/family-income-priority`**. The prior 94.6% model and completed audits were saved on `improve-model-accuracy` at commit `bb51395` before branching.

**The current `predictions.csv` on this branch uses the income-aware policy.** It is also archived at `results_family_income/predictions.csv`. Its external accuracy is unknown. The tested 94.6% file remains unchanged at `results_effort/predictions.csv`; `best_submission.json` still identifies it as the best measured submission. `active_submission.json` identifies the new policy being tested.

The original README is unchanged.

## User-selected policy

The user selected a modest advantage for lower-income applicants when merit is close, rather than only an exact-tie rule. Gross estimated family income is used as an indicator of potential parental financial support. The dataset does not measure actual transfers, family size, wealth, or whether parents are willing to provide support, so income remains an imperfect proxy for financial need.

The ranking is:

```
score = R score + 0.14381742782570264 * weekly work hours
        - 0.67 * ln(family income / 60000)
```

We keep the work-hours coefficient from the tested 94.6% model. The only new substantive factor is the negative income contribution. At equal R score and working hours, lower family income strictly raises the ranking score. At identical final scores, lower income is preferred before candidate ID resolves remaining ties. Region and postal code do not enter the allocation score.

Halving income adds approximately **0.4644 R-score points**. Therefore an applicant from a $30,000 household has a 0.4644-point advantage over an otherwise identical applicant from a $60,000 household, and a 0.9288-point advantage over one from a $120,000 household. The $60,000 denominator is a neutral reference constant, not a financial eligibility cutoff; changing it would shift every score equally and leave the allocation unchanged.

The 0.67 need weight is an initial policy setting previously explored in aggregate-score sensitivity analysis. It is not a learned causal effect or a validated optimum for hidden labels. It can be adjusted explicitly with `--need-weight`; a sweep is saved in `results_family_income/policy_sensitivity.csv`. No historical committee-income coefficient is reused as a positive wealth advantage.

## Measured allocation changes

The selected count remains **1,600 of 4,000 (40%)**, within the README's mandatory 36–44% interval. Compared with the tested effort-only submission, 106 decisions change: **53 applicants enter and 53 leave**.

| Measure | Newly selected | No longer selected |
|---|---:|---:|
| Applicants |53|53|
| Median family income |$33,154|$99,839|
| Median R score |27.97|28.82|
| Median weekly working hours |13|10|

This explicitly trades some academic/working merit near the cutoff for financial need. It should be judged as a policy choice, not described as a free accuracy gain.

Across all selected applicants:

| Measure | Tested effort-only policy | Income-aware policy |
|---|---:|---:|
| Mean R score |30.7441|30.7227|
| Median R score |30.35|30.35|
| Median family income |$62,125|$59,187.50|
| Centre selection rate |40.008%|39.039%|
| Remote selection rate |39.988%|41.400%|

These are observable allocation statistics, not hidden-reference accuracy or equal-opportunity results. The next external submission will establish whether the change improves the indicative accuracy/F1 metrics. The official IVADO rubric also includes equity; its hidden-reference EOP cannot be measured locally.

## Implementation and checks

`model_family_income.py` validates inputs, computes the monotonic need score, uses deterministic income/ID tie breaking, and writes the branch's submission and audit. `model_corrige.py` now invokes this policy. The earlier exploratory `model_need.py` used a different hours coefficient of 0.135; its artifacts remain preserved and should not be confused with this controlled change to the tested model.

The existing density audit scripts now reference `results_effort/predictions.csv` explicitly, so their historical 94.6% audit is not silently relabeled as an audit of this new policy. This keeps prior evidence reproducible as the active submission changes.

Tests cover lower-income priority at equal merit, the doubling-income effect, monotonic selection when one applicant's income decreases, deterministic row-order-independent allocation, absence of direct geography effects, zero-weight behavior, invalid inputs, and empty change-group statistics. Submission checks verify the exact candidate IDs, two required columns, binary decisions, and fixed budget. The prior scored CSV and original README are verified byte-for-byte unchanged.

Run from the challenge directory:

```bash
python model_corrige.py
python model_corrige.py --need-weight 0.5  # Optional explicit weaker policy
python -m unittest discover -p 'test_model_*.py' -v
```

The first command uses the approved default of 0.67. The second regenerates the active CSV with a different strength; it does not establish a better score. Record the resulting file hash and evaluator feedback before comparing versions.
