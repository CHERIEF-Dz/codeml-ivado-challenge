# Model improvement report

## Outcome and submission to test

Branch: `improve-model-accuracy`.

The best measured historical-label model achieved **88.80% held-out accuracy**, compared with **87.55%** for the baseline on the same test set. The selected budget/fairness policy achieved **88.40%**. **95% accuracy was not achieved.** The utility model achieved 95.70% ROC AUC; AUC measures ranking quality and must not be presented as accuracy.

The challenge explicitly scores against an independent hidden reference, not the historical committee decisions. Consequently, the submission selected as our **best bet for the hidden reference** is a geographically neutralized logistic ranking. Its hidden-reference accuracy is unknown. The user will evaluate it externally and provide feedback.

**Test `predictions.csv` in this directory.** It contains exactly 4,000 candidate IDs and 1,600 positive decisions (40%). The file is also saved as `results_improved/predictions_merit.csv`. The historically validated alternative is `results_improved/predictions.csv`; the prior submission is preserved as `results_improved/previous_predictions.csv`.

## What changed and why

### 1. Comparable, reproducible model search

`model_improved.py` evaluates 14 configurations: four regularization strengths for logistic regression; the baseline forest and three tuned forests; and six histogram gradient-boosting configurations. All use the same 6,000 training, 2,000 validation, and 2,000 test rows. Splits are stratified jointly by regional group and historical decision, using seed 42. Identifiers and labels are explicitly excluded from features. Training data has no missing values, duplicate IDs, or duplicate feature rows.

Numeric scaling enables logistic regression to learn a smooth additive decision boundary. Log-income and log-distance terms allow diminishing returns instead of assuming strictly linear monetary/distance effects. Categories are encoded in a fitted preprocessing pipeline with unknown-category handling. The best historical-label results came from logistic regression, rather than larger forests or boosting.

### 2. Accuracy-aware allocation policy selection

The former pipeline selected the smallest historical equal-opportunity gap regardless of accuracy and had a fallback referencing an undefined `best_row`. The replacement selects the highest validation accuracy subject to a historical equal-opportunity gap of at most 0.05. Ties favor a lower gap, then an allocation rate nearer 40%.

The search evaluates 5 allocation rates from 36% through 44% and 31 regional probability adjustments per model: 2,170 policies total. Selected policy: logistic regression with C=1, 40% allocation, and a +0.135 remote-group probability adjustment. These choices were frozen before test evaluation. The adjusted values are ranking scores, not calibrated probabilities.

This is an empirical tradeoff against historical decisions. It does not establish fairness against the hidden reference, and the validation tolerance is not a guarantee on future data.

### 3. Exact budgets and deterministic predictions

Ranking with a stable sort selects an exact number of candidates, replacing randomized Fairlearn prediction and a budget assertion that merely detected violations. For 4,000 applicants, the tested budget boundaries select exactly 1,440 and 1,760 people. The final best-bet submission selects 1,600. Equal scores are resolved in input order.

### 4. Full-data refitting

After configuration selection and held-out evaluation, the selected model is cloned and refitted using all 10,000 historical records for submission. The test metrics describe the earlier training-only model, not an evaluation of the refitted model against its own training labels.

### 5. Best bet for the independent reference

`model_merit.py` fits the selected C=1 logistic model, then subtracts the learned contributions of region, postal code, raw distance, and log-distance from its decision score. It retains academic R score, study program, income, working hours, and first-generation status. The top 40% of the resulting ranking receive awards. `model_corrige.py` now runs this submission strategy. The old EG script is preserved as `model_historical_eg.py` for provenance, with its original behavior and limitations.

The rationale is that geographic penalties in a model trained to reproduce the audited committee should not determine the independent merit ranking. Retaining income and working hours assumes they can contain relevant economic or effort information even though they also correlate with region. This is a modeling assumption, not an identified causal correction. Other retained features can still carry regional information, and the removed geographic effects cannot be proven entirely discriminatory from these data.

The best-bet submission selects **42.20%** of centre applicants and **36.79%** of remote applicants, a 5.41 percentage-point selection gap. The prior submission selected 45.87% and 29.05%, a 16.81-point gap. A smaller selection gap is observable; improved hidden-reference accuracy and equal opportunity remain unverified. We did not impose equal group selection rates.

## Held-out benchmark results

All metrics below use historical `decision_octroi`, on the identical untouched 2,000-row test set. Model and policy selection used validation data only.

| Model/policy | Accuracy | F1 | ROC AUC | Historical EOP gap | Selection rate |
|---|---:|---:|---:|---:|---:|
| Baseline random forest | 87.55% | 84.29% | 94.86% | 0.94 points | 39.35% |
| Utility logistic, C=10, threshold 0.5 | 88.80% | 86.14% | 95.70% | 1.98 points | 40.90% |
| Selected logistic, C=1, constrained ranking | 88.40% | 85.48% | 95.71% | 3.42 points | 40.00% |

The constrained model's AUC is calculated from the unadjusted model probabilities. Its allocation metrics use the adjusted ranking. The selected policy improved historical accuracy by 0.85 points but had a larger historical EOP gap than the test baseline. It met the 5-point tolerance. This distinction matters: it should not be reported as an across-the-board fairness improvement.

The merit submission is a separate, explicitly assumption-based hidden-reference candidate. The table is not its measured accuracy. A single split and extensive validation search leave selection uncertainty; no statistical significance or hard performance ceiling is claimed. The original changelog's 90.4% result is not comparable evidence because it was not reproduced on this identical split.

## Verification and reproducibility

Seven targeted tests cover budget boundaries, tie determinism, regional adjustment, invalid scores/rates, manually checked fairness metrics, feature leakage prevention, and invariance of merit scores when only geographic fields change. All passed. Submission checks verify candidate IDs, row count, binary values, and exact budget. The benchmark is recorded in `results_improved/metrics.json`, all validation policies in `validation_pareto.csv`, and the plot in `pareto.png`.

Exact installed dependencies are recorded in `requirements-benchmark.txt` (Python 3.13.3).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-benchmark.txt
python model_improved.py       # Reproduce historical benchmark; writes results_improved/
python model_corrige.py        # Generate best-bet predictions.csv
python -m unittest discover -p 'test_model_improved.py' -v
```

Run these commands from this directory. For our test environment, `MPLCONFIGDIR` was set to a temporary writable directory.

## Next decision after external feedback

Provide the hidden-reference utility/accuracy, equity score, and total score for the merit submission. If feasible, evaluate the historically validated alternative too. Those results can distinguish whether geographic neutralization helps the actual target. Repeated adaptation to scorer feedback can overfit the evaluation set, so track versions and use a separate final evaluation if available. Reaching 95% cannot be guaranteed from the current biased training labels; independent reference labels or an explicit definition of the reference would provide the strongest additional evidence.
