# DBSCAN, HDBSCAN and Isolation Forest audit

## Conclusion

These methods are useful for auditing proxy variables and unusual profiles. **They did not provide meaningful evidence of an accuracy improvement.** The externally scored 94.6% submission remains unchanged.

The audit does not classify unusual applicants as undeserving or their records as invalid. Clustering finds similarity and density, not the independent reference labels used by the judges.

## Implementation

`audit_density.py` fits transformations and detectors on the 10,000 historical applications and then audits the 4,000 evaluation applicants. Its four features are R score, weekly working hours, log1p household income, and log1p home-to-campus distance. They are standardized using historical data only. IDs, labels, explicit region/postal codes, program and first-generation status are excluded from distances; those fields may still be used to describe the resulting groups.

The primary DBSCAN configuration was specified before inspecting label performance: `min_samples=20`, with epsilon equal to the 90th percentile of historical 20-neighbor distances, counting the point itself consistently with DBSCAN. This gives **eps=0.777418** in standardized coordinates. This is a diagnostic heuristic, not a certified optimal setting.

DBSCAN has no native prediction method. Evaluation applicants are assigned to the nearest historical core point when its distance is within epsilon; otherwise they are marked unclustered. The script also records ambiguous border points that touch multiple core clusters. None occurred in the primary four-dimensional configuration.

Isolation Forest uses 300 trees, seed 41, and `contamination=0.05`. The 5% is a chosen historical review threshold, not an estimate that 5% of records are erroneous. HDBSCAN is fitted to historical data only, with minimum cluster sizes 100, 250 and 500 and `min_samples=20`.

## Actual audit results

| Method | Population | Result |
|---|---|---|
| DBSCAN | 10,000 historical | One large cluster of 9,796; 204 unclustered |
| DBSCAN transfer | 4,000 evaluation | 108 flagged: 2.70% |
| Isolation Forest | 10,000 historical | 500 flagged: 5%, set by the review threshold |
| Isolation Forest | 4,000 evaluation | 216 flagged: 5.40% |
| HDBSCAN, four features | 10,000 historical | Two clusters; 6,147 unclustered: 61.47% |

Of the 108 DBSCAN-flagged evaluation applicants, **61 currently receive awards**. Only **4** are within 0.5 R-score-equivalent units of the effort-model decision cutoff. Isolation Forest flags 121 current awardees; only 9 of its flagged applicants are near the cutoff. There are 509 applicants in that cutoff band overall. The detectors agree on 94 flagged evaluation applicants.

This means density flags mostly identify feature-distribution tails, including very strong academic applicants, rather than locating the difficult decisions near the grant boundary. We do not know whether any particular flagged prediction is wrong because the reference labels are unavailable. The score-cutoff band is a diagnostic convention, not a calibrated confidence interval.

### Regional representation

| Evaluation flag rate | Centres | Remote regions |
|---|---:|---:|
| DBSCAN |2.11%|3.56%|
| Isolation Forest |4.22%|7.13%|

Automatically deleting flagged training rows would also change regional representation. Isolation Forest removes 7.05% of remote training applicants versus 3.63% of centre applicants. DBSCAN removes 2.475% versus 1.75%. These differences do not alone prove discrimination, but they argue against treating density flags as automatic exclusion rules.

HDBSCAN provides direct proxy-audit evidence:

| Historical cluster | Applicants | Regional composition | Historical award rate |
|---|---:|---|---:|
| A |1,203|100% remote|16.87%|
| B |2,650|99.36% centres|48.04%|

Region and postal code were excluded from clustering, yet numerical proxies largely recovered geography in these dense groups. This supports the README's warning about indirect regional information. Their different award rates do not establish unfairness on their own because academic and economic profiles also differ. Most historical applicants remain outside those HDBSCAN clusters.

### Parameter and feature sensitivity

Eighteen DBSCAN configurations were evaluated: two feature views, three `min_samples` choices (10, 20, 40), and three neighbor-distance percentiles (80th, 90th, 95th). All nine four-feature configurations produced one main cluster, but evaluation noise rates ranged from 0.525% to 9.725%. Thus the flagged count is sensitive to the density threshold.

Clustering only R score and working hours produced many apparent clusters. These mostly align with integer-valued working hours. For example, the primary two-feature DBSCAN radius is smaller than the standardized distance represented by a one-hour step, preventing direct connections across adjacent hour values. HDBSCAN similarly produced 13 two-feature clusters at minimum size 250. Perturbing hours by less than half an hour solely as a diagnostic destroyed that structure: across three seeds, it became 0–2 clusters with 87–100% unclustered. Original data were not changed. These fragile clusters are not persuasive evidence for distinct merit groups.

## Does using the detectors improve the model?

We ran four variants on the **same five region-group/label-stratified folds** (seed 41). Every scaler, detector, epsilon estimate and training-row exclusion was fitted solely inside the relevant training fold. All held-out applicants remained in evaluation.

The classifier is the same C=1 logistic pipeline used to estimate the effort weight. The extra-cluster variant adds one-hot cluster membership to its inputs. DBSCAN finds one main cluster in each fold, so that feature largely represents a noise flag.

| Historical-label experiment | Accuracy | Macro F1 | Change in correctly predicted historical rows |
|---|---:|---:|---:|
| Baseline logistic |88.72%|88.209%|—|
| Remove Isolation Forest training outliers |88.73%|88.221%|+1 of 10,000|
| Remove DBSCAN training outliers |88.67%|88.159%|−5 of 10,000|
| Add DBSCAN membership as a feature |88.68%|88.167%|−4 of 10,000|

Isolation Forest corrected 17 historical predictions and broke 16. That one-net-example difference does not support a reliable gain. These are metrics against biased committee decisions, **not the independent reference**. They must not be compared directly to the 94.6% external score.

After full-data refitting, retaining only the learned R/work-hours contributions as in the current allocation model:

| Training variant | Work-hours coefficient | Evaluation decisions changed from 94.6% submission |
|---|---:|---:|
| Baseline |0.14381743|0|
| Isolation Forest trimming |0.14906503|2|
| DBSCAN trimming |0.14435741|0|
| DBSCAN feature |0.14364325|0|

The DBSCAN variants yield **exactly the same 4,000 final award decisions**. The Isolation Forest variant differs on two candidates, so its accuracy can change by at most ±0.05 percentage points if all 4,000 are scored. The direction is unknown; it was not submitted externally. No new prediction file replaces the tested submission.

## How to use this audit

Use density flags to review unusual data, check regional representation, and monitor incoming-data drift. Before correcting or excluding any record, look for actual evidence of a data error. Track whether flagged groups have worse outcomes once independently reviewed reference labels become available. Keep the current merit ranking for allocation unless a change earns an independently measured improvement.

An applicant-level review table is in `audit_results/candidate_density_audit.csv`. The plot in `audit_results/density_audit.png` shows the R/work-hours projection; flagging itself uses all four audit features. The notebook `audit_rapport.ipynb` includes the new results and interpretation.

Reproduce from the challenge directory:

```bash
python audit_density.py
python research/audit_hdbscan.py
python research/validate_density.py
python -m unittest discover -p 'test_density_audit.py' -v
```

Artifacts include the parameter sweep, flag rates by group, historical cluster profiles, full fold-by-fold validation metrics and candidate audit table. Three targeted tests verify feature exclusion, nearest-core assignment including ambiguous borders, and the no-core case. Original README and scored submission bytes are verified unchanged.

Method references: [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html), and [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html). These document the algorithms; the numerical findings above come from the repository data and scripts.
