# Active family-income submission

On branch **`feature/family-income-priority`**, submit **`predictions.csv`** to test the new family-income option. It ranks applicants using:

```
R score + 0.14381742782570264 * weekly work hours
        - 0.67 * ln(family income / 60000)
```

The user selected a modest priority for lower-income applicants when merit is close. The work-hours coefficient is unchanged from the tested 94.6% model. The new rule changes 106 decisions (53 entrants and 53 departures) and selects exactly 1,600 of 4,000 candidates. Its accuracy and macro F1 are **not yet measured**.

The 94.6% submission is preserved at `results_effort/predictions.csv` and on `improve-model-accuracy` at commit `bb51395`. The current file is also archived at `results_family_income/predictions.csv`. See `active_submission.json` for its hash and `best_submission.json` for the best externally measured artifact.

The full rationale, income-strength interpretation, allocation comparisons, validation, and reproduction commands are in [family_income_report.md](family_income_report.md). The original README is unchanged. Previous research and density-audit reports describe the earlier scored model and remain available as supporting evidence.
