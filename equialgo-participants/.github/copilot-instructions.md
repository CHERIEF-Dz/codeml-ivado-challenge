# Copilot Instructions for ÉquiAlgo Challenge

## General Constraints
- **Data Integrity**: NEVER modify, delete, or overwrite any files in the `data/` directory. These are read-only source datasets.
- **Budget Constraint**: Any final prediction model must maintain a global grant rate between 36% and 44% on the evaluation set.
- **Fairness Target**: Focus on reducing the parity gap between 'Centre' and 'Eloignee' regions.

## Workflow Preferences
- Always maintain consistency with the `baseline_model.ipynb` for data preprocessing (e.g., region grouping).
- When analyzing proxies, prioritize both linear correlation and and non-linear dependencies (e.g., Random Forest feature importance).
- Ensure all generated CSVs follow the exact format: `id_candidat,decision_octroi`.
