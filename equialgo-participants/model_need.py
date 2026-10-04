"""Unscored next trial: academic performance, working effort, and financial need.

Coefficients are an exploratory hypothesis informed by aggregate external scores,
not a validated reconstruction of the hidden reference. Never replaces the best
scored predictions.csv. See research_report.md for assumptions and sensitivity.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from model_effort import effort_predictions
from model_improved import ROOT, group

HOURS_WEIGHT = 0.135
NEED_WEIGHT = 0.67


def need_predictions(candidates):
    income = candidates.revenu_familial_estime.to_numpy(dtype=float)
    if not np.isfinite(income).all() or (income <= 0).any():
        raise ValueError('Household incomes must be finite and positive.')
    adjusted = candidates.copy()
    # The denominator is a constant and does not affect the ranking.
    adjusted['cote_r_equivalent'] = (candidates.cote_r_equivalent
        - NEED_WEIGHT * np.log(income / 60000.0))
    return effort_predictions(adjusted, HOURS_WEIGHT)


def main():
    candidates = pd.read_csv(ROOT / 'data/candidats_evaluation.csv')
    pred = need_predictions(candidates)
    if len(pred) != 4000 or pred.sum() != 1600:
        raise ValueError('Expected 4,000 applicants and 1,600 awards.')
    output = ROOT / 'results_need'
    output.mkdir(exist_ok=True)
    submission = pd.DataFrame({'id_candidat': candidates.id_candidat, 'decision_octroi': pred})
    submission.to_csv(output / 'predictions.csv', index=False)
    submission.to_csv(ROOT / 'predictions_next_test.csv', index=False)
    best = pd.read_csv(ROOT / 'results_effort/predictions.csv').set_index('id_candidat')
    old = best.loc[candidates.id_candidat, 'decision_octroi'].to_numpy()
    remote = group(candidates)
    audit = {'status':'Unscored experiment; best tested file is unchanged',
        'score':'R + 0.135 * weekly_hours - 0.67 * log(income / 60000)',
        'rows':len(pred), 'selected':int(pred.sum()),
        'changed_from_effort_94_6':int(np.sum(old != pred)),
        'centre_selection_rate':float(pred[~remote].mean()),
        'remote_selection_rate':float(pred[remote].mean()),
        'external_accuracy':None,
        'sha256':hashlib.sha256((output/'predictions.csv').read_bytes()).hexdigest()}
    (output/'submission_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
