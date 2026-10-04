"""Next external-test candidate: academic performance plus working effort.

The hours weight is estimated from historical decisions, controlling for
geographic and income effects during fitting. Only R score and hours enter the
final ranking. Whether effort belongs in the hidden reference is unverified.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model_improved import ROOT, NUM, features, group, models


def fit_effort_weight(training):
    model = next(model for name, model in models() if name == 'logistic_C1.0')
    model.fit(features(training), training.decision_octroi)
    scale = model[0].named_transformers_['num'].scale_
    slopes = model[1].coef_[0][:len(scale)] / scale
    r_slope = slopes[NUM.index('cote_r_equivalent')]
    hours_slope = slopes[NUM.index('heures_travail_semaine')]
    if not np.isfinite([r_slope, hours_slope]).all() or r_slope <= 0 or hours_slope < 0:
        raise ValueError('Data do not support a positive academic/effort ranking.')
    return float(hours_slope / r_slope)


def effort_predictions(candidates, hours_weight):
    required = ['id_candidat', 'cote_r_equivalent', 'heures_travail_semaine']
    if candidates[required].isna().any().any() or not candidates.id_candidat.is_unique:
        raise ValueError('Unique IDs and complete R scores/work hours are required.')
    values = candidates[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(hours_weight) or hours_weight < 0:
        raise ValueError('Inputs must be finite and the hours weight nonnegative.')
    if (values[:, 1] < 0).any():
        raise ValueError('Working hours must be nonnegative.')
    scores = values[:, 0] + hours_weight * values[:, 1]
    order = np.lexsort((candidates.id_candidat.to_numpy(dtype=str), -scores))
    predictions = np.zeros(len(candidates), dtype=int)
    predictions[order[:round(0.40 * len(candidates))]] = 1
    return predictions


def main():
    training = pd.read_csv(ROOT / 'data/donnees_demandes.csv')
    candidates = pd.read_csv(ROOT / 'data/candidats_evaluation.csv')
    strata = group(training).astype(str) + '_' + training.decision_octroi.astype(str).to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    fold_weights = [fit_effort_weight(training.iloc[train]) for train, _ in folds.split(training, strata)]
    weight = fit_effort_weight(training)
    pred = effort_predictions(candidates, weight)
    if len(candidates) != 4000 or pred.sum() != 1600:
        raise ValueError('Expected exactly 4,000 applicants and 1,600 awards.')
    output = ROOT / 'results_effort'
    output.mkdir(exist_ok=True)
    submission = pd.DataFrame({'id_candidat': candidates.id_candidat, 'decision_octroi': pred})
    submission.to_csv(output / 'predictions.csv', index=False)
    submission.to_csv(ROOT / 'predictions.csv', index=False)
    academic = pd.read_csv(ROOT / 'results_academic/predictions.csv').set_index('id_candidat')
    previous = academic.loc[candidates.id_candidat, 'decision_octroi'].to_numpy()
    remote = group(candidates)
    audit = {
        'strategy': 'R score plus learned working-hours contribution; top 40%',
        'hours_weight_in_r_score_units': weight,
        'five_fold_weight_estimates': fold_weights,
        'fold_seed': 41,
        'selection_rate': float(pred.mean()), 'selected': int(pred.sum()), 'rows': len(pred),
        'centre_selection_rate': float(pred[~remote].mean()),
        'remote_selection_rate': float(pred[remote].mean()),
        'changed_from_academic_92_23': int(np.sum(pred != previous)),
        'new_awards': int(np.sum((pred == 1) & (previous == 0))),
        'removed_awards': int(np.sum((pred == 0) & (previous == 1))),
        'external_accuracy': None, 'external_f1_macro': None,
        'reference_warning': 'Historical weight stability does not validate hidden-reference accuracy or fairness.',
        'sha256': hashlib.sha256((ROOT / 'predictions.csv').read_bytes()).hexdigest(),
    }
    # External feedback applies only to the exact scored bytes, never a new fit.
    feedback_path = ROOT / 'external_evaluation_log.json'
    if feedback_path.exists():
        for feedback in json.loads(feedback_path.read_text()):
            if feedback.get('sha256') == audit['sha256']:
                audit['external_accuracy'] = feedback.get('reported_accuracy')
                audit['external_f1_macro'] = feedback.get('f1_macro')
                audit['external_accuracy_source'] = feedback.get('source')
    (output / 'submission_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
