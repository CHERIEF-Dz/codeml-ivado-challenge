"""Best-bet hidden-reference submission: remove learned geographic contributions.

This is a transparent modeling assumption, not measured hidden-reference accuracy.
Economic features can still be regional proxies; no causal fairness guarantee is made.
"""
import json

import numpy as np
import pandas as pd

from model_improved import ROOT, allocation, features, group, models


def neutral_scores(model, df):
    prep, classifier = model.steps[0][1], model.steps[-1][1]
    encoded = prep.transform(features(df))
    names = prep.get_feature_names_out()
    geographic = np.array([name.startswith(('cat__region_administrative_', 'cat__code_postal_3_'))
                           or name in {'num__distance_domicile_campus_km', 'num__log_distance'}
                           for name in names])
    return classifier.decision_function(encoded) - encoded[:, geographic] @ classifier.coef_[0, geographic]


def main():
    (ROOT / 'results_improved').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / 'data/donnees_demandes.csv')
    candidates = pd.read_csv(ROOT / 'data/candidats_evaluation.csv')
    model = dict(models())['logistic_C1.0']
    model.fit(features(df), df.decision_octroi)
    scores = neutral_scores(model, candidates)
    pred = allocation(scores, group(candidates), rate=0.40)
    submission = pd.DataFrame({'id_candidat':candidates.id_candidat, 'decision_octroi':pred})
    assert len(submission) == 4000 and submission.id_candidat.is_unique
    assert pred.sum() == 1600
    submission.to_csv(ROOT / 'predictions.csv', index=False)
    submission.to_csv(ROOT / 'results_improved/predictions_merit.csv', index=False)
    audit = {'strategy':'logistic_C1.0 with geographic score contributions removed; top 40%',
             'hidden_accuracy':None, 'selected':int(pred.sum()), 'selection_rate':float(pred.mean()),
             'centre_rate':float(pred[~group(candidates)].mean()),
             'remote_rate':float(pred[group(candidates)].mean())}
    (ROOT / 'results_improved/merit_submission.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
