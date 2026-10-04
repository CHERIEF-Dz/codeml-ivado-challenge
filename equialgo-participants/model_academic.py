"""Academic-only candidate based on the README's explicit merit signal.

This is a new hypothesis for the independent reference, not a measured accuracy
improvement. Region and socioeconomic proxies do not enter the ranking.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REMOTE = {'Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine'}


def academic_predictions(candidates):
    required = ['id_candidat', 'cote_r_equivalent']
    if candidates[required].isna().any().any():
        raise ValueError('Candidate IDs and R scores must be present.')
    if not candidates.id_candidat.is_unique:
        raise ValueError('Candidate IDs must be unique.')
    scores = candidates.cote_r_equivalent.to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('R scores must be finite.')
    # IDs only resolve exact score ties; their values are never predictive features.
    order = np.lexsort((candidates.id_candidat.to_numpy(dtype=str), -scores))
    predictions = np.zeros(len(candidates), dtype=int)
    predictions[order[:round(0.40 * len(candidates))]] = 1
    return predictions


def main():
    candidates = pd.read_csv(ROOT / 'data/candidats_evaluation.csv')
    pred = academic_predictions(candidates)
    if len(candidates) != 4000 or pred.sum() != 1600:
        raise ValueError('Expected 4,000 applicants and exactly 1,600 awards.')
    submission = pd.DataFrame({'id_candidat': candidates.id_candidat, 'decision_octroi': pred})
    output_dir = ROOT / 'results_academic'
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / 'predictions.csv', index=False)
    submission.to_csv(ROOT / 'predictions.csv', index=False)
    remote = candidates.region_administrative.isin(REMOTE).to_numpy()
    previous = pd.read_csv(ROOT / 'results_improved/predictions_merit.csv').set_index('id_candidat')
    prior = previous.loc[candidates.id_candidat, 'decision_octroi'].to_numpy()
    audit = {
        'strategy': 'Academic R score only; global top 40%; candidate ID breaks exact ties',
        'reason': 'README names R score as merit and flags income, hours, distance, and postal code as regional proxies.',
        'hidden_reference_accuracy': None,
        'previous_external_feedback': {'reported_value': 0.90, 'metric': 'not clarified by user'},
        'rows': len(submission), 'selected': int(pred.sum()), 'selection_rate': float(pred.mean()),
        'centre_selection_rate': float(pred[~remote].mean()),
        'remote_selection_rate': float(pred[remote].mean()),
        'changed_from_previous_attempt': int(np.sum(pred != prior)),
        'minimum_selected_r_score': float(candidates.loc[pred == 1, 'cote_r_equivalent'].min()),
        'sha256': hashlib.sha256((ROOT / 'predictions.csv').read_bytes()).hexdigest(),
    }
    (output_dir / 'submission_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
