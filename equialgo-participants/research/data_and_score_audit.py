"""Independent analysis using public features and aggregate user feedback only.

Run: python equialgo-independent-analysis.py --project /path/to/equialgo-participants
All latent-noise fits are illustrative assumptions, NOT held-out accuracy estimates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, root
from scipy.special import ndtr, expit
from scipy.stats import ks_2samp

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results'/'data_and_score_audit.json')
args = parser.parse_args()
tr = pd.read_csv(args.project / 'data/donnees_demandes.csv')
ev = pd.read_csv(args.project / 'data/candidats_evaluation.csv')

def read_submission(path):
    df = pd.read_csv(args.project / path).set_index('id_candidat')
    return df.loc[ev.id_candidat, 'decision_octroi'].to_numpy()

academic = read_submission('results_academic/predictions.csv')
effort = read_submission('results_effort/predictions.csv')
r = ev.cote_r_equivalent.to_numpy()
h = ev.heures_travail_semaine.to_numpy()
num = list(ev.select_dtypes('number').columns)
report = {
    'warning': 'Only reported external metrics are measured hidden performance. Latent-noise fits assume an unobserved additive R-plus-hours target and independent noise; they neither identify the true reference nor establish an accuracy ceiling.',
    'data_audit': {
        'training_rows': len(tr), 'evaluation_rows': len(ev),
        'missing_training_cells': int(tr.isna().sum().sum()),
        'missing_evaluation_cells': int(ev.isna().sum().sum()),
        'overlapping_ids': len(set(tr.id_candidat) & set(ev.id_candidat)),
        'duplicate_training_feature_rows': int(tr.drop(columns=['id_candidat', 'decision_octroi']).duplicated().sum()),
        'duplicate_evaluation_feature_rows': int(ev.drop(columns=['id_candidat']).duplicated().sum()),
        'identical_features_across_train_evaluation': len(pd.merge(tr, ev, on=list(ev.columns[1:]))),
        'two_sample_ks': {c: {'statistic': float(ks_2samp(tr[c], ev[c]).statistic), 'p_value': float(ks_2samp(tr[c], ev[c]).pvalue)} for c in num},
    },
    'reported_external': {'academic_accuracy': 0.9223, 'academic_macro_f1': 0.919, 'effort_accuracy': 0.946},
    'macro_f1_rounding_prevalence_ranges': {},
    'latent_noise_sensitivity': [],
}

N = len(ev)
k = int(academic.sum())
for decimals in [1, 2]:
    feasible = []
    for actual_positive in range(N + 1):
        for errors in [310, 311, 312]:
            tp = (actual_positive + k - errors) / 2
            tn = N - errors - tp
            if tp % 1 or tp < 0 or tp > min(actual_positive, k) or tn < 0:
                continue
            acc = (N - errors) / N
            macro = .5 * (2 * tp / (actual_positive + k) + 2 * tn / (2 * N - actual_positive - k))
            if abs(acc - .9223) < .00005001 and abs(macro - .919) < .5 * 10 ** (-decimals - 2):
                feasible.append(actual_positive)
    report['macro_f1_rounding_prevalence_ranges'][str(decimals)] = {
        'min_actual_positive': min(feasible), 'max_actual_positive': max(feasible),
        'min_rate': min(feasible) / N, 'max_rate': max(feasible) / N,
        'accuracy_error_count': 311,
        'assumption': 'accuracy and binary macro-F1 computed over exactly4000 same rows; conventional nearest rounding',
    }

def latent_probability(alpha, sigma, cdf):
    scores = r + alpha * h
    threshold = brentq(lambda t: cdf((scores - t) / sigma).mean() - .40,
                       scores.min() - 10 * sigma, scores.max() + 10 * sigma)
    return cdf((scores - threshold) / sigma)

for family, cdf in [('normal', ndtr), ('logistic', expit)]:
    for effort_accuracy in [.94575, .946, .94625]:
        def residual(x):
            p = latent_probability(x[0], np.exp(x[1]), cdf)
            return [np.mean(np.where(academic, p, 1 - p)) - .92225,
                    np.mean(np.where(effort, p, 1 - p)) - effort_accuracy]
        sol = root(residual, [.15, np.log(.5)])
        if not sol.success or np.max(np.abs(residual(sol.x))) > 1e-7:
            raise RuntimeError('Sensitivity fitting did not converge.')
        alpha, sigma = sol.x[0], np.exp(sol.x[1])
        p = latent_probability(alpha, sigma, cdf)
        trial = {'family': family, 'effort_accuracy_assumption': effort_accuracy,
                 'hours_coefficient': float(alpha), 'noise_scale': float(sigma), 'policies': []}
        for weight in [.10, .12, .14381742782570264, .15, .16, .18, .20, .25]:
            pred = np.zeros(N, dtype=int)
            order = np.lexsort((ev.id_candidat.to_numpy(dtype=str), -(r + weight * h)))
            pred[order[:1600]] = 1
            trial['policies'].append({'hours_coefficient': weight,
                'illustrative_expected_accuracy': float(np.mean(np.where(pred, p, 1 - p))),
                'changed_from_effort': int(np.sum(pred != effort))})
        report['latent_noise_sensitivity'].append(trial)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(args.output)
