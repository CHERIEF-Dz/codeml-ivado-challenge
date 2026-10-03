"""Reproducible utility/fairness benchmark. Labels are historical, not ground truth."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
REMOTE = {'Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine'}
NUM = ['cote_r_equivalent', 'revenu_familial_estime', 'heures_travail_semaine',
       'distance_domicile_campus_km', 'premiere_generation_universitaire']
CAT = ['programme_etudes', 'region_administrative', 'code_postal_3']


def features(df):
    x = df[NUM + CAT].copy()
    # Smooth diminishing returns are plausible in income/distance effects.
    x['log_income'] = np.log1p(x['revenu_familial_estime'])
    x['log_distance'] = np.log1p(x['distance_domicile_campus_km'])
    return x


def group(df):
    return df['region_administrative'].isin(REMOTE).to_numpy()


def allocation(scores, remote, rate=0.40, remote_bonus=0.0):
    """Exact budget; stable input-order tie breaking; no labels used."""
    if not 0.36 <= rate <= 0.44:
        raise ValueError('Allocation rate must be between 0.36 and 0.44.')
    if len(scores) != len(remote) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and match the group vector.')
    adjusted = np.asarray(scores) + remote_bonus * np.asarray(remote)
    selected = np.argsort(-adjusted, kind='stable')[:round(rate * len(scores))]
    pred = np.zeros(len(scores), dtype=int)
    pred[selected] = 1
    return pred


def metrics(y, pred, remote, scores=None):
    y = np.asarray(y)
    out = {'accuracy': float(accuracy_score(y, pred)), 'f1': float(f1_score(y, pred)),
           'selection_rate': float(np.mean(pred))}
    tprs, rates = [], []
    for name, mask in [('centre', ~remote), ('remote', remote)]:
        positive = mask & (y == 1)
        tpr = float(np.mean(pred[positive])) if positive.any() else None
        rate = float(np.mean(pred[mask]))
        out[name] = {'count': int(mask.sum()), 'accuracy': float(accuracy_score(y[mask], pred[mask])),
                     'tpr': tpr, 'selection_rate': rate}
        tprs.append(tpr)
        rates.append(rate)
    out['eop_gap'] = abs(tprs[0] - tprs[1]) if None not in tprs else None
    out['selection_gap'] = abs(rates[0] - rates[1])
    if scores is not None:
        out['auc'] = float(roc_auc_score(y, scores))
        out['log_loss'] = float(log_loss(y, scores))
    return out


def models():
    numeric = NUM + ['log_income', 'log_distance']
    for c in [0.01, 0.1, 1.0, 10.0]:
        prep = ColumnTransformer([('num', StandardScaler(), numeric),
                                  ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])
        yield f'logistic_C{c}', make_pipeline(prep, LogisticRegression(C=c, max_iter=2000, random_state=42))
    prep = ColumnTransformer([('num', 'passthrough', NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])
    yield 'baseline_rf', make_pipeline(prep, RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1))
    for leaf in [2, 5, 10]:
        yield f'rf_leaf{leaf}', make_pipeline(prep, RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf, max_features=0.8, random_state=42, n_jobs=-1))
    for leaves in [7, 15, 31]:
        for regularization in [1.0, 10.0]:
            yield f'hgb_leaves{leaves}_l2{regularization}', make_pipeline(prep,
                HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=leaves,
                l2_regularization=regularization, early_stopping=True, random_state=42))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-eop-gap', type=float, default=0.05)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_improved')
    args = parser.parse_args()
    if not 0 <= args.max_eop_gap <= 1:
        parser.error('--max-eop-gap must be between 0 and 1')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / 'data/donnees_demandes.csv')
    candidates = pd.read_csv(ROOT / 'data/candidats_evaluation.csv')
    if df.isna().any().any() or candidates.isna().any().any():
        raise ValueError('Missing data: add an explicit imputation strategy before training.')
    strata = group(df).astype(str) + '_' + df.decision_octroi.astype(str).to_numpy()
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=42)
    rest_strata = group(rest).astype(str) + '_' + rest.decision_octroi.astype(str).to_numpy()
    val, test = train_test_split(rest, test_size=0.5, stratify=rest_strata, random_state=42)
    xtrain, xval, xtest = map(features, [train, val, test])
    trials, fitted, ordinary = [], {}, []
    for name, model in models():
        model.fit(xtrain, train.decision_octroi)
        fitted[name] = model
        score = model.predict_proba(xval)[:, 1]
        standard = metrics(val.decision_octroi, (score >= 0.5).astype(int), group(val), score)
        ordinary.append({'model': name, **standard})
        # Tune allocation policy on validation only, including regional sensitivity.
        for rate in [0.36, 0.38, 0.40, 0.42, 0.44]:
            for bonus in np.linspace(-0.15, 0.30, 31):
                pred = allocation(score, group(val), rate, bonus)
                m = metrics(val.decision_octroi, pred, group(val))
                trials.append({'model': name, 'rate': rate, 'remote_bonus': float(bonus), **m})
        print(name, f'validation accuracy={standard["accuracy"]:.4f}', flush=True)
    table = pd.DataFrame([{k:v for k,v in r.items() if k not in ['centre','remote']} for r in trials])
    table.to_csv(args.output_dir / 'validation_pareto.csv', index=False)
    admissible = [r for r in trials if r['eop_gap'] <= args.max_eop_gap]
    if not admissible:
        raise RuntimeError('No validation policy satisfies the requested fairness tolerance.')
    chosen = max(admissible, key=lambda r: (r['accuracy'], -r['eop_gap'], -abs(r['rate']-0.40)))
    utility = max(ordinary, key=lambda r: r['accuracy'])
    baseline_score = fitted['baseline_rf'].predict_proba(xtest)[:, 1]
    utility_score = fitted[utility['model']].predict_proba(xtest)[:, 1]
    score = fitted[chosen['model']].predict_proba(xtest)[:, 1]
    test_pred = allocation(score, group(test), chosen['rate'], chosen['remote_bonus'])
    results = {'label_warning': 'Accuracy and EOP are measured against biased historical committee decisions; hidden reference unavailable.',
        'split': {'seed':42, 'train':len(train), 'validation':len(val), 'test':len(test)},
        'versions': {'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__},
        'fairness_tolerance':args.max_eop_gap, 'validation_models':ordinary, 'selected_policy':chosen,
        'utility_model':utility['model'],
        'test_baseline':metrics(test.decision_octroi, (baseline_score>=0.5).astype(int), group(test), baseline_score),
        'test_utility':metrics(test.decision_octroi, (utility_score>=0.5).astype(int), group(test), utility_score),
        'test_selected_policy':metrics(test.decision_octroi, test_pred, group(test), score)}
    # No selection based on test results. Refit the frozen model/policy on all data.
    final = clone(fitted[chosen['model']])
    final.fit(features(df), df.decision_octroi)
    candidate_score = final.predict_proba(features(candidates))[:, 1]
    pred = allocation(candidate_score, group(candidates), chosen['rate'], chosen['remote_bonus'])
    submission = pd.DataFrame({'id_candidat':candidates.id_candidat, 'decision_octroi':pred})
    assert len(submission)==4000 and submission.id_candidat.is_unique
    assert set(submission.decision_octroi.unique()) <= {0,1}
    assert 0.36 <= pred.mean() <= 0.44
    submission.to_csv(args.output_dir / 'predictions.csv', index=False)
    results['submission'] = {'count':len(pred), 'selected':int(pred.sum()), 'rate':float(pred.mean()),
        'centre_rate':float(pred[~group(candidates)].mean()), 'remote_rate':float(pred[group(candidates)].mean())}
    (args.output_dir / 'metrics.json').write_text(json.dumps(results, indent=2))
    plt.figure(figsize=(9,6))
    plt.scatter(table.eop_gap, table.accuracy, c=table.rate, s=8, alpha=0.4)
    plt.scatter(chosen['eop_gap'], chosen['accuracy'], marker='*', s=160, color='red')
    plt.axvline(args.max_eop_gap, linestyle='--', color='black')
    plt.xlabel('Historical-label equal opportunity gap')
    plt.ylabel('Historical-label validation accuracy')
    plt.colorbar(label='Allocation rate')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'pareto.png')
    print(json.dumps({k:v for k,v in results.items() if k.startswith('test_') or k=='submission'}, indent=2))

if __name__ == '__main__':
    main()
