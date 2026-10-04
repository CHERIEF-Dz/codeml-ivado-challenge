"""Historical-only HDBSCAN proxy audit; cluster IDs are not merit labels.

No prediction files are changed. Compare numeric feature views and cluster sizes,
then check sensitivity to integer work-hour measurement resolution.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'audit_results'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / 'data/donnees_demandes.csv')
remote = ~df.region_administrative.isin(['Montreal', 'Capitale-Nationale'])
score = df.cote_r_equivalent + 0.143817 * df.heures_travail_semaine
selected = score.rank(method='first', ascending=False).le(4000)
X = df[['cote_r_equivalent', 'heures_travail_semaine', 'revenu_familial_estime', 'distance_domicile_campus_km']].copy()
X['revenu_familial_estime'] = np.log1p(X['revenu_familial_estime'])
X['distance_domicile_campus_km'] = np.log1p(X['distance_domicile_campus_km'])
rows = []
for feature_set in ['merit_2d', 'continuous_4d']:
    matrix = X.iloc[:, :2] if feature_set == 'merit_2d' else X
    z = StandardScaler().fit_transform(matrix)
    for min_cluster_size in [100, 250, 500]:
        clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=20, n_jobs=1, copy=True)
        labels = clustering.fit_predict(z)
        clusters = sorted(set(labels) - {-1})
        inlier = labels >= 0
        result = {
            'feature_set': feature_set, 'min_cluster_size': min_cluster_size,
            'min_samples': 20, 'clusters': len(clusters),
            'noise_count': int((~inlier).sum()), 'noise_fraction': float((~inlier).mean()),
            'region_adjusted_rand_including_noise': float(adjusted_rand_score(remote, labels)),
            'silhouette_excluding_noise': float(silhouette_score(z[inlier], labels[inlier], sample_size=min(2000, int(inlier.sum())), random_state=41)) if len(clusters)>1 else None,
            'groups': []
        }
        for label in [-1] + clusters:
            m = labels == label
            if not m.any(): continue
            g = {
                'label': int(label), 'count': int(m.sum()), 'remote_share': float(remote[m].mean()),
                'historical_acceptance': float(df.decision_octroi[m].mean()),
                'effort_selection': float(selected[m].mean()),
                'mean_r': float(df.cote_r_equivalent[m].mean()), 'mean_hours': float(df.heures_travail_semaine[m].mean()),
                'mean_income': float(df.revenu_familial_estime[m].mean()), 'mean_distance': float(df.distance_domicile_campus_km[m].mean()),
                'centre_historical_acceptance': float(df.decision_octroi[m & ~remote].mean()) if (m & ~remote).any() else None,
                'remote_historical_acceptance': float(df.decision_octroi[m & remote].mean()) if (m & remote).any() else None,
            }
            result['groups'].append(g)
        rows.append(result)
        print(json.dumps(result), flush=True)
(OUT/'hdbscan_results.json').write_text(json.dumps(rows, indent=2))

# Diagnostic perturbation only: original data and allocation remain unchanged.
original = StandardScaler().fit_transform(X.iloc[:, :2])
reference = HDBSCAN(min_cluster_size=250,min_samples=20,n_jobs=1,copy=True).fit_predict(original)
jitter_results=[]
for seed in [41,42,43]:
    altered=X.iloc[:, :2].copy()
    altered['heures_travail_semaine'] = altered['heures_travail_semaine'] + np.random.default_rng(seed).uniform(-.49,.49,len(altered))
    perturbed=StandardScaler().fit_transform(altered)
    labels=HDBSCAN(min_cluster_size=250,min_samples=20,n_jobs=1,copy=True).fit_predict(perturbed)
    jitter_results.append({'jitter_hours':.49,'seed':seed,'clusters':len(set(labels)-{-1}),
        'noise_fraction':float(np.mean(labels==-1)),
        'ari_with_unjittered':float(adjusted_rand_score(reference,labels))})
(OUT/'hdbscan_jitter.json').write_text(json.dumps(jitter_results,indent=2)+'\n')
