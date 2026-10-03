import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

def run_experiment(name, remove_proxies=False, balanced_sampling=False, eps=0.05):
    print(f"\n--- Test: {name} ---")
    demandes = pd.read_csv('data/donnees_demandes.csv')
    candidats = pd.read_csv('data/candidats_evaluation.csv')
    
    ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
    demandes['groupe'] = np.where(demandes['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')
    
    # 1. Équilibrage AVANT split si demandé
    if balanced_sampling:
        centre = demandes[demandes['groupe'] == 'Centre']
        eloignee = demandes[demandes['groupe'] == 'Eloignee']
        m = min(len(centre), len(eloignee))
        df_final = pd.concat([centre.sample(m, random_state=42), eloignee.sample(m, random_state=42)])
    else:
        df_final = demandes

    train, test = train_test_split(df_final, test_size=0.3, random_state=42)
    
    # 2. Gestion des colonnes
    CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']
    PROXYS = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']
    
    def encode(df_train, df_test, drop_cols=[]):
        # On retire les colonnes à supprimer
        d_train = df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + drop_cols, errors='ignore')
        d_test = df_test.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + drop_cols, errors='ignore')
        
        # On encode les catégories restantes
        remaining_cats = [c for c in CATEGORIELLES if c not in drop_cols]
        X = pd.get_dummies(d_train, columns=remaining_cats)
        Xc = pd.get_dummies(d_test, columns=remaining_cats).reindex(columns=X.columns, fill_value=0)
        return X, Xc

    drop_list = PROXYS if remove_proxies else []
    X_train, X_test = encode(train, test, drop_list)
    y_train, y_test = train['decision_octroi'], test['decision_octroi']
    s_train = np.where(train['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')
    s_test = np.where(test['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

    # 3. Modèle
    base = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=42)
    mitigator = ExponentiatedGradient(base, constraints=DemographicParity(), eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    
    preds = mitigator.predict(X_test)
    
    # 4. Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    diff = demographic_parity_difference(y_test, preds, sensitive_features=s_test)
    
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Parity Diff: {diff:.4f}")
    return acc, f1, diff

# Tests d'hypothèses
results = {}
results['Baseline_Simple'] = run_experiment('Baseline', remove_proxies=False, balanced_sampling=False, eps=0.1)
results['Balanced_Only'] = run_experiment('Balanced Sampling', remove_proxies=False, balanced_sampling=True, eps=0.1)
results['RemoveProxies_Only'] = run_experiment('Remove Proxies', remove_proxies=True, balanced_sampling=False, eps=0.1)
results['Combo_Best'] = run_experiment('Balanced + Remove Proxies', remove_proxies=True, balanced_sampling=True, eps=0.05)

print("\n--- Résumé Final ---")
for k, v in results.items():
    print(f"{k}: Acc={v[0]:.3f}, F1={v[1]:.3f}, Diff={v[2]:.3f}")
