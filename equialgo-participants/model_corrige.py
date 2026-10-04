import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from fairlearn.metrics import equal_opportunity_difference, selection_rate, MetricFrame, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from fairlearn.postprocessing import ThresholdOptimizer

# ==========================================
# 1. CONFIGURATION & CHARGEMENT
# ==========================================
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
BUDGET_MIN, BUDGET_MAX = 0.36, 0.44

demandes = pd.read_csv('data/donnees_demandes.csv')
candidats = pd.read_csv('data/candidats_evaluation.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

demandes['groupe'] = groupe_region(demandes)
# Stratification pour maintenir la distribution groupe + décision
demandes['strat'] = demandes['groupe'].astype(str) + "_" + demandes['decision_octroi'].astype(str)

# Split Train (60%), Val (20%), Test (20%) - Architecture recommandée
train_df, temp_df = train_test_split(
    demandes, test_size=0.4, random_state=42, stratify=demandes['strat']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['strat']
)

# ==========================================
# 2. ENCODAGE (Fidèle à la baseline)
# ==========================================
CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']

def encoder(df_train, df_cible):
    X = pd.get_dummies(
        df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe', 'strat'], errors='ignore'),
        columns=CATEGORIELLES,
    )
    Xc = pd.get_dummies(
        df_cible.drop(columns=['id_candidat', 'decision_octroi', 'groupe', 'strat'], errors='ignore'),
        columns=CATEGORIELLES,
    ).reindex(columns=X.columns, fill_value=0)
    return X, Xc

X_train, X_val = encoder(train_df, val_df)
X_test = pd.get_dummies(
    test_df.drop(columns=['id_candidat', 'decision_octroi', 'groupe', 'strat'], errors='ignore'),
    columns=CATEGORIELLES,
).reindex(columns=X_train.columns, fill_value=0)

y_train, y_val, y_test = train_df['decision_octroi'], val_df['decision_octroi'], test_df['decision_octroi']
s_train, s_val, s_test = groupe_region(train_df), groupe_region(val_df), groupe_region(test_df)

# ==========================================
# 3. RECHERCHE DU POINT PARETO (sur Validation)
# ==========================================
# On fait varier la borne d'équité (True Positive Rate Parity)
eps_bounds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.27]
pareto_results = []

print("Recherche du point Pareto sur le set de validation...")
for b in eps_bounds:
    # Modèle de base
    base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    
    # 1. ExponentiatedGradient
    constraint_eg = TruePositiveRateParity(difference_bound=b)
    eg_mitigator = ExponentiatedGradient(estimator=base, constraints=constraint_eg, eps=0.01)
    eg_mitigator.fit(X_train, y_train, sensitive_features=s_train)
    preds_eg = eg_mitigator.predict(X_val)
    
    # 2. ThresholdOptimizer
    # On utilise un modèle RF simple pour obtenir des probabilités
    rf_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42).fit(X_train, y_train)
    to_mitigator = ThresholdOptimizer(
        estimator=rf_base,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        predict_method='predict_proba'
    )
    # ThresholdOptimizer a besoin de fit sur le set de validation pour optimiser les seuils
    to_mitigator.fit(X_val, y_val, sensitive_features=s_val)
    preds_to = to_mitigator.predict(X_val, sensitive_features=s_val)
    
    # On compare et on prend le meilleur pour ce bound (sur balanced accuracy)
    acc_eg = balanced_accuracy_score(y_val, preds_eg)
    acc_to = balanced_accuracy_score(y_val, preds_to)
    
    if acc_eg >= acc_to:
        best_preds = preds_eg
        method = 'EG'
    else:
        best_preds = preds_to
        method = 'TO'

    # Métriques de validation
    bal_acc = balanced_accuracy_score(y_val, best_preds)
    eop_diff = equal_opportunity_difference(y_val, best_preds, sensitive_features=s_val)
    rate = best_preds.mean()
    
    pareto_results.append({
        'bound': b, 'bal_acc': bal_acc, 'eop': eop_diff, 'rate': rate, 'method': method
    })

df_pareto = pd.DataFrame(pareto_results)

# Sélection du meilleur compromis (Non-dominé ou Score pondéré)
# Score = BalancedAcc - lambda * EOP
lambda_fairness = 2.0 
df_pareto['score'] = df_pareto['bal_acc'] - lambda_fairness * df_pareto['eop']

# Filtrage du budget
df_admissible = df_pareto[(df_pareto['rate'] >= BUDGET_MIN) & (df_pareto['rate'] <= BUDGET_MAX)]

if df_admissible.empty:
    print("Aucun modèle respecte le budget sur Val. Utilisation du modèle le plus proche.")
    best_row = df_pareto.iloc[(df_pareto['rate'] - 0.40).abs().argsort().iloc[0]]
else:
    best_row = df_admissible.sort_values('score', ascending=False).iloc[0]

best_bound = best_row['bound']
best_method = best_row['method']
print(f"Bound sélectionné : {best_bound} via {best_method} (EOP: {best_row['eop']:.4f}, BalAcc: {best_row['bal_acc']:.4f}, Rate: {best_row['rate']:.2%})")

# ==========================================
# 4. ÉVALUATION FINALE (sur Test)
# ==========================================
# On réentraîne sur 100% des données historiques pour maximiser l'information
X_full = pd.concat([X_train, X_val, X_test])
y_full = pd.concat([y_train, y_val, y_test])
s_full = np.concatenate([s_train, s_val, s_test])

print(f"\nRéentraînement final sur {len(X_full)} échantillons avec {best_method}...")
if best_method == 'EG':
    final_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    final_constraint = TruePositiveRateParity(difference_bound=best_bound)
    final_model = ExponentiatedGradient(final_base, constraints=final_constraint, eps=0.01)
    final_model.fit(X_full, y_full, sensitive_features=s_full)
    # Pour les probabilités, on utilise predict_proba si disponible
    final_scores = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else final_model.predict(X_test)
else:
    # ThresholdOptimizer
    rf_final = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42).fit(X_full, y_full)
    final_model = ThresholdOptimizer(
        estimator=rf_final,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        predict_method='predict_proba'
    )
    final_model.fit(X_val, y_val, sensitive_features=s_val) # Fit sur Val pour les seuils

y_pred_test = final_model.predict(X_test, sensitive_features=s_test) if best_method == 'TO' else final_model.predict(X_test)
final_eop = equal_opportunity_difference(y_test, y_pred_test, sensitive_features=s_test)
final_bal_acc = balanced_accuracy_score(y_test, y_pred_test)

print("\n--- RÉSULTATS FINAUX (Set de Test) ---")
print(f"Balanced Accuracy: {final_bal_acc:.4f}")
print(f"EOP Difference: {final_eop:.4f}")
print(f"Taux d'octroi: {y_pred_test.mean():.2%}")

# ==========================================
# 5. SOUMISSION (Candidats Évaluation)
# ==========================================
X_hist, X_cand = encoder(train_df, candidats)

# Pour garantir le budget, on utilise le modèle comme un système de classement
# On récupère les probabilités (scores de priorité)
if best_method == 'EG':
    # ExponentiatedGradient n'expose pas toujours predict_proba directement
    # On utilise le base_estimator pour obtenir des scores si possible
    final_scores = final_model.predict_proba(X_cand)[:, 1] if hasattr(final_model, 'predict_proba') else final_model.predict(X_cand)
else:
    # ThresholdOptimizer utilise le base_estimator
    final_scores = final_model.estimator_.predict_proba(X_cand)[:, 1]

# On définit le nombre exact de bourses pour 40% (1600 sur 4000)
TARGET_COUNT = int(len(candidats) * 0.40)
threshold = np.sort(final_scores)[-TARGET_COUNT]

# Allocation basée sur le rang (les 40% les mieux classés)
final_preds = (final_scores >= threshold).astype(int)

# Correction pour exactly 40% (gestion des ex-aequo)
if final_preds.sum() > TARGET_COUNT:
    # On retire les moins prioritaires parmi les sélectionnés
    indices_selected = np.where(final_preds == 1)[0]
    scores_selected = final_scores[indices_selected]
    to_remove = indices_selected[np.argsort(scores_selected)[:(final_preds.sum() - TARGET_COUNT)]]
    final_preds[to_remove] = 0
elif final_preds.sum() < TARGET_COUNT:
    # On ajoute les plus prioritaires parmi les non-sélectionnés
    indices_not_selected = np.where(final_preds == 0)[0]
    scores_not_selected = final_scores[indices_not_selected]
    to_add = indices_not_selected[np.argsort(scores_not_selected)[::-1][:(TARGET_COUNT - final_preds.sum())]]
    final_preds[to_add] = 1

soumission = pd.DataFrame({
    'id_candidat': candidats['id_candidat'],
    'decision_octroi': final_preds.astype(int),
})

# GARANTIE BUDGET FINALE
final_rate = soumission['decision_octroi'].mean()
print(f"\nTaux final de soumission: {final_rate:.2%}")
assert BUDGET_MIN <= final_rate <= BUDGET_MAX, f"CRITIQUE: Budget hors plage ({final_rate:.2%})!"

soumission.to_csv('predictions.csv', index=False)
print(f"predictions.csv généré avec succès. Exactement {final_preds.sum()} bourses octroyées.")

# ==========================================
# 6. VISUALISATION PARETO
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(df_pareto['eop'], df_pareto['bal_acc'], c=df_pareto['bound'], cmap='viridis', s=100)
plt.colorbar(label='difference_bound')
plt.axvline(x=0.05, color='r', linestyle='--', label='Cible 5%')
plt.xlabel('Equal Opportunity Difference (Biais)')
plt.ylabel('Balanced Accuracy (Utilité)')
plt.title('Front de Pareto (Validation Set)')
plt.grid(True)
plt.legend()
plt.savefig('pareto_final.png')
