import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from fairlearn.metrics import equal_opportunity_difference, selection_rate
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity

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
    # Hyperparamètres STRICTS de la baseline
    base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    constraint = TruePositiveRateParity(difference_bound=b)
    
    mitigator = ExponentiatedGradient(
        estimator=base, 
        constraints=constraint, 
        eps=0.01, # Paramètre d'optimisation
    )
    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    
    preds_val = mitigator.predict(X_val)
    
    # Métriques de validation
    acc = accuracy_score(y_val, preds_val)
    eop_diff = equal_opportunity_difference(y_val, preds_val, sensitive_features=s_val)
    rate = preds_val.mean()
    
    pareto_results.append({
        'bound': b, 'acc': acc, 'eop': eop_diff, 'rate': rate
    })

df_pareto = pd.DataFrame(pareto_results)

# Filtrage STRICT du budget sur le set de validation
df_admissible = df_pareto[(df_pareto['rate'] >= BUDGET_MIN) & (df_pareto['rate'] <= BUDGET_MAX)]

if df_admissible.empty:
    print("Aucun modèle respecte le budget sur Val. Utilisation du modèle le plus proche.")
    best_bound = df_pareto.iloc[len(df_pareto)//2]['bound']
else:
    # On choisit le meilleur compromis : EOP la plus basse possible parmi les admissibles
    best_row = df_admissible.sort_values('eop').iloc[0]
    best_bound = best_row['bound']

print(f"Bound sélectionné : {best_bound} (EOP: {best_row['eop']:.4f}, Acc: {best_row['acc']:.4f}, Rate: {best_row['rate']:.2%})")

# ==========================================
# 4. ÉVALUATION FINALE (sur Test)
# ==========================================
final_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
final_constraint = TruePositiveRateParity(difference_bound=best_bound)
final_model = ExponentiatedGradient(final_base, constraints=final_constraint, eps=0.01)
final_model.fit(X_train, y_train, sensitive_features=s_train)

y_pred_test = final_model.predict(X_test)
final_eop = equal_opportunity_difference(y_test, y_pred_test, sensitive_features=s_test)
final_acc = accuracy_score(y_test, y_pred_test)

print("\n--- RÉSULTATS FINAUX (Set de Test) ---")
print(f"Accuracy: {final_acc:.4f}")
print(f"EOP Difference: {final_eop:.4f}")
print(f"Taux d'octroi: {y_pred_test.mean():.2%}")

# ==========================================
# 5. SOUMISSION (Candidats Évaluation)
# ==========================================
X_hist, X_cand = encoder(train_df, candidats)
final_preds = final_model.predict(X_cand)

soumission = pd.DataFrame({
    'id_candidat': candidats['id_candidat'],
    'decision_octroi': final_preds.astype(int),
})

# GARANTIE BUDGET FINALE
final_rate = soumission['decision_octroi'].mean()
assert BUDGET_MIN <= final_rate <= BUDGET_MAX, f"CRITIQUE: Budget hors plage ({final_rate:.2%})!"

soumission.to_csv('predictions.csv', index=False)
print(f"\npredictions.csv généré avec succès. Taux final: {final_rate:.2%}")

# ==========================================
# 6. VISUALISATION PARETO
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(df_pareto['eop'], df_pareto['acc'], c=df_pareto['bound'], cmap='viridis', s=100)
plt.colorbar(label='difference_bound')
plt.axvline(x=0.05, color='r', linestyle='--', label='Cible 5%')
plt.xlabel('Equal Opportunity Difference (Biais)')
plt.ylabel('Accuracy (Utilité)')
plt.title('Front de Pareto (Validation Set)')
plt.grid(True)
plt.legend()
plt.savefig('pareto_final.png')
