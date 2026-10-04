import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION & CHARGEMENT
# ==========================================
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
BUDGET_TARGET = 0.40

demandes = pd.read_csv('data/donnees_demandes.csv')
candidats = pd.read_csv('data/candidats_evaluation.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

demandes['groupe'] = groupe_region(demandes)

# --- STRATÉGIE 50/50 ---
# On crée un dataset d'entraînement équilibré entre Centre et Eloignee
df_centre = demandes[demandes['groupe'] == 'Centre']
df_eloignee = demandes[demandes['groupe'] == 'Eloignee']

# On prend un nombre égal d'échantillons (basé sur le groupe minoritaire)
n_samples = min(len(df_centre), len(df_eloignee))
train_balanced = pd.concat([
    df_centre.sample(n_samples, random_state=42),
    df_eloignee.sample(n_samples, random_state=42)
])

# Split 70% Train / 30% Test (sur le set équilibré pour l'entraînement)
train_df, test_df = train_test_split(
    train_balanced, test_size=0.3, random_state=42
)

# ==========================================
# 2. ENCODAGE & FILTRAGE
# ==========================================
CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']
PROXYS_REGION = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']

def encoder(df_train, df_cible, drop_cols=[]):
    # On supprime les colonnes spécifiées (ex: la région et ses proxys)
    X = pd.get_dummies(
        df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + drop_cols, errors='ignore'),
        columns=[c for c in CATEGORIELLES if c not in drop_cols],
    )
    Xc = pd.get_dummies(
        df_cible.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + drop_cols, errors='ignore'),
        columns=[c for c in CATEGORIELLES if c not in drop_cols],
    ).reindex(columns=X.columns, fill_value=0)
    return X, Xc

y_train, y_test = train_df['decision_octroi'], test_df['decision_octroi']

# ==========================================
# 3. MODÈLE A : AVEC RÉGION
# ==========================================
print("\n--- Entraînement Modèle A (Avec Région) ---")
X_train_a, X_test_a = encoder(train_df, test_df)
X_train_a_cand, X_cand_a = encoder(train_df, candidats)

model_a = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model_a.fit(X_train_a, y_train)

# Evaluation
preds_test_a = model_a.predict(X_test_a)
print(f"Accuracy: {accuracy_score(y_test, preds_test_a):.4f}")
print(f"F1 Macro: {f1_score(y_test, preds_test_a, average='macro'):.4f}")

# Soumission A (Ranking pour 40%)
scores_a = model_a.predict_proba(X_cand_a)[:, 1]
threshold_a = np.sort(scores_a)[-int(len(candidats) * BUDGET_TARGET)]
final_preds_a = (scores_a >= threshold_a).astype(int)
# Ajustement exact
if final_preds_a.sum() > int(len(candidats) * BUDGET_TARGET):
    indices = np.where(final_preds_a == 1)[0]
    to_rem = indices[np.argsort(scores_a[indices])[:(final_preds_a.sum() - int(len(candidats) * BUDGET_TARGET))]]
    final_preds_a[to_rem] = 0

pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': final_preds_a}).to_csv('predictions_with_region.csv', index=False)
print("predictions_with_region.csv généré.")

# ==========================================
# 4. MODÈLE B : SANS RÉGION (NI PROXYS)
# ==========================================
print("\n--- Entraînement Modèle B (Sans Région ni Proxys) ---")
X_train_b, X_test_b = encoder(train_df, test_df, drop_cols=PROXYS_REGION)
X_train_b_cand, X_cand_b = encoder(train_df, candidats, drop_cols=PROXYS_REGION)

model_b = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model_b.fit(X_train_b, y_train)

# Evaluation
preds_test_b = model_b.predict(X_test_b)
print(f"Accuracy: {accuracy_score(y_test, preds_test_b):.4f}")
print(f"F1 Macro: {f1_score(y_test, preds_test_b, average='macro'):.4f}")

# Soumission B (Ranking pour 40%)
scores_b = model_b.predict_proba(X_cand_b)[:, 1]
threshold_b = np.sort(scores_b)[-int(len(candidats) * BUDGET_TARGET)]
final_preds_b = (scores_b >= threshold_b).astype(int)
# Ajustement exact
if final_preds_b.sum() > int(len(candidats) * BUDGET_TARGET):
    indices = np.where(final_preds_b == 1)[0]
    to_rem = indices[np.argsort(scores_b[indices])[:(final_preds_b.sum() - int(len(candidats) * BUDGET_TARGET))]]
    final_preds_b[to_rem] = 0

pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': final_preds_b}).to_csv('predictions_without_region.csv', index=False)
print("predictions_without_region.csv généré.")
