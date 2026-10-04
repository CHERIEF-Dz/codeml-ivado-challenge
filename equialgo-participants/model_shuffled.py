import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference

# ==========================================
# 1. CONFIGURATION & CHARGEMENT
# ==========================================
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
BUDGET_TARGET = 0.40

demandes = pd.read_csv('data/donnees_demandes.csv')
candidats = pd.read_csv('data/candidats_evaluation.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

# ==========================================
# 2. STRATÉGIE DE MÉLANGE (SHUFFLING REGION)
# ==========================================
# L'idée est de briser la corrélation entre la région et la décision d'octroi 
# en mélangeant les étiquettes de région tout en gardant la distribution globale.
df_shuffled = demandes.copy()
shuffled_regions = df_shuffled['region_administrative'].sample(frac=1, random_state=42).values
df_shuffled['region_administrative'] = shuffled_regions

# On définit les groupes sur les données mélangées
df_shuffled['groupe'] = groupe_region(df_shuffled)

# Split 70% Train / 30% Test
train_df, test_df = train_test_split(
    df_shuffled, test_size=0.3, random_state=42
)

# ==========================================
# 3. ENCODAGE
# ==========================================
CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']

def encoder(df_train, df_cible):
    X = pd.get_dummies(
        df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe'], errors='ignore'),
        columns=CATEGORIELLES,
    )
    Xc = pd.get_dummies(
        df_cible.drop(columns=['id_candidat', 'decision_octroi', 'groupe'], errors='ignore'),
        columns=CATEGORIELLES,
    ).reindex(columns=X.columns, fill_value=0)
    return X, Xc

X_train, X_test = encoder(train_df, test_df)
y_train, y_test = train_df['decision_octroi'], test_df['decision_octroi']
s_test = groupe_region(test_df)

# ==========================================
# 4. ENTRAÎNEMENT
# ==========================================
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)

# Evaluation
preds_test = model.predict(X_test)
acc = accuracy_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test, average='macro')
dp_diff = demographic_parity_difference(y_test, preds_test, sensitive_features=s_test)

print("\n--- Résultats Modèle Mélangé (Shuffled Region) ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Macro: {f1:.4f}")
print(f"Demographic Parity Diff: {dp_diff:.4f}")

# ==========================================
# 5. SOUMISSION (Ranking 40%)
# ==========================================
X_train_cand, X_cand = encoder(train_df, candidats)
scores_cand = model.predict_proba(X_cand)[:, 1]
threshold = np.sort(scores_cand)[-int(len(candidats) * BUDGET_TARGET)]
final_preds = (scores_cand >= threshold).astype(int)

# Ajustement exact
if final_preds.sum() > int(len(candidats) * BUDGET_TARGET):
    indices = np.where(final_preds == 1)[0]
    to_rem = indices[np.argsort(scores_cand[indices])[:(final_preds.sum() - int(len(candidats) * BUDGET_TARGET))]]
    final_preds[to_rem] = 0

pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': final_preds}).to_csv('predictions_shuffled.csv', index=False)
print("\npredictions_shuffled.csv généré avec succès.")
