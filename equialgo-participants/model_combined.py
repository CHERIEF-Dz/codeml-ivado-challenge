import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
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
# 2. CRÉATION DE L'INDEX DE DÉSAVANTAGE GÉOGRAPHIQUE (Combined Feature)
# ==========================================
def create_geo_index(df):
    # Normalisation simple pour combiner les variables
    # Plus l'index est haut, plus le candidat est "éloigné/désavantagé"
    
    # 1. Binaire région (0 ou 1)
    is_eloignee = np.where(df['region_administrative'].isin(ELOIGNEES), 1, 0)
    
    # 2. Distance normalisée (0 à 1)
    dist_min, dist_max = df['distance_domicile_campus_km'].min(), df['distance_domicile_campus_km'].max()
    dist_norm = (df['distance_domicile_campus_km'] - dist_min) / (dist_max - dist_min + 1e-6)
    
    # 3. Revenu inverse normalisé (0 à 1) - Moins on a d'argent, plus on est désavantagé
    rev_min, rev_max = df['revenu_familial_estime'].min(), df['revenu_familial_estime'].max()
    rev_norm_inv = 1 - ((df['revenu_familial_estime'] - rev_min) / (rev_max - rev_min + 1e-6))
    
    # Index combiné (Moyenne pondérée)
    # On donne du poids à la région et à la distance
    geo_index = (is_eloignee * 0.5) + (dist_norm * 0.3) + (rev_norm_inv * 0.2)
    return geo_index

demandes['geo_disadvantage_index'] = create_geo_index(demandes)
candidats['geo_disadvantage_index'] = create_geo_index(candidats)

# ==========================================
# 3. PRÉPARATION DES DONNÉES
# ==========================================
# On garde la Cote R et l'index, mais on supprime les colonnes sensibles brutes
# pour éviter que le modèle apprenne un biais trop direct.
DROP_COLS = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']

# Split 70% Train / 30% Test (comme demandé)
train_df, test_df = train_test_split(
    demandes, test_size=0.3, random_state=42
)

def encoder(df_train, df_cible):
    X = pd.get_dummies(
        df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + DROP_COLS, errors='ignore'),
        columns=['programme_etudes'],
    )
    Xc = pd.get_dummies(
        df_cible.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + DROP_COLS, errors='ignore'),
        columns=['programme_etudes'],
    ).reindex(columns=X.columns, fill_value=0)
    return X, Xc

X_train, X_test = encoder(train_df, test_df)
y_train, y_test = train_df['decision_octroi'], test_df['decision_octroi']
s_test = groupe_region(test_df)

# ==========================================
# 4. ENTRAÎNEMENT (Modèle Indexé)
# ==========================================
# On utilise un RF avec un peu plus de régularisation pour éviter le sur-apprentissage du biais
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=30, random_state=42)
model.fit(X_train, y_train)

# Evaluation
preds_test = model.predict(X_test)
acc = accuracy_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test, average='macro')
dp_diff = demographic_parity_difference(y_test, preds_test, sensitive_features=s_test)

print("\n--- Résultats Modèle Index Géographique ---")
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

# Ajustement exact pour 40%
if final_preds.sum() > int(len(candidats) * BUDGET_TARGET):
    indices = np.where(final_preds == 1)[0]
    to_rem = indices[np.argsort(scores_cand[indices])[:(final_preds.sum() - int(len(candidats) * BUDGET_TARGET))]]
    final_preds[to_rem] = 0

pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': final_preds}).to_csv('predictions_combined.csv', index=False)
print("\npredictions_combined.csv généré avec succès.")
