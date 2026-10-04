import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity

# ==========================================
# 1. CONFIGURATION & CHARGEMENT
# ==========================================
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
BUDGET_TARGET = 0.40

demandes = pd.read_csv('data/donnees_demandes.csv')
candidats = pd.read_csv('data/candidats_evaluation.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

# --- STRATÉGIE BALANCE (50/50) ---
df_centre = demandes[groupe_region(demandes) == 'Centre']
df_eloignee = demandes[groupe_region(demandes) == 'Eloignee']
n_samples = min(len(df_centre), len(df_eloignee))
train_balanced = pd.concat([
    df_centre.sample(n_samples, random_state=42),
    df_eloignee.sample(n_samples, random_state=42)
])

# Split 70% Train / 30% Test
train_df, test_df = train_test_split(train_balanced, test_size=0.3, random_state=42)

# ==========================================
# 2. ENCODAGE (Sans Région pour éviter le biais direct)
# ==========================================
PROXYS_REGION = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']

def encoder(df_train, df_cible):
    X = pd.get_dummies(
        df_train.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + PROXYS_REGION, errors='ignore'),
        columns=['programme_etudes'],
    )
    Xc = pd.get_dummies(
        df_cible.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + PROXYS_REGION, errors='ignore'),
        columns=['programme_etudes'],
    ).reindex(columns=X.columns, fill_value=0)
    return X, Xc

X_train, X_test = encoder(train_df, test_df)
y_train, y_test = train_df['decision_octroi'], test_df['decision_octroi']
s_train = groupe_region(train_df)
s_test = groupe_region(test_df)

def get_predictions(constraints, name):
    print(f"\n--- Entraînement Modèle : {name} ---")
    base = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    mitigator = ExponentiatedGradient(estimator=base, constraints=constraints, eps=0.01)
    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    
    # Evaluation
    preds_test = mitigator.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds_test):.4f}")
    print(f"F1 Macro: {f1_score(y_test, preds_test, average='macro'):.4f}")
    
    # Soumission Ranking 40%
    X_train_cand, X_cand = encoder(train_df, candidats)
    
    # ExponentiatedGradient n'a pas predict_proba, on utilise predict pour le ranking 
    # ou on accède au base_estimator si possible. 
    # La méthode la plus sûre pour ExponentiatedGradient est d'utiliser predict() 
    # mais pour un ranking, nous allons utiliser le modèle de base entraîné.
    
    preds_cand = mitigator.predict(X_cand)
    
    # Puisque nous n'avons pas de probabilités directes, nous utilisons les prédictions binaires
    # et nous ajustons pour atteindre exactement 40%.
    final_preds = preds_cand.astype(int)
    
    if final_preds.sum() > int(len(candidats) * BUDGET_TARGET):
        # Si trop de bourses, on retire aléatoirement parmi les octroyés
        indices = np.where(final_preds == 1)[0]
        to_rem = np.random.choice(indices, final_preds.sum() - int(len(candidats) * BUDGET_TARGET), replace=False)
        final_preds[to_rem] = 0
    elif final_preds.sum() < int(len(candidats) * BUDGET_TARGET):
        # Si pas assez, on ajoute aléatoirement parmi les refusés
        indices = np.where(final_preds == 0)[0]
        to_add = np.random.choice(indices, int(len(candidats) * BUDGET_TARGET) - final_preds.sum(), replace=False)
        final_preds[to_add] = 1
        
    filename = f"predictions_{name.lower().replace(' ', '_')}.csv"
    pd.DataFrame({'id_candidat': candidats['id_candidat'], 'decision_octroi': final_preds}).to_csv(filename, index=False)
    print(f"{filename} généré.")

# ==========================================
# 3. EXÉCUTION DES DEUX VERSIONS
# ==========================================

# Version 1 : Parité Démographique (Taux d'octroi égal)
get_predictions(DemographicParity(), "Parite Demographique")

# Version 2 : Égalité des Chances (TPR égal)
get_predictions(TruePositiveRateParity(), "Egalite des Chances")
