import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configuration
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
demandes = pd.read_csv('data/donnees_demandes.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

demandes['groupe'] = groupe_region(demandes)

# 1. Balanced Training (Same as Model B)
df_centre = demandes[demandes['groupe'] == 'Centre']
df_eloignee = demandes[demandes['groupe'] == 'Eloignee']
n_samples = min(len(df_centre), len(df_eloignee))
train_balanced = pd.concat([
    df_centre.sample(n_samples, random_state=42),
    df_eloignee.sample(n_samples, random_state=42)
])

# 2. No Region / Proxies
PROXYS_REGION = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']
X = pd.get_dummies(
    train_balanced.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + PROXYS_REGION, errors='ignore'),
    columns=['programme_etudes'],
)
y = train_balanced['decision_octroi']

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model.fit(X, y)

# 3. Calculate Grant Rate per Region on full historical set
# We use the model to predict the whole dataset to see how it distributes grants
X_full = pd.get_dummies(
    demandes.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + PROXYS_REGION, errors='ignore'),
    columns=['programme_etudes'],
).reindex(columns=X.columns, fill_value=0)

demandes['pred_octroi'] = model.predict(X_full)

# Calculate rate per group
taux_octroi_groupe = demandes.groupby('groupe')['pred_octroi'].mean() * 100

print("--- Taux d'octroi par groupe (Modèle Sans Région) ---")
print(taux_octroi_groupe.map("{:.2f}%".format))
print("\nEcart de parité démographique :", abs(taux_octroi_groupe['Centre'] - taux_octroi_groupe['Eloignee']))
