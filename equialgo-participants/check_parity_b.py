import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference

# Configuration
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
demandes = pd.read_csv('data/donnees_demandes.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

demandes['groupe'] = groupe_region(demandes)
PROXYS_REGION = ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km', 'revenu_familial_estime']

# Balanced training set
df_centre = demandes[demandes['groupe'] == 'Centre']
df_eloignee = demandes[demandes['groupe'] == 'Eloignee']
n_samples = min(len(df_centre), len(df_eloignee))
train_balanced = pd.concat([
    df_centre.sample(n_samples, random_state=42),
    df_eloignee.sample(n_samples, random_state=42)
])

# Categorical encoding without region
X = pd.get_dummies(
    train_balanced.drop(columns=['id_candidat', 'decision_octroi', 'groupe'] + PROXYS_REGION, errors='ignore'),
    columns=['programme_etudes'],
)
y = train_balanced['decision_octroi']
s = groupe_region(train_balanced)

# Model B (Without Region)
model_b = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
model_b.fit(X, y)

# Calculate parity on the training set (as proxy)
preds = model_b.predict(X)
dp_diff = demographic_parity_difference(y, preds, sensitive_features=s)
eop_diff = equal_opportunity_difference(y, preds, sensitive_features=s)

print(f"--- Model B (Without Region) Parity Metrics ---")
print(f"Demographic Parity Difference: {dp_diff:.4f}")
print(f"Equal Opportunity Difference: {eop_diff:.4f}")
