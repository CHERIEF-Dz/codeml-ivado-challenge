import pandas as pd
import numpy as np
from fairlearn.metrics import demographic_parity_difference

# Configuration
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
demandes = pd.read_csv('data/donnees_demandes.csv')

def groupe_region(df):
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

# Load predictions
df_dp = pd.read_csv('predictions_parite_demographique.csv')
df_eo = pd.read_csv('predictions_egalite_des_chances.csv')

# Merge with original data to get regions
data_dp = demandes[['id_candidat', 'region_administrative']].merge(df_dp, on='id_candidat')
data_eo = demandes[['id_candidat', 'region_administrative']].merge(df_eo, on='id_candidat')

data_dp['groupe'] = groupe_region(data_dp)
data_eo['groupe'] = groupe_region(data_eo)

def calc_gap(df, name):
    rates = df.groupby('groupe')['decision_octroi'].mean()
    gap = abs(rates['Centre'] - rates['Eloignee'])
    print(f"\n--- {name} ---")
    print(f"Taux Centre: {rates['Centre']:.2%}")
    print(f"Taux Eloignee: {rates['Eloignee']:.2%}")
    print(f"Gap: {gap:.2%}")
    return gap

gap_dp = calc_gap(data_dp, "Parité Démographique")
gap_eo = calc_gap(data_eo, "Égalité des Chances")
