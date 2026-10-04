import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
demandes = pd.read_csv('data/donnees_demandes.csv')
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
demandes['groupe'] = np.where(demandes['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')

# 2. Merit Consistency Analysis (Cote R deciles)
# Create deciles for Cote R to see the grant rate per academic level
demandes['decile_cote_r'] = pd.qcut(
    demandes['cote_r_equivalent'],
    q=10,
    duplicates='drop'
)

# Group by decile and region
audit_merite = demandes.groupby(
    ['decile_cote_r', 'groupe'], 
    observed=False
)['decision_octroi'].mean().unstack()

print("--- Analyse de Cohérence du Mérite (Taux d'octroi par décile de Cote R) ---")
print(audit_merite)

# 3. Visualization
plt.figure(figsize=(12, 6))
audit_merite.plot(marker='o', linewidth=2)
plt.title("Taux d'octroi vs Déciles de Cote R par Groupe")
plt.xlabel("Décile de Cote R (Croissant)")
plt.ylabel("Taux d'octroi")
plt.grid(True, alpha=0.3)
plt.legend(title="Groupe")
plt.savefig('merit_consistency.png')
print("\nGraphique 'merit_consistency.png' généré.")

# 4. Summary Statistics
print("\n--- Résumé du Biais par Décile ---")
audit_merite['diff'] = audit_merite['Centre'] - audit_merite['Eloignee']
print(audit_merite[['Centre', 'Eloignee', 'diff']])
