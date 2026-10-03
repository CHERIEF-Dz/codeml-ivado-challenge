# Justification du choix de correction - ÉquiAlgo

## Problématique identifiée
Le modèle de base souffre d'un biais régional marqué : les centres urbains (Montréal, Capitale-Nationale) ont un taux d'octroi nettement supérieur (~48%) aux régions éloignées (~27%), alors que l'écart de mérite (cote R moyenne) est minime (28.0 vs 27.3).

## Approche adoptée : Équilibrage de l'échantillonnage (Resampling)
Au lieu de simplement supprimer la variable `region_administrative` (ce qui est inefficace à cause des variables proxys comme le code postal ou le revenu), nous avons opté pour un **échantillonnage équilibré** durant la phase d'entraînement.

### Détails techniques :
1. **Under-sampling stratégique** : Nous avons créé un ensemble d'entraînement où les candidats des centres urbains et des régions éloignées sont représentés à parts égales (50% / 50%).
2. **Objectif** : Forcer le modèle à apprendre les caractéristiques du mérite indépendamment de la densité démographique de la région. Le modèle ne peut plus "s'appuyer" sur la prédominance statistique des centres urbains pour prédire l'octroi.
3. **Définition du Mérite** : Nous considérons que le mérite doit être évalué de manière transverse. En équilibrant les groupes, on réduit l'influence du biais historique du comité qui favorisait systemicment les centres.

## Justification Éthique
Ce choix repose sur le principe d'**Égalité des Chances**. En neutralisant la disproportion numérique des groupes dans les données d'entraînement, nous visons à ce que :
- Un candidat qualifié en région éloignée ait la même probabilité d'être sélectionné qu'un candidat qualifié en centre urbain.
- Le modèle ne reproduise pas la discrimination historique tout en conservant sa capacité prédictive basée sur les indicateurs de performance académique.

## Validation
La performance est mesurée via le F1-Score et l'Accuracy, tout en surveillant l'écart de parité démographique et le taux de vrais positifs (TPR) par groupe pour s'assurer que la correction ne dégrade pas l'utilité globale du modèle.
