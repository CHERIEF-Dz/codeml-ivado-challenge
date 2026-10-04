# Changelog - ÉquiAlgo Challenge

## Modifications apportées au pipeline de soumission

### 1. Garantie Stricte du Budget (40%)
- **Problème**: L'utilisation d'un simple `assert` sur le taux d'octroi était fragile et pouvait mener à un score de zéro si le budget était légèrement dépassé.
- **Solution**: Implémentation d'une sélection basée sur le **classement (Ranking)**. Le modèle est utilisé pour attribuer un score de priorité à chaque candidat. On sélectionne ensuite exactement les 1 600 candidats les mieux classés (sur 4 000), garantissant mathématiquement un taux d'octroi de 40.00%.

### 2. Optimisation du Front de Pareto
- **Problème**: Le choix du modèle se basait uniquement sur la minimisation de l'EOP, sacrifiant potentiellement trop d'utilité.
- **Solution**: Introduction d'un score de compromis pondéré : $\text{Score} = \text{BalancedAccuracy} - 2.0 \times \text{EOP}$. Cela permet de sélectionner un modèle qui réduit significativement le biais tout en maintenant une performance robuste.

### 3. Comparaison de Mitigateurs (EG vs TO)
- **Solution**: Intégration de `ExponentiatedGradient` (réduction) et `ThresholdOptimizer` (post-processing). Pour chaque borne d'équité, le pipeline sélectionne la méthode offrant la meilleure `Balanced Accuracy`.

### 4. Maximisation des Données
- **Solution**: Mise en place d'un réentraînement final sur **100% des données historiques** (Train + Val + Test) après avoir fixé les hyperparamètres sur le set de validation.

### 5. Audit de Cohérence du Mérite
- **Analyse**: Création d'un audit basé sur les déciles de "Cote R".
- **Résultat**: L'analyse montre que le biais historique est maximal dans les tranches de mérite moyennes (différence de taux d'octroi allant jusqu'à 32% pour un même niveau académique), justifiant pleinement l'utilisation de méthodes de mitigation comme l'Égalité des Chances.

### 6. Métriques de Performance
- **Amélioration**: Passage de l'Accuracy simple à la **Balanced Accuracy** pour une évaluation plus juste de la performance globale.
