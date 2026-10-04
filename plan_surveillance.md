# Plan de surveillance en production — ÉquiAlgo

## Objectifs

Détecter les erreurs de données, les violations du budget d'octroi, les disparités entre groupes et la dérive du modèle sans confondre l'accord avec le comité historique et le mérite des candidat·es. Ce plan est une proposition : le comité doit approuver les cibles d'équité, les seuils d'alerte et les responsabilités avant tout déploiement.

## Contrôles à chaque lot de décisions

- Vérifier le schéma, les valeurs manquantes ou hors plage, les identifiants en double et la correspondance entre les entrées et les décisions.
- Vérifier que `decision_octroi` est binaire, que la sortie respecte le format exigé et que le taux global d'octroi est compris entre **36 % et 44 %**.
- Calculer les effectifs, taux d'octroi et écarts absolus par groupe Centre/Éloignée et par région. Afficher les intervalles de confiance et signaler les petits effectifs; ne pas interpréter un écart instable comme une preuve concluante.
- Archiver les versions des données, du modèle, des paramètres, des métriques et des décisions, en limitant l'accès aux données individuelles.

## Suivi mensuel et audit trimestriel

Chaque mois, suivre les distributions des caractéristiques et des scores, notamment la distance domicile-campus, le code postal, les heures travaillées et le revenu, identifiés comme informatifs sur la région. Comparer les mesures au jeu de référence et vérifier si la qualité de données ou les catégories changent.

Chaque trimestre, faire examiner par une personne indépendante les taux d'octroi par groupe et région, la parité démographique, les taux de vrais positifs lorsqu'une mesure indépendante du mérite est disponible, les erreurs par groupe, les recours et les décisions révisées. Documenter les limites statistiques et les modifications depuis le dernier audit.

Le critère primaire retenu est la **parité démographique** : comparer les taux de sélection Centre/Éloignée, sans conditionner sur `decision_octroi`. Toutefois, le classifieur est entraîné sur cette étiquette historique; la contrainte d'équité ne supprime pas tous les biais potentiellement appris. `decision_octroi` décrit la pratique passée du comité et ne doit pas être présentée comme vérité du mérite. Tant qu'un résultat indépendant n'est pas disponible, les mesures conditionnelles sur cette étiquette doivent être rapportées seulement comme diagnostics de reproduction des décisions historiques, avec cette réserve explicite.

## Alertes et réponse

Déclencher une revue si :

1. le taux global d'octroi sort de la plage de **36–44 %**;
2. l'écart de la métrique d'équité adoptée dépasse le seuil préapprouvé par le comité lors de deux périodes consécutives;
3. un groupe est trop peu représenté pour une estimation fiable;
4. une dérive importante des données, des scores ou des taux d'erreur est observée;
5. un problème de qualité, une plainte ou un recours révèle un effet inattendu.

En cas d'alerte, suspendre l'automatisation pour le périmètre concerné, faire traiter les dossiers par une personne habilitée, vérifier données et proxys, analyser les erreurs et les effets par groupe, puis consigner la cause, les mesures correctives et l'approbation avant réactivation. Les candidat·es doivent disposer d'un mécanisme de recours humain.

## Gouvernance du critère d'équité

Le modèle final d'essai utilise `DemographicParity`. Ce choix donne priorité à l'accès comparable aux bourses et évite de traiter la décision historique biaisée comme indicateur de mérite. La parité démographique ne garantit pas à elle seule une répartition juste selon toutes les autres dimensions; ses compromis doivent être réexaminés avec les personnes concernées. L'égalité des chances peut être rapportée ultérieurement à titre diagnostique, mais seulement avec une mesure de mérite indépendante et légitime.
