# DESCRIPTION DU PROJET

## Vidéo de démonstration
[![Démonstration](image1.png)](https://youtu.be/94AKMvSX4UI)

## Collecte des données

La collecte des données repose principalement sur deux sources :

- L'API de la SNCF pour les données relatives aux TER et TGV : https://api.sncf.com/v1
- Des bases de données open source de la SNCF et de l'INSEE.

Ces sources offrent un accès à de nombreuses données sur les réseaux de transports publics en France (perturbations, trajets, lignes, etc.). Un script automatisé interroge régulièrement cette API pour récupérer les données nécessaires à l'analyse.

## Fonctionnalités

Ce tableau de bord se décompose en cinq parties principales, chacune étant modifiable selon une plage horaire et une période choisie :

  1.	Analyse Générale des Perturbations : Visualisation du nombre de trajets, de perturbations, pourcentage de trajets perturbés, temps moyen et médian des perturbations, ainsi que le temps moyen de retard causé par perturbation. Pour comprendre les tendances globales des perturbations et évaluer l'efficacité des opérations à grande échelle.
  2.	Détail des Causes de Perturbations : Suivi et analyse des causes des perturbations au fil de la journée, avec des statistiques par type de perturbation pour comprendre les facteurs principaux contribuant aux retards et interruptions.
  3.	Détail des Perturbations par Réseau : Analyse détaillée des statistiques de perturbations pour chaque réseau, incluant un décryptage des causes spécifiques impactant chaque réseau. Pour comprendre les particularités et défis propres à chaque réseau et adapter les stratégies de gestion en conséquence.
  4.	Analyse Géographique : Visualisation des retards et des durées des perturbations par gare, commune, département, région ou région SNCF, avec un détail des causes associées à ces retards pour une compréhension locale et régionale. Pour comprendre les variations géographiques des perturbations et identifier les zones nécessitant une attention particulière.
  5.	Détail des Lignes : Analyse des perturbations et des retards par ligne spécifique, permettant d'identifier les lignes les plus affectées et les causes sous-jacentes des perturbations. Pour comprendre les problèmes spécifiques à chaque ligne et optimiser la gestion des ressources et des interventions.

