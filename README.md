# DESCRIPTION DU PROJET

## Vidéo de démonstration
[![Démonstration](image1.png)](https://youtu.be/vNuyLwfcpvA)

## Collecte des données

La collecte des données repose principalement sur deux sources :

- L'API de la SNCF pour les données relatives aux TER et TGV : https://api.sncf.com/v1
- Des bases de données open source de la SNCF et de l'INSEE.

Ces sources offrent un accès à de nombreuses données sur les réseaux de transports publics en France (perturbations, trajets, lignes, etc.). Un script automatisé interroge régulièrement cette API pour récupérer les données nécessaires à l'analyse.

## Fonctionnalités

Ce tableau de bord répond efficacement à une variété de questions liées à la gestion des perturbations sur les réseaux TGV et TER, accessible en seulement quelques clics. Voici quelques exemples :

  •	Quel type de perturbation cause le plus de retards en moyenne ?
  •	Combien de TER ont été en circulation et combien ont été perturbés entre 19h et 21h lors de la première semaine de juillet ?
  •	Quel est le pourcentage de perturbations en heures de pointe et en heures creuses pour le mois de juin ?
  •	Quels sont les types de causes les plus fréquentes sur les trains OUIGO ?
  •	Quelle est la gare parisienne où les perturbations sont en moyenne gérées le plus rapidement ?
  •	Comment se répartit dans la journée le nombre d'incidents liés aux bagages oubliés ? Quelle est la durée moyenne de retard causée par cette cause ?

La visualisation des données est intuitive et accessible à tous, permettant une compréhension rapide des informations critiques pour une gestion efficace des réseaux de transport. Ce tableau de bord se décompose en cinq parties principales, chacune étant modifiable selon une plage horaire et une période choisie :
1.	Analyse Générale des Perturbations : Visualisation du nombre de trajets, de perturbations, pourcentage de trajets perturbés, temps moyen et médian des perturbations, ainsi que le temps moyen de retard causé par perturbation. Pour comprendre les tendances globales des perturbations et évaluer l'efficacité des opérations à grande échelle.
2.	Détail des Causes de Perturbations : Suivi et analyse des causes des perturbations au fil de la journée, avec des statistiques par type de perturbation pour comprendre les facteurs principaux contribuant aux retards et interruptions.
3.	Détail des Perturbations par Réseau : Analyse détaillée des statistiques de perturbations pour chaque réseau, incluant un décryptage des causes spécifiques impactant chaque réseau. Pour comprendre les particularités et défis propres à chaque réseau et adapter les stratégies de gestion en conséquence.
4.	Analyse Géographique : Visualisation des retards et des durées des perturbations par gare, commune, département, région ou région SNCF, avec un détail des causes associées à ces retards pour une compréhension locale et régionale. Pour comprendre les variations géographiques des perturbations et identifier les zones nécessitant une attention particulière.
5.	Détail des Lignes : Analyse des perturbations et des retards par ligne spécifique, permettant d'identifier les lignes les plus affectées et les causes sous-jacentes des perturbations. Pour comprendre les problèmes spécifiques à chaque ligne et optimiser la gestion des ressources et des interventions.

