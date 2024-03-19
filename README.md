![](https://villefranche-sur-mer.fr/wp-content/uploads/2023/02/Banniere_travaux_2_975x466.png)

# DESCRIPTION DU PROJET

Comme de nombreux usagers, que ce soit en Île-de-France ou ailleurs en France, j'utilise quotidiennement les transports en commun. Et, comme beaucoup, je fais face régulièrement à des perturbations sur mes trajets. Ce projet a pour but de quantifier et d'analyser ces perturbations, puis de publier ces informations sur des plateformes comme Instagram.

L'idée à la suite de cela serait d'élaborer deux modèles prédictifs des retards. Un modèle de classification qui aura pour but pour un trajet donné d'informer l'utilisateur s'il y a de forte chance que ce trajet soit perturbé (pourrait permettre de choisir entre deux trajets plus ou moins équivalent en temps). Le deuxième modèle sera un modèle de classification et aura pour but de quantifier le retard pour un trajet donné.

## Collecte des données

La collecte des données s'appuie sur deux principales sources via des API :

L'API de la SNCF pour les données relatives aux TER et TGV. : https://api.sncf.com/v1 
L'API de Navitia pour les transports en commun en Île-de-France. : https://api.navitia.io/v1

Ces APIs offrent un accès à un bon nombre de données sur les réseaux de transports publics en France (Perturbations, trajets, lignes, etc...) . Un script automatisé interroge régulièrement ces APIs pour récupérer les données nécessaires à l'analyse.
