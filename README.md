![](https://villefranche-sur-mer.fr/wp-content/uploads/2023/02/Banniere_travaux_2_975x466.png)

# DESCRIPTION DU PROJET

Comme de nombreux usagers, que ce soit en Île-de-France ou ailleurs en France, j'utilise quotidiennement les transports en commun. Et, comme beaucoup, je fais face régulièrement à des perturbations sur mes trajets. Ce projet a pour but d'un côté de quantifier et d'analyser ces perturbations, puis de publier ces informations sur des plateformes comme Instagram.D'un autre côté ce projet a pour but de prédire les retards éventuels.

## Collecte des données

La collecte des données s'appuie sur deux principales sources via des API :

L'API de la SNCF pour les données relatives aux TER et TGV. : https://api.sncf.com/v1 
L'API de Navitia pour les transports en commun en Île-de-France. : https://api.navitia.io/v1

Ces APIs offrent un accès à un bon nombre de données sur les réseaux de transports publics en France (Perturbations, trajets, lignes, etc...) . Un script automatisé interroge régulièrement ces APIs pour récupérer les données nécessaires à l'analyse.
