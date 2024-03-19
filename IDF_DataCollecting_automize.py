import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import re
from Functions import to_sql_database


def fetch_data_disruptions_idf(today, yesterday):
    """
    Récupère les données sur les perturbations dans le réseau de transport d'Île-de-France (IDF) et les enregistre dans une base de données SQL.

    Cette fonction interroge l'API Navitia pour récupérer des informations sur les perturbations survenues entre deux dates spécifiées (`yesterday` et `today`).
    Elle traite les données reçues pour en extraire les détails pertinents concernant chaque perturbation, puis stocke ces informations dans une table SQL.
    Les perturbations concernant uniquement certains types de transports sont filtrées et seules celles affectant d'autres types que les bus sont conservées.
    En cas d'erreur lors de la récupération des données, un message d'erreur est affiché.

    Args:
        today (str): La date du jour au format YYYYMMDDTHHMMSS.
        yesterday (str): La date de la veille au format YYYYMMDDTHHMMSS.
    """

    # Configuration initiale pour la requête à l'API Navitia
    url_disruptions_idf = 'https://api.navitia.io/v1/coverage/fr-idf/disruptions/'
    params = {
        'since': yesterday,  # Date de début pour la récupération des données
        'until': today,  # Date de fin
        'start_page': 0,  # Page de début pour la pagination des résultats
    }

    headers_idf = {'Authorization': os.getenv('api_key_navitia')}  # Clé d'API pour l'autorisation
    response = requests.get(url_disruptions_idf, headers=headers_idf)  # Première requête pour récupérer les données
    data_disruption_idf = response.json()  # Conversion de la réponse en format JSON

    # Initialisation de la liste pour stocker les informations des perturbations
    disruptions_idf = []

    # Calcul du nombre de pages de résultats basé sur la pagination retournée par l'API
    nb_page = int(
        data_disruption_idf['pagination']['total_result'] / data_disruption_idf['pagination']['items_per_page']) + 1

    # Boucle pour parcourir toutes les pages de résultats
    for page in range(1, nb_page + 1):

        # Vérification du statut de la réponse pour s'assurer que la requête a réussi
        if response.status_code == 200:

            data_disruption_idf = response.json()

            # Extraction et traitement des données de chaque perturbation
            for disruption in data_disruption_idf['disruptions']:

                # Extraction des détails pertinents de la perturbation
                cause_disruption = disruption['cause']

                # Filtre les perturbations de type 'perturbation' et vérifie l'existence d'objets impactés
                if cause_disruption == 'perturbation' and 'impacted_objects' in disruption:

                    # Extraction des détails de l'objet impacté
                    disruption_start = disruption['application_periods'][0]['begin']

                    if disruption_start >= yesterday[:8]:
                        disruption_end = disruption['application_periods'][0]['end']

                        for impacted_object in disruption['impacted_objects']:

                            id_disruption = disruption['disruption_id'] + ":" + impacted_object['pt_object']['id'].split(':')[-1]

                            # Filtre supplémentaire pour les objets de type 'line'
                            if 'line' in impacted_object['pt_object']:

                                # Extraction et traitement des informations spécifiques aux lignes impactées
                                line_code = impacted_object['pt_object']['line']['code']
                                vehicle_type = impacted_object['pt_object']['line']['physical_modes'][0]['name']
                                network_id = impacted_object['pt_object']['line']['network']['id']

                                # Filtre pour exclure les bus des perturbations traitées
                                if vehicle_type != 'Bus':

                                    # Traitement des sections impactées, si présentes
                                    if 'impacted_section' in impacted_object:
                                        impacted_stop_from_id = impacted_object['impacted_section']['from']['id']
                                        impacted_stop_to_id = impacted_object['impacted_section']['to']['id']
                                    else:
                                        impacted_stop_from_id = None
                                        impacted_stop_to_id = None

                                    # Construction d'un dictionnaire avec les informations de la perturbation pour ajout à la liste
                                    disruptions_idf_info = {
                                        'id_disruption': id_disruption,
                                        'disruption_start': disruption_start,
                                        'disruption_end': disruption_end,
                                        'line_code': line_code,
                                        'vehicle_type': vehicle_type,
                                        'network_id': network_id,
                                        'impacted_stop_from_id': impacted_stop_from_id,
                                        'impacted_stop_to_id': impacted_stop_to_id,
                                        'data_date': yesterday[0:8]
                                    }

                                    disruptions_idf.append(disruptions_idf_info)
        else:
            # Affichage d'un message d'erreur si la requête a échoué
            print(f"Erreur lors de la récupération des données: {response.status_code}")

        # Mise à jour du paramètre de pagination pour la requête suivante
        params['start_page'] = page
        response = requests.get(url_disruptions_idf, headers=headers_idf, params=params)

    # Conversion de la liste des perturbations en DataFrame pour traitement ultérieur
    df_disruptions_idf = pd.DataFrame(disruptions_idf)

    # Suppression des doublons basée sur des critères spécifiques
    duplicates = df_disruptions_idf.duplicated(subset=['id_disruption'], keep='first')
    df_disruptions_idf = df_disruptions_idf[~duplicates]

    to_sql_database(df_disruptions_idf, 'SNCF_IDF', 'disruptions_idf')


def fetch_data_vehicle_journey_idf(today, yesterday):
    """
    Collecte les données des voyages de la journée sur le réseau Île-de-France (IDF) et remplit une table SQL associée.

    Cette fonction interroge l'API Navitia pour obtenir les données de voyage dans la région Île-de-France, en excluant certains modes de transport (bus, navette ferroviaire, train et funiculaire). Elle traite la réponse de l'API pour extraire les informations pertinentes sur chaque voyage, y compris les identifiants, les noms, et les identifiants des premiers et derniers arrêts. Elle identifie également les perturbations associées à chaque voyage. Les données collectées sont ensuite structurées dans un DataFrame pandas et insérées dans une table SQL spécifiée.

    Paramètres :
        today (str) : La date actuelle au format 'YYYYMMDDTHHMMSS'.
        yesterday (str) : La date de la veille au format 'YYYYMMDDTHHMMSS'.
    """

    # Point de terminaison de l'API pour récupérer les données des trajets de véhicules dans la région Île-de-France
    url_vehicle_idf = 'https://api.navitia.io/v1/coverage/fr-idf/vehicle_journeys/'

    # Paramètres pour la requête à l'API, incluant la plage de dates et les exclusions de certains modes de transport
    params = {
        'since': yesterday,  # Date de début pour la récupération des données
        'until': today,  # Date de fin pour la récupération des données
        'start_page': 0,  # Page initiale pour la pagination
        'forbidden_uris[]': 'physical_mode:Bus',  # Exclut les trajets en bus
        'forbidden_id[]': [
            'physical_mode:RailShuttle',  # Exclut les navettes ferroviaires
            'physical_mode:Train',  # Exclut les trajets en TER
            'physical_mode:Funicular'  # Exclut les funiculaires
        ]
    }
    # En-tête d'autorisation avec la clé API
    headers_idf = {'Authorization': os.getenv('api_key_navitia')}

    # Requête API initiale
    response = requests.get(url_vehicle_idf, headers=headers_idf, params=params)
    data_vehicle_idf = response.json()  # Parse la réponse JSON en dictionnaire

    vehicle_idf = []  # Liste pour stocker les données de trajet traitées
    # Calcul du nombre de pages pour la pagination basé sur le total des résultats et les éléments par page
    nb_page = int(data_vehicle_idf['pagination']['total_result'] / data_vehicle_idf['pagination']['items_per_page']) + 1

    # Pattern pour matcher les ID du type C00000
    pattern = r"C\d{5}"

    for page in range(1, nb_page + 1):
        if response.status_code == 200:  # Vérifie si la requête à l'API a réussi
            data_vehicle_idf = response.json()  # Rafraîchit les données pour la page actuelle

            for vehicle_journey in data_vehicle_idf['vehicle_journeys']:  # Itère à travers chaque voyage de véhicule
                # Extrait et traite les détails du trajet

                vehicle_id = vehicle_journey['id']
                matches = re.findall(pattern, vehicle_id)
                line_id = matches[0] if matches else None
                vehicle_name = vehicle_journey['name']
                first_stop_id = vehicle_journey['stop_times'][0]['stop_point']['id']
                last_stop_id = vehicle_journey['stop_times'][len(vehicle_journey['stop_times']) - 1]['stop_point']['id']

                # Vérifie l'existence de perturbations et extrait l'identifiant de la première perturbation si présente
                id_disruption = vehicle_journey['disruptions'][0]['id'] + ':' + str(line_id) \
                    if len(vehicle_journey['disruptions']) >= 1 else None

                # Construit un dictionnaire des détails du trajet
                vehicle_idf_info = {
                    'vehicle_id': vehicle_id,
                    'line_id': line_id,
                    'vehicle_name': vehicle_name,
                    'first_stop_id': first_stop_id,
                    'last_stop_id': last_stop_id,
                    'id_disruption': id_disruption,
                    'data_date': yesterday[0:8]
                }

                vehicle_idf.append(vehicle_idf_info)  # Ajoute les détails du trajet à la liste

        else:
            # Logue un message d'erreur si la requête à l'API a échoué
            print(f"Erreur lors de la récupération des données : {response.status_code}")

        # Met à jour le paramètre 'start_page' pour la prochaine requête à l'API
        params['start_page'] = page
        response = requests.get(url_vehicle_idf, headers=headers_idf, params=params)

    # Convertit la liste des détails de trajet en un DataFrame pandas
    df_vehicle_idf = pd.DataFrame(vehicle_idf)

    to_sql_database(df_vehicle_idf, 'SNCF_IDF', 'vehicle_journeys_idf')


if __name__ == "__main__":

    today = datetime.now().strftime('%Y%m%d') + 'T000000'
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d') + 'T000000'

    # Collecte des données du réseau idf
    fetch_data_disruptions_idf(today, yesterday)
    fetch_data_vehicle_journey_idf(today, yesterday)