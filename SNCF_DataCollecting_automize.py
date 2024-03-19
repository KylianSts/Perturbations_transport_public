import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from Functions import to_sql_database


def fetch_data_disruptions_tgv_ter(today, yesterday):
    """
    Collecte les données des perturbations de la journée sur les réseaux TGV et TER et remplit une table SQL associée.

    Cette fonction interroge l'API SNCF pour obtenir les données de perturbations affectant les trains TGV et TER pour la période spécifiée. Elle traite la réponse de l'API pour extraire les informations pertinentes sur chaque perturbation, y compris les identifiants, les périodes d'application, les messages de cause, et les détails sur les arrêts impactés. Les données collectées sont ensuite structurées dans un DataFrame pandas et insérées dans une table SQL spécifiée.

    Paramètres :
        today (str) : La date actuelle au format 'YYYYMMDDTHHMMSS'.
        yesterday (str) : La date de la veille au format 'YYYYMMDDTHHMMSS'.
    """

    # URL pour récupérer les données de perturbations pour les réseaux TGV et TER
    url_disruptions_tgv_ter = 'https://api.sncf.com/v1/coverage/sncf/disruptions/'
    # Paramètres pour la requête à l'API
    params = {
        'since': yesterday,  # Date de début pour la récupération des données
        'until': today,  # Date de fin pour la récupération des données
        'start_page': 0,  # Page initiale pour la pagination
    }
    # En-tête d'autorisation avec la clé API
    headers = {'Authorization': os.getenv('api_key_sncf')}

    # Requête API initiale
    response = requests.get(url_disruptions_tgv_ter, headers=headers, params=params)
    data_disruption_tgv_ter = response.json()  # Parse la réponse JSON en dictionnaire

    disruptions_tgv_ter = []  # Liste pour stocker les données de perturbation traitées
    # Calcul du nombre de pages pour la pagination
    nb_page = int(data_disruption_tgv_ter['pagination']['total_result'] / data_disruption_tgv_ter['pagination'][
        'items_per_page']) + 1

    for page in range(1, nb_page + 1):
        if response.status_code == 200:  # Vérifie si la requête à l'API a réussi
            data_disruption_tgv_ter = response.json()  # Rafraîchit les données pour la page actuelle

            for disruption in data_disruption_tgv_ter['disruptions']:  # Itère à travers chaque perturbation
                # Extrait les détails de la perturbation
                id_disruption = disruption['id']
                disruption_start = disruption['application_periods'][0]['begin']

                if disruption_start >= yesterday[:8]:
                    disruption_end = disruption['application_periods'][0]['end']
                    cause_delay = disruption['messages'][0]['text'] if 'messages' in disruption and disruption[
                        'messages'] else None

                    for impacted_object in disruption['impacted_objects']:
                        vehicle_id = impacted_object['pt_object']['id']
                        # Filtrage pour les véhicules de type Train (TGV ou TER)
                        if 'Train' in vehicle_id:
                            train_type = 'TGV' if 'LongDistanceTrain' in vehicle_id else 'TER / Intercités'

                            for impacted_stop in impacted_object.get('impacted_stops', []):
                                # Extraction des informations pour chaque arrêt impacté
                                stop_point = impacted_stop['stop_point']
                                stop_id = stop_point['id']
                                stop_name = stop_point['name']
                                lon = stop_point['coord']['lon']
                                lat = stop_point['coord']['lat']

                                # Construction du dictionnaire des informations de perturbation
                                disruptions_tgv_ter_info = {
                                    'id_disruption': id_disruption,
                                    'id_stop': stop_id,
                                    'name_stop': stop_name,
                                    'lon': lon,
                                    'lat': lat,
                                    'disruption_start': disruption_start,
                                    'disruption_end': disruption_end,
                                    'vehicle_id': vehicle_id,
                                    'train_type': train_type,
                                    'base_arrival_time': impacted_stop.get('base_arrival_time'),
                                    'base_departure_time': impacted_stop.get('base_departure_time'),
                                    'amended_arrival_time': impacted_stop.get('amended_arrival_time'),
                                    'amended_departure_time': impacted_stop.get('amended_departure_time'),
                                    'cause_delay': cause_delay,
                                    'data_date': yesterday[0:8]
                                }
                                disruptions_tgv_ter.append(disruptions_tgv_ter_info)
        else:
            # Logue un message d'erreur si la requête à l'API a échoué
            print(f"Erreur lors de la récupération des données : {response.status_code}")

        params['start_page'] = page
        response = requests.get(url_disruptions_tgv_ter, headers=headers, params=params)

    # Création d'un DataFrame à partir de la liste des perturbations collectées
    df_disruptions_tgv_ter = pd.DataFrame(disruptions_tgv_ter)

    # Identification et suppression des doublons éventuels basés sur l'identifiant de perturbation, l'identifiant de l'arrêt, et l'identifiant du véhicule
    # keep=False marque tous les doublons comme True, et ~ inverse la sélection, supprimant ainsi tous les doublons
    duplicates = df_disruptions_tgv_ter.duplicated(subset=['id_disruption', 'id_stop', 'vehicle_id'], keep='first')
    df_disruptions_tgv_ter = df_disruptions_tgv_ter[~duplicates]

    # Conversion des horaires de base et modifiés en objets timedelta pour permettre des calculs de temps
    # Cela est nécessaire pour convertir les chaînes de caractères formatées en HHMMSS en objets timedelta représentant des durées
    df_disruptions_tgv_ter['base_arrival_time'] = pd.to_timedelta(
        df_disruptions_tgv_ter['base_arrival_time'].str[:2] + ':' + df_disruptions_tgv_ter['base_arrival_time'].str[
                                                                    2:4] + ':' + df_disruptions_tgv_ter[
                                                                                     'base_arrival_time'].str[4:])
    df_disruptions_tgv_ter['base_departure_time'] = pd.to_timedelta(
        df_disruptions_tgv_ter['base_departure_time'].str[:2] + ':' + df_disruptions_tgv_ter['base_departure_time'].str[
                                                                      2:4] + ':' + df_disruptions_tgv_ter[
                                                                                       'base_departure_time'].str[4:])
    df_disruptions_tgv_ter['amended_arrival_time'] = pd.to_timedelta(
        df_disruptions_tgv_ter['amended_arrival_time'].str[:2] + ':' + df_disruptions_tgv_ter[
                                                                           'amended_arrival_time'].str[2:4] + ':' +
        df_disruptions_tgv_ter['amended_arrival_time'].str[4:])
    df_disruptions_tgv_ter['amended_departure_time'] = pd.to_timedelta(
        df_disruptions_tgv_ter['amended_departure_time'].str[:2] + ':' + df_disruptions_tgv_ter[
                                                                             'amended_departure_time'].str[2:4] + ':' +
        df_disruptions_tgv_ter['amended_departure_time'].str[4:])

    # Calcul des retards à l'arrivée et au départ en soustrayant l'heure de base de l'heure modifiée, puis conversion du résultat en minutes
    df_disruptions_tgv_ter['arrival_delay'] = (df_disruptions_tgv_ter['amended_arrival_time'] - df_disruptions_tgv_ter[
        'base_arrival_time']).apply(lambda x: x.total_seconds()) / 60

    df_disruptions_tgv_ter['departure_delay'] = (df_disruptions_tgv_ter['amended_departure_time'] -
                                                 df_disruptions_tgv_ter['base_departure_time']).apply(
        lambda x: x.total_seconds()) / 60

    # Suppression des colonnes d'horaires originales et modifiées, ne conservant que les retards calculés pour simplifier la table finale
    df_disruptions_tgv_ter = df_disruptions_tgv_ter.drop(
        ['base_arrival_time', 'base_departure_time', 'amended_arrival_time', 'amended_departure_time'], axis=1)

    to_sql_database(df_disruptions_tgv_ter, 'SNCF_TGV_TER', 'disruptions_tgv_ter')


def fetch_data_vehicle_journeys_tgv_ter(today, yesterday):
    """
    Collecte les données des voyages des trains TGV et TER entre deux dates et les enregistre dans une table SQL.

    Paramètres :
        today (str) : La date actuelle au format 'YYYYMMDDTHHMMSS'.
        yesterday (str) : La date de la veille au format 'YYYYMMDDTHHMMSS'.

    """

    # URL de base pour l'API des trajets de véhicules (trains) de la SNCF.
    url_vehicle_tgv_ter = 'https://api.sncf.com/v1/coverage/sncf/vehicle_journeys/'

    # Paramètres de la requête API, incluant la période de recherche et l'exclusion de certains modes de transport.
    params = {
        'since': yesterday,  # Début de la période de recherche
        'until': today,  # Fin de la période de recherche
        'start_page': 0,  # Page de départ pour la pagination
        'depth': 3,
        'forbidden_id[]': [  # Modes de transport exclus de la recherche
            'physical_mode:Bus',
            'physical_mode:Coach',
            'physical_mode:RapidTransit',
            'physical_mode:RailShuttle',
            'physical_mode:Tramway'
        ],
    }

    # Headers de la requête, incluant l'autorisation via une clé API stockée dans les variables d'environnement.
    headers = {'Authorization': os.getenv('api_key_sncf')}

    # Première requête à l'API pour récupérer les données.
    response = requests.get(url_vehicle_tgv_ter, headers=headers, params=params)
    data_vehicle_tgv_ter = response.json()

    # Initialisation d'une liste pour stocker les informations sur les trajets des véhicules.
    vehicle_tgv_ter = []

    # Calcul du nombre total de pages à parcourir, basé sur les informations de pagination de la réponse.
    nb_page = int(
        data_vehicle_tgv_ter['pagination']['total_result'] / data_vehicle_tgv_ter['pagination']['items_per_page']) + 1

    # Boucle sur toutes les pages de résultats.
    for page in range(1, nb_page + 1):
        # Vérification si la requête a été réussie.
        if response.status_code == 200:
            # Chargement des données de la page actuelle.
            data_vehicle_tgv_ter = response.json()

            # Extraction et traitement des données de chaque trajet de véhicule.
            for vehicle_journey in data_vehicle_tgv_ter['vehicle_journeys']:
                vehicle_id = vehicle_journey['id']
                route_id = vehicle_journey['journey_pattern']['route']['id']
                time_begin = vehicle_journey['stop_times'][0]['departure_time']
                time_end = vehicle_journey['stop_times'][len(vehicle_journey['stop_times']) - 1]['departure_time']
                train_type = 'Train grande vitesse' if 'LongDistanceTrain' in vehicle_id else 'TER / Intercités'

                # Gestion des identifiants de perturbation, s'il y en a.
                id_disruption = vehicle_journey['disruptions'][0]['id'] if len(
                    vehicle_journey['disruptions']) >= 1 else None

                # Création d'un dictionnaire avec les informations du trajet et ajout à la liste.
                vehicle_tgv_ter_info = {
                    'vehicle_id': vehicle_id,
                    'route_id': route_id,
                    'time_begin': time_begin,
                    'time_end': time_end,
                    'train_type': train_type,
                    'id_disruption': id_disruption,
                    'data_date': yesterday[0:8]
                }
                vehicle_tgv_ter.append(vehicle_tgv_ter_info)
        else:
            # Affichage d'un message d'erreur en cas de problème avec la requête.
            print(f"Erreur lors de la récupération des données: {response.status_code}")

        # Mise à jour des paramètres pour la pagination, passage à la page suivante.
        params['start_page'] = page
        response = requests.get(url_vehicle_tgv_ter, headers=headers, params=params)

    # Conversion de la liste des informations des trajets en DataFrame pour un traitement ultérieur.
    df_vehicle_tgv_ter = pd.DataFrame(vehicle_tgv_ter)

    to_sql_database(df_vehicle_tgv_ter, 'SNCF_TGV_TER', 'vehicle_journeys_tgv_ter')



if __name__ == "__main__":

    today = datetime.now().strftime('%Y%m%d') + 'T000000'
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d') + 'T000000'

    # Collecte des données des trajets de TGV et TER
    fetch_data_disruptions_tgv_ter(today, yesterday)
    fetch_data_vehicle_journeys_tgv_ter(today, yesterday)