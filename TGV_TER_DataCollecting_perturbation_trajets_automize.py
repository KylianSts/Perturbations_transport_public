import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from Functions import fetch_page, to_csv_file
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_data_disruptions_tgv_ter(today, yesterday):
    """
    Collecte les données des perturbations de la journée sur les réseaux TGV et TER et remplit une table SQL associée.

    Cette fonction interroge l'API SNCF pour obtenir les données de perturbations affectant les trains TGV et TER pour la période spécifiée.
    Elle traite la réponse de l'API pour extraire les informations pertinentes sur chaque perturbation,
    y compris les identifiants, les périodes d'application, les messages de cause, et les détails sur les arrêts impactés.
    Les données collectées sont ensuite structurées dans un DataFrame pandas et insérées dans un fichier CSV spécifié.

    Paramètres :
        today (str) : La date actuelle au format 'YYYYMMDDTHHMMSS'.
        yesterday (str) : La date de la veille au format 'YYYYMMDDTHHMMSS'.
    """
    # URL de l'API SNCF pour les perturbations TGV et TER
    url_disruptions_tgv_ter = 'https://api.sncf.com/v1/coverage/sncf/disruptions/'
    # En-tête d'autorisation avec la clé API
    headers = {'Authorization': os.getenv('api_key_sncf')}

    # Liste pour stocker les données de perturbation traitées
    disruptions_tgv_ter = []
    # Numéro de page initial pour la pagination
    page = 0

    # Boucle pour récupérer toutes les pages de données de perturbation
    while True:
        # Paramètres de la requête à l'API avec la date de début, la date de fin et le numéro de page
        params = {
            'since': yesterday,
            'until': today,
            'start_page': page,
        }
        # Requête à l'API pour récupérer les données de perturbation
        response = requests.get(url_disruptions_tgv_ter, headers=headers, params=params)

        # Vérification du code de statut de la réponse
        if response.status_code != 200:
            print(f"Erreur lors de la récupération des données : {response.status_code}")
            break

        # Conversion de la réponse JSON en dictionnaire Python
        data_disruption_tgv_ter = response.json()

        # Sortie de la boucle si aucune perturbation n'est présente dans la réponse ou si toutes les pages ont été récupérées
        if 'disruptions' not in data_disruption_tgv_ter or page >= data_disruption_tgv_ter['pagination'][
            'total_result'] / data_disruption_tgv_ter['pagination']['items_per_page']:
            break

        # Traitement des données de perturbation pour chaque perturbation dans la réponse
        for disruption in data_disruption_tgv_ter['disruptions']:
            id_disruption = disruption['id']
            disruption_start = disruption['application_periods'][0]['begin']

            # Filtrage des perturbations pour celles qui commencent après la date spécifiée
            if disruption_start >= yesterday[:8]:
                disruption_end = disruption['application_periods'][0]['end']
                cause_delay = disruption['messages'][0]['text'] if 'messages' in disruption and disruption[
                    'messages'] else None

                # Extraction des détails de chaque arrêt impacté par la perturbation
                for impacted_object in disruption['impacted_objects']:
                    vehicle_id = impacted_object['pt_object']['id']
                    if 'Train' in vehicle_id:
                        train_type = 'TGV' if 'LongDistanceTrain' in vehicle_id else 'TER / Intercités'

                        for impacted_stop in impacted_object.get('impacted_stops', []):
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
                                'data_date': datetime.strptime(yesterday[0:8], '%Y%m%d').strftime('%d/%m/%Y')
                            }
                            disruptions_tgv_ter.append(disruptions_tgv_ter_info)

        # Incrémentation du numéro de page pour récupérer la page suivante
        page += 1

    # Création d'un DataFrame à partir des données de perturbation collectées
    df_disruptions_tgv_ter = pd.DataFrame(disruptions_tgv_ter)

    # Suppression des doublons basés sur l'identifiant de perturbation, l'identifiant de l'arrêt et l'identifiant du véhicule
    duplicates = df_disruptions_tgv_ter.duplicated(subset=['id_disruption', 'id_stop', 'vehicle_id'], keep='first')
    df_disruptions_tgv_ter = df_disruptions_tgv_ter[~duplicates]

    # Conversion des colonnes d'horaires en objets timedelta pour permettre des calculs de temps
    time_columns = ['base_arrival_time', 'base_departure_time', 'amended_arrival_time', 'amended_departure_time']
    for col in time_columns:
        df_disruptions_tgv_ter[col] = pd.to_timedelta(
            df_disruptions_tgv_ter[col].str[:2] + ':' + df_disruptions_tgv_ter[col].str[2:4] + ':' +
            df_disruptions_tgv_ter[col].str[4:],
            errors='coerce'
        )

    # Calcul des retards à l'arrivée et au départ en minutes
    df_disruptions_tgv_ter['arrival_delay'] = (df_disruptions_tgv_ter['amended_arrival_time'] - df_disruptions_tgv_ter[
        'base_arrival_time']).dt.total_seconds() / 60
    df_disruptions_tgv_ter['departure_delay'] = (df_disruptions_tgv_ter['amended_departure_time'] -
                                                 df_disruptions_tgv_ter['base_departure_time']).dt.total_seconds() / 60

    # Suppression des colonnes d'horaires originales
    df_disruptions_tgv_ter.drop(columns=time_columns, inplace=True)

    df_disruptions_tgv_ter['disruption_end'] = pd.to_datetime(df_disruptions_tgv_ter['disruption_end'].apply(lambda x: x[9:]),
                                                     format="%H%M%S").dt.time

    df_disruptions_tgv_ter['disruption_start'] = pd.to_datetime(df_disruptions_tgv_ter['disruption_start'].apply(lambda x: x[9:]),
                                                       format="%H%M%S").dt.time

    df_disruptions_tgv_ter = df_disruptions_tgv_ter[df_disruptions_tgv_ter['arrival_delay'] >= 0]

    # Appel de la fonction pour écrire les données dans un fichier CSV
    to_csv_file(df_disruptions_tgv_ter, 'TGV_TER_perturbation.csv')


def fetch_data_vehicle_journeys_tgv_ter(today, yesterday):
    # URL de l'API SNCF pour récupérer les informations sur les trajets des véhicules TGV et TER
    url_vehicle_tgv_ter = 'https://api.sncf.com/v1/coverage/sncf/vehicle_journeys/'
    headers = {'Authorization': os.getenv('api_key_sncf')}  # En-tête contenant la clé d'autorisation de l'API
    params = {
        'since': yesterday,
        'until': today,
        'start_page': 0,
        'depth': 3,  # Profondeur de récupération des données
        'forbidden_id[]': [  # ID des modes de transport à exclure
            'physical_mode:Bus',
            'physical_mode:Coach',
            'physical_mode:RapidTransit',
            'physical_mode:RailShuttle',
            'physical_mode:Tramway'
        ],
    }

    # Première requête pour déterminer le nombre total de pages
    initial_response = fetch_page(url_vehicle_tgv_ter, headers, params)
    if initial_response:
        total_result = initial_response['pagination']['total_result']
        items_per_page = initial_response['pagination']['items_per_page']
        nb_page = (total_result // items_per_page) + (1 if total_result % items_per_page else 0)

        vehicle_tgv_ter = []  # Liste pour stocker les informations sur les trajets des véhicules
        futures = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            # Création de tâches asynchrones pour chaque page de résultats
            for page in range(1, nb_page + 1):
                params_copy = params.copy()
                params_copy['start_page'] = page
                futures.append(executor.submit(fetch_page, url_vehicle_tgv_ter, headers, params_copy))

            # Attente de la fin de toutes les tâches asynchrones
            for future in as_completed(futures):
                data_vehicle_tgv_ter = future.result()
                if data_vehicle_tgv_ter and 'vehicle_journeys' in data_vehicle_tgv_ter:
                    # Extraction des informations sur les trajets de chaque véhicule
                    for vehicle_journey in data_vehicle_tgv_ter['vehicle_journeys']:
                        vehicle_info = extract_vehicle_journey_info(vehicle_journey, yesterday)
                        if vehicle_info:
                            vehicle_tgv_ter.append(vehicle_info)

        # Conversion des données en DataFrame Pandas
        df_vehicle_tgv_ter = pd.DataFrame(vehicle_tgv_ter)
        # Écriture des données dans un fichier CSV
        to_csv_file(df_vehicle_tgv_ter, 'TGV_TER_trajet_vehicules.csv')
    else:
        print("Erreur lors de la récupération des informations initiales.")


def extract_vehicle_journey_info(vehicle_journey, yesterday):
    try:
        # Extraction de l'identifiant du véhicule
        vehicle_id = ':'.join(vehicle_journey['id'].split(':')[1:])

        # Extraction de l'identifiant de l'itinéraire
        route_id = vehicle_journey['journey_pattern']['route']['id']

        # Extraction de l'heure de départ du premier arrêt
        time_begin_hhmmss = vehicle_journey['stop_times'][0]['departure_time']

        # Extraction de l'heure de départ du dernier arrêt
        time_end_hhmmss = vehicle_journey['stop_times'][-1]['departure_time']

        # Conversion des heures de format HHMMSS à HH:MM:SS
        time_begin = datetime.strptime(time_begin_hhmmss, "%H%M%S").strftime("%H:%M:%S")
        time_end = datetime.strptime(time_end_hhmmss, "%H%M%S").strftime("%H:%M:%S")

        # Détermination du type de train (TGV ou TER/Intercités)
        train_type = 'Train grande vitesse' if 'LongDistanceTrain' in vehicle_id else 'TER / Intercités'

        # Récupération de l'identifiant de perturbation s'il est disponible, sinon None
        id_disruption = vehicle_journey['disruptions'][0]['id'] if vehicle_journey['disruptions'] else None

        # Création d'un dictionnaire contenant les informations extraites
        return {
            'vehicle_id': vehicle_id,
            'route_id': route_id,
            'time_begin': time_begin,
            'time_end': time_end,
            'train_type': train_type,
            'id_disruption': id_disruption,
            'data_date': datetime.strptime(yesterday[0:8], '%Y%m%d').strftime('%d/%m/%Y')  # Conversion de la date
        }
    except Exception as e:
        # Gestion des erreurs et affichage d'un message en cas de problème
        print(f"Erreur lors de l'extraction des données d'un voyage: {e}")
        return None


if __name__ == "__main__":
*
    today = datetime.now().strftime('%Y%m%d') + 'T000000'
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d') + 'T000000'

    # Collecte des données des trajets de TGV et TER
    fetch_data_disruptions_tgv_ter(today, yesterday)
    fetch_data_vehicle_journeys_tgv_ter(today, yesterday)