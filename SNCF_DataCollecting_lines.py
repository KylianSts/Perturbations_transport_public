import requests
import pandas as pd
import os
from Functions import to_sql_database

def fetch_data_lines_idf():
    """
    Collecte les données des lignes de transport du réseau Île-de-France (IDF) et les stocke dans une base de données SQL.
    """

    # URL de l'API Navitia pour accéder aux données des lignes du réseau IDF.
    url_line_idf = 'https://api.navitia.io/v1/coverage/fr-idf/lines/'

    # Paramètres de la requête pour filtrer les modes de transport et gérer la pagination.
    params = {
        'start_page': 0,  # Page initiale pour la pagination
        'depth': 3,
        'forbidden_uris[]': 'physical_mode:Bus',  # Exclut les bus de la requête
        'forbidden_id[]': [  # Autres modes de transport exclus
            'physical_mode:RailShuttle',
            'physical_mode:Train',
            'physical_mode:Funicular'
        ]
    }

    # Configuration de l'en-tête de la requête, incluant l'autorisation via la clé API.
    headers_idf = {'Authorization': os.getenv('api_key_navitia')}

    # Envoi de la première requête à l'API et récupération de la réponse.
    response = requests.get(url_line_idf, headers=headers_idf, params=params)
    data_line_idf = response.json()

    # Initialisation d'une liste pour accumuler les données des lignes.
    line_idf = []

    # Calcul du nombre total de pages à parcourir basé sur les informations de pagination.
    nb_page = int(data_line_idf['pagination']['total_result'] / data_line_idf['pagination']['items_per_page']) + 1

    # Boucle sur toutes les pages de résultats.
    for page in range(1, nb_page + 1):
        # Itération sur chaque ligne de transport récupérée.
        for line in data_line_idf['lines']:
            # Extraction des données d'intérêt de chaque ligne.
            line_id = line['id']
            line_code = line['code']
            line_color = '#' + line['color']
            vehicle_type = line['physical_modes'][0]['name']
            opening_time = line['opening_time']
            closing_time = line['closing_time']
            network_id = line['network']['id']
            network_name = line['network']['name']

            # Création d'un dictionnaire pour chaque ligne et ajout à la liste.
            lines_idf_info = {
                'line_id': line_id,
                'line_code': line_code,
                'line_color': line_color,
                'vehicle_type': vehicle_type,
                'opening_time': opening_time,
                'closing_time': closing_time,
                'network_id': network_id,
                'network_name': network_name
            }
            line_idf.append(lines_idf_info)

        # Mise à jour des paramètres pour la pagination et requête de la page suivante.
        params['start_page'] = page
        response = requests.get(url_line_idf, headers=headers_idf, params=params)
        data_line_idf = response.json()

    # Conversion de la liste des lignes en DataFrame.
    df_line_idf = pd.DataFrame(line_idf)

    # Nettoyage des identifiants de ligne pour enlever le préfixe et garder uniquement l'identifiant unique.
    df_line_idf['line_id'] = df_line_idf['line_id'].map(lambda x: x.split(':')[-1])

    to_sql_database(df_line_idf, 'SNCF_IDF', 'pt_lines_idf')


def fetch_data_lines_tgv_ter():
    """
    Collecte les données des lignes de train TGV et TER du réseau SNCF et les stocke dans une base de données SQL.
    """

    # URL de base pour l'API SNCF qui fournit les données sur les lignes TGV et TER.
    url_lines_tgv_ter = 'https://api.sncf.com/v1/coverage/sncf/lines/'

    # Paramètres de la requête pour exclure certains modes de transport et initialiser la pagination.
    params = {
        'start_page': 0,  # Démarre la pagination à la page 0.
        'forbidden_id[]': [  # Liste des modes de transport à exclure de la requête.
            'physical_mode:Bus',
            'physical_mode:Coach',
            'physical_mode:RapidTransit',
            'physical_mode:RailShuttle',
            'physical_mode:Tramway',
        ],
    }

    # Configuration des en-têtes de la requête, incluant l'autorisation via la clé API.
    headers = {'Authorization': os.getenv('api_key_sncf')}

    # Effectue la première requête à l'API et stocke la réponse.
    response = requests.get(url_lines_tgv_ter, headers=headers, params=params)
    data_lines_tgv_ter = response.json()

    # Initialise une liste pour stocker les informations sur les lignes TGV et TER.
    lines_tgv_ter = []

    # Calcule le nombre total de pages à partir des informations de pagination fournies par l'API.
    nb_page = int(
        data_lines_tgv_ter['pagination']['total_result'] / data_lines_tgv_ter['pagination']['items_per_page']) + 1

    # Itère sur chaque page de résultats.
    for page in range(1, nb_page + 1):
        # Boucle à travers chaque ligne trouvée dans la réponse de l'API.
        for line in data_lines_tgv_ter['lines']:
            # Détermine le type de train (TGV ou TER/Intercités) à partir du mode physique.
            type_train_id = line['physical_modes'][0]['id']
            if 'LongDistanceTrain' in type_train_id:
                train_type = 'Train grande vitesse'
            else:
                train_type = 'TER / Intercités'

            # Extrait les informations pertinentes de chaque ligne.
            route_id = line['routes'][0]['id']
            route_name = line['routes'][0]['name']
            network_name = line['commercial_mode']['name']

            # Gère les cas où les informations d'ouverture et de fermeture sont disponibles.
            if 'opening_time' in line:
                opening_time = line['opening_time']
                closing_time = line['closing_time']
            else:
                opening_time = None  # Affecte None si les horaires ne sont pas fournis.
                closing_time = None

            # Construit un dictionnaire avec les informations de la ligne et l'ajoute à la liste.
            lines_tgv_ter_info = {
                'route_id': route_id,
                'route_name': route_name,
                'train_type': train_type,
                'network_name': network_name,
                'opening_time': opening_time,
                'closing_time': closing_time
            }
            lines_tgv_ter.append(lines_tgv_ter_info)

        # Met à jour les paramètres pour la prochaine page et refait une requête.
        params['start_page'] = page
        response = requests.get(url_lines_tgv_ter, headers=headers, params=params)
        data_lines_tgv_ter = response.json()

    # Convertit la liste des informations des lignes en DataFrame.
    df_lines_tgv_ter = pd.DataFrame(lines_tgv_ter)

    # Supprime les doublons éventuels basés sur l'identifiant de la route.
    df_lines_tgv_ter = df_lines_tgv_ter.drop_duplicates(subset='route_id')

    to_sql_database(df_lines_tgv_ter, 'SNCF_TGV_TER', 'pt_lines_tgv_ter')


if __name__ == "__main__":

    # Recupération des données sur les lignes
    fetch_data_lines_idf()
    fetch_data_lines_tgv_ter()