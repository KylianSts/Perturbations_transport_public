import pandas as pd
from pandas import DataFrame
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, time
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import ast
from streamlit_option_menu import option_menu
import re
import numpy as np

st.set_page_config(layout="wide")
style_metric_cards()

INCIVILITE = [
    "Acte de vandalisme",
    "Présence d'un bagage abandonné",
    "Présence d'individus sur les voies",
    "Incendie aux abords de la voie",
    "Intervention des forces de l'ordre",
    "Présence de manifestants sur les voies",
    "Agression d'un agent SNCF"
]

PROBLEMES_TECHNIQUES = [
    "Défaillance de matériel",
    "Panne de signalisation",
    "Panne d'un passage à niveau",
    "Panne d'un aiguillage",
    "Défaut d'alimentation électrique",
    "Indisponibilité d'un matériel",
    "Dérangement d'une installation en gare",
    "Modification de matériel",
    "Incident technique sur la voie",
    "Dérangement d'un équipement technique ou informatique"
]

CONDITIONS_OPERATIONNELLES = [
    "Régulation du trafic",
    "Réutilisation d'un train",
    "Difficultés lors de la préparation du train",
    "Prise en charge de clients en correspondance",
    "Conditions de départ non réunies",
    "Mise à quai tardive en gare origine",
    "Arrêt exceptionnel en gare",
    "Saturation des voies en gare",
    "Confirmation tardive de la voie",
    "Erreur d'itinéraire",
]

CONDITIONS_EXTERNES = [
    "Travaux sur les voies",
    "Obstacle sur la voie",
    "Incident de circulation",
    "Incident sur un réseau ferré étranger",
    "Accident de personne",
    "Conditions météorologiques",
    "Présence d'animaux sur la voie",
    "Embouteillage",
    "Incident ferroviaire",
    "Accident à un passage à niveau",
    "Heurt d'un animal",
    "Déclenchement du signal d'alarme",
    "Choc nécessitant une vérification technique sur le train"
]

ASSISTANCE_PASSAGERS = [
    "Assistance à un voyageur malade à bord",
    "Assistance à un voyageur",
    "Affluence de voyageurs entraînant un arrêt prolongé"
]

############################################################################################
def date_range(start: datetime, end: datetime) -> List[datetime]:
    return [(start + timedelta(days=i)).strftime('%d/%m/%Y') for i in range((end - start).days + 1)]
def collect_data_week(start: datetime, end: datetime, *dfs: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Filtre les DataFrames pour inclure uniquement les données des dates spécifiées.

    Args:
        week_start_date (datetime): Date de début de la semaine.
        week_end_date (datetime): Date de fin de la semaine.
        *dfs (DataFrame): Un ou plusieurs DataFrames à filtrer.

    Returns:
        Tuple[DataFrame, ...]: DataFrames filtrés.
    """
    list_date = date_range(start, end)
    return tuple(df[df['data_date'].isin(list_date)] for df in dfs)
def define_type_causes(cause: str) -> str:
    if cause in INCIVILITE:
        return 'incivilite'
    elif cause in PROBLEMES_TECHNIQUES:
        return 'problemes techniques'
    elif cause in CONDITIONS_OPERATIONNELLES:
        return 'conditions operationnelles'
    elif cause in CONDITIONS_EXTERNES:
        return 'conditions externes'
    elif cause in ASSISTANCE_PASSAGERS:
        return 'assistance passagers'
    else:
        return 'Non Connu'
@st.cache_data
def load_disruption_data(filepath: str) -> pd.DataFrame:
    """
    Charge et nettoie les données de perturbation.

    Args:
    filepath (str): Chemin du fichier CSV contenant les données de perturbation.

    Returns:
    DataFrame: DataFrame nettoyée et prête à être utilisée pour l'analyse.
    """
    # Charge le fichier CSV dans un DataFrame
    df = pd.read_csv(filepath)

    # Convertit les colonnes 'disruption_start' et 'disruption_end' en objets datetime
    df['disruption_start'] = pd.to_datetime(df['disruption_start'], format='%H:%M:%S')
    df['disruption_end'] = pd.to_datetime(df['disruption_end'], format='%H:%M:%S')

    # Calcule la durée de chaque perturbation en soustrayant 'disruption_start' de 'disruption_end'
    df['duration_disruption'] = df['disruption_end'] - df['disruption_start']

    # Supprime les lignes où la durée de perturbation est négative (par précaution)
    df = df[~(df['duration_disruption'] < pd.Timedelta(0))]

    # Remplace les valeurs manquantes dans la colonne 'cause_delay' par 'Non connu'
    df['cause_delay'] = df['cause_delay'].fillna('Non connu')

    # Applique la fonction define_type_causes pour déterminer et assigner le groupe de retard ('group_delay')
    df['group_delay'] = df['cause_delay'].apply(define_type_causes)

    return df
@st.cache_data
def load_and_clean_data() -> Tuple[pd.DataFrame]:
    """
    Fonction qui charge toutes les bases utilisées et les prépare pour l'analyse.
    """
    # Détermine la semaine en cours


    ########################################
    # Chargement des données de perturbation et des véhicules de trajet
    df_disruption = load_disruption_data("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\\TGV_TER_perturbation.csv")
    df_vehicle = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\\TGV_TER_trajet_vehicules.csv")

    # Chargement des informations sur les lignes, filtrant NightJet
    df_lines = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\\TGV_TER_lines_info.csv")
    df_lines = df_lines[df_lines['network_name'] != 'NightJet']
    df_lines = df_lines[df_lines['network_name'] != 'Intercités de nuit']

    # Suppression des doublons dans les perturbations uniques
    df_disruption_unique = df_disruption.drop_duplicates(['id_disruption', 'vehicle_id'])

    # Fusion des DataFrames des véhicules et des perturbations
    df_merge_inter = pd.merge(df_vehicle, df_disruption.drop(['data_date', 'train_type'], axis=1),
                              on=['vehicle_id', 'id_disruption'], how='left')
    df_merge = pd.merge(df_merge_inter, df_lines.drop('train_type', axis=1), on='route_id', how='left')

    # Suppression des doublons de la colonne 'vehicle_id' dans la DataFrame fusionnée
    df_merge = df_merge[~df_merge.duplicated(subset=['vehicle_id'], keep='first')]
    ########################################


    ########################################
    # Lecture du fichier des résultats précédents
    df_past = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Rapport\\Resultats_rapport.csv", sep=",")

    df_ref = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\Base_SNCF\\referentiel-gares-voyageurs.csv", sep=";")
    df_ref = df_ref[
        ['Code UIC', 'Code postal',
         'Code Commune', 'Commune', 'Code département', 'Département',
         'Longitude', 'Latitude', 'DTG', 'Région SNCF',
         'Unité gare', 'UT']
    ]
    # Chargement des fichiers permetant de créer uen base avec la liste des gare et leurs fréquentation
    df_freq = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\Base_SNCF\\frequentation-gares.csv",
                          sep=";", usecols=[0, 1, 2])

    df_pop = pd.read_excel(
        "C:\\Users\\Kyliv\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\\Base_INSEE\\base-pop-historiques-1876-2021.xlsx",
        skiprows=5, usecols=[0, 1, 2, 3])

    df_region_name = pd.read_csv("C:\\Users\\Kyliv\\DATA_PROJECT\\TGV_TER_PRODUCT\\Data\\Base_INSEE\\region_2022.csv")

    df_merge_inter = pd.merge(df_ref, df_freq.drop(['Code postal'], axis=1), on='Code UIC')
    df_merge_inter['Code Commune'].fillna(0, inplace=True)

    def format_code_commune(x):
        x = int(x)
        if x < 10:
            return '00' + str(x)
        elif x < 100:
            return '0' + str(x)
        else:
            return str(x)

    df_merge_inter['CODGEO'] = df_merge_inter.apply(
        lambda row: str(row['Code postal'])[:2] + format_code_commune(row['Code Commune']), axis=1)

    df_sp = pd.merge(df_merge_inter.drop(['Code postal', 'Code Commune'], axis=1),
                     df_pop.drop(['DEP', 'LIBGEO'], axis=1), on='CODGEO', how='left')

    df_sp = pd.merge(df_sp, df_region_name[['REG', 'LIBELLE']], on='REG', how='left')
    ########################################

    return df_disruption, df_vehicle, df_lines, df_disruption_unique, df_merge, df_past, df_ref, df_sp
@st.cache_data
def get_data_between_hours(start_time: time, end_time: time, *dfs: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Filtre les DataFrames pour inclure uniquement les données entre les heures spécifiées.

    Args:
        start_time (time): Heure de début.
        end_time (time): Heure de fin.
        *dfs (DataFrame): Un ou plusieurs DataFrames à filtrer.

    Returns:
        Tuple[DataFrame, ...]: DataFrames filtrés.
    """
    filtered_dfs = []
    for df in dfs:
        # Conversion des colonnes 'time_begin' et 'time_end' en objets de type time
        df['time_begin'] = df['time_begin'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
        df['time_end'] = df['time_end'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())

        # Filtrage des données en fonction de l'intervalle de temps sélectionné
        filtered_df = df[(df['time_begin'] >= start_time) & (df['time_end'] <= end_time)]
        filtered_dfs.append(filtered_df)

    return tuple(filtered_dfs)
@st.cache_data
def get_data_selected_date(*dfs:pd.DataFrame, selected_dates: Tuple[datetime]=None,
                           start_time: time, end_time: time) -> Tuple[pd.DataFrame]:
    if selected_dates is None:
        week_start_date = datetime.now() - timedelta(days=datetime.now().weekday()) - timedelta(days=7)
        week_end_date = datetime.now() + timedelta(days=(-1 - datetime.now().weekday()))
        return collect_data_week(week_start_date, week_end_date, *dfs)
    else:
        start, end = selected_dates
        dfs = collect_data_week(start, end, *dfs)
        filtered_df = []
        for df in dfs:
            filtered_df.append(get_data_between_hours(start_time, end_time, df)[0])

        return tuple(filtered_df)

def get_data_selected_date2(*dfs:pd.DataFrame, selected_dates: Tuple[datetime]=None) -> Tuple[DataFrame]:
    if selected_dates is None:
        week_start_date = datetime.now() - timedelta(days=datetime.now().weekday()) - timedelta(days=7)
        week_end_date = datetime.now() + timedelta(days=(-1 - datetime.now().weekday()))
        return collect_data_week(week_start_date, week_end_date, *dfs)
    else:
        start, end = selected_dates
        return collect_data_week(start, end, *dfs)
def function_comparaison_day(df: pd.DataFrame, day: int) -> Tuple[List, List]:
    """
    Extracts values and corresponding dates for a specific day from a DataFrame and a present value.

    Args:
    - df: DataFrame containing daily values.
    - present_value: Series containing the present value data.
    - day: Index of the day to extract (0 for the first day, 1 for the second, etc.).

    Returns:
    - Tuple containing two lists:
      - List of values extracted for each DataFrame and the present value.
      - List of corresponding dates for the extracted values.
    """
    L = []       # List to store values for each DataFrame and present value
    L_date = []  # List to store dates for each value extracted

    # Iterate over the indices of the DataFrame passed as argument
    for index in df.index:
        # Convert JSON string to pandas Series
        data_series = pd.Series(ast.literal_eval(df.loc[index]))

        # Find the index of the specific day in the series sorted by date
        ind = pd.to_datetime(data_series.reset_index()['index'], format='%d/%m/%Y').sort_values().index[day]

        # Add the value to the L list
        L.append(data_series[ind])

        # Add the corresponding date to the L_date list
        L_date.append(data_series.index[ind])

    # Return the lists L and L_date as a tuple
    return L, L_date
def seconds_to_minutes(seconds: int | float) -> str:
    """
    Convertit un temps donné en minutes en une chaîne de caractères représentant ce temps en heures et minutes.

    Paramètres:
    minutes (float): Le nombre total de minutes à convertir.

    Retourne:
    str: Une chaîne de caractères au format 'XhYY', où X est le nombre d'heures et YY est le nombre de minutes restantes.
    """
    # Calculer le nombre d'heures entières en divisant les minutes par 60
    seconds= int(seconds)
    minutes = seconds // 60

    # Calculer les minutes restantes après avoir extrait les heures entières
    remaining_seconds = seconds % 60

    # Retourner une chaîne de caractères formatée sous la forme 'XhYY'
    # '02d' garantit que les minutes seront affichées avec deux chiffres, par exemple '5' devient '05'
    return f"{minutes}m{remaining_seconds:02d}"
def minutes_to_hours(minutes: int | float) -> str:
    """
    Convertit un temps donné en minutes en une chaîne de caractères représentant ce temps en heures et minutes.

    Paramètres:
    minutes (float): Le nombre total de minutes à convertir.

    Retourne:
    str: Une chaîne de caractères au format 'XhYY', où X est le nombre d'heures et YY est le nombre de minutes restantes.
    """
    # Calculer le nombre d'heures entières en divisant les minutes par 60
    minutes= int(minutes)
    hours = minutes // 60

    # Calculer les minutes restantes après avoir extrait les heures entières
    remaining_minutes = minutes % 60

    # Retourner une chaîne de caractères formatée sous la forme 'XhYY'
    # '02d' garantit que les minutes seront affichées avec deux chiffres, par exemple '5' devient '05'
    return f"{hours}h{remaining_minutes:02d}"
def hours_to_days(hours: int | float) -> str:
    """
    Convertit une durée en heures en une chaîne de caractères au format 'XjYYh' où X est le nombre de jours
    et YY est le nombre d'heures restant.

    Args:
        hours (float): La durée en heures à convertir.

    Returns:
        str: La durée convertie sous forme de chaîne de caractères au format 'XjYYh'.
    """
    # Calcul du nombre de jours entiers en divisant les heures par 24
    hours = int(hours)
    days = hours // 24

    # Calcul du nombre d'heures restantes en prenant le reste de la division des heures par 24
    remaining_hours = hours % 24

    # Retourne la chaîne de caractères formatée avec le nombre de jours et le nombre d'heures restantes
    return f"{days}j{remaining_hours:02d}h"
def interval_hour() -> Tuple[List[datetime.time], List[str]]:
    """
    Génère des intervalles de 10 minutes pour une journée et des noms d'intervalles horaires.

    Returns:
        Tuple[List[datetime], List[str]]: Une liste de temps à intervalles de 10 minutes et une liste de noms d'intervalles horaires.
    """
    # Début et fin de la période (une journée entière)
    start_time = datetime(2023, 1, 1, 0, 0)
    end_time = datetime(2023, 1, 2, 0, 0)

    # Liste pour stocker les temps à intervalles de 10 minutes
    ten_intervals = []
    current_time = start_time
    while current_time < end_time:
        ten_intervals.append(current_time.time())
        current_time += timedelta(minutes=10)

    # Liste pour stocker les noms des intervalles horaires
    hours_intervals_name = [''] * len(ten_intervals)

    # Remplir les noms des intervalles horaires toutes les heures
    for i in range(0, len(ten_intervals), 6):  # 6 intervalles de 10 minutes par heure
        hours_intervals_name[i] = ten_intervals[i].strftime('%H:%M')

    # Assurer que le premier et dernier intervalle sont correctement nommés
    hours_intervals_name[0] = '00:00'
    hours_intervals_name[-1] = '23:59'

    return ten_intervals, hours_intervals_name
def disruption_across_day(ten_intervals: List[datetime.time], *dfs: pd.DataFrame) -> List[List[int]] | List[int]:
    """
    Calcule le nombre de perturbations pour chaque intervalle de 10 minutes au cours d'une journée pour plusieurs DataFrames.

    Args:
        ten_intervals (List[datetime.time]): Liste des temps à intervalles de 10 minutes.
        dfs (pd.DataFrame): DataFrames contenant les colonnes 'disruption_start' et 'disruption_end' avec les temps de début et de fin des perturbations.

    Returns:
        Tuple[List[int]]: Tuple contenant une liste pour chaque DataFrame avec le nombre de perturbations pour chaque intervalle de 10 minutes.
    """
    list_disruption_by_ten = []

    for df in dfs:
        start_times = [t.time() for t in df['disruption_start']]
        end_times = [t.time() for t in df['disruption_end']]

        disruption_by_ten = []

        for interval in ten_intervals:
            nb_disruption = sum(start < interval < end for start, end in zip(start_times, end_times))
            disruption_by_ten.append(nb_disruption)

        list_disruption_by_ten.append(disruption_by_ten)

    return list_disruption_by_ten if len(dfs) > 1 else disruption_by_ten
def get_UIC():
    None
def metrics_disruprion(df_past:pd.DataFrame, value:pd.Series, value_name:str, start:datetime, end:datetime,
                       is_default:bool = True, label_name:str = None, add_text:str = None,
                       delta_color:str='inverse', type_text:str = None) -> None:
    if type_text == 'pourcentage':
        annotation_text = f'{value:.2f}%'
    elif type_text == 'minutes':
        annotation_text = minutes_to_hours(value)
    elif type_text == 'heures':
        annotation_text = hours_to_days(value)
    else:
        annotation_text = f'{value:,}'
    if is_default:
        delta_value = f"{float(value / df_past[value_name][len(df_past) - 2] - 1) * 100:.2f}%"
        if add_text is None:
            add_text = 'sur la semaine'
    else:
        delta_value = None
        if add_text is None:
            add_text = f'du {start} au {end}'
    st.metric(
        label=f'{label_name} {add_text}',
        value=annotation_text,
        delta=delta_value,
        delta_color=delta_color
    )
############################################################################################




##################### FONCTIONS DE GRAPH ##########################################################
def grah_line_chart(values_list: List[pd.Series], start: datetime, end: datetime, days: List[str],
                    colors: Optional[List[str]] = None, title: Optional[str] = None, ytitle: Optional[str] = None,
                    label_names: Optional[List[str]] = None, alpha_list: Optional[List[float]] = None,
                    type_text: Optional[str] = None, show_legend: bool = False,
                    show_values: Optional[List[bool]] = None,
                    height: int = 400, width: int = 700, is_default: bool = True) -> None:
    # Configuration des paramètres par défaut
    if is_default:
        marker_size = 10
        xaxis_dict = dict(
            tickmode='array',
            tickvals=list(range(len(days))),
            ticktext=days,
            tickfont=dict(size=14)
        )
        text_title = ' sur la semaine'
    else:
        if len(values_list[0]) > 7:
            show_values = [False] * len(values_list)
            marker_size = 0
        else:
            marker_size = 10
        xaxis_dict = None
        text_title = ''

    # Calcul du maximum de l'axe y pour ajuster l'échelle
    y_max = max([series.max() * 1.2 for series in values_list])

    # Configuration de l'opacité par défaut si non spécifiée
    if alpha_list is None:
        alpha_list = [0.7] * len(values_list)

    # Configuration de l'affichage des valeurs par défaut si non spécifiée
    if show_values is None:
        show_values = [True] * len(values_list)

    # Initialisation de la figure Plotly
    fig = go.Figure(layout=dict(
        title=dict(
            text=f'{title}{text_title} <br> du {start} au {end}',
            x=0.5,
            xanchor='center'
        )
    ))

    # Ajout des séries de données à la figure
    for i in range(len(values_list)):
        df_value = values_list[i].reset_index()
        df_value['data_date'] = pd.to_datetime(df_value['data_date'], format="%d/%m/%Y")
        df_value = df_value.sort_values(by='data_date')
        df_value['data_date'] = df_value['data_date'].dt.strftime("%d/%m/%Y")

        fig.add_trace(go.Scatter(
            x=df_value.iloc[:, 0], y=df_value.iloc[:, -1],
            mode='lines+markers',
            name=label_names[i] if label_names is not None else f'Série {i + 1}',
            line=dict(color=colors[i] if colors else None, width=4),
            marker=dict(color=colors[i], size=marker_size),
            opacity=alpha_list[i]
        ))

        # Ajout des annotations de valeurs sur le graphique si nécessaire
        if show_values[i]:
            for j, v in enumerate(values_list[i]):
                if type_text == 'pourcentage':
                    annotation_text = f'{v:.2f}%'
                elif type_text == 'minutes':
                    annotation_text = minutes_to_hours(v)
                elif type_text == 'heures':
                    annotation_text = hours_to_days(v)
                else:
                    annotation_text = f'{v}'

                fig.add_annotation(
                    x=j, y=v * 1.05,
                    text=annotation_text,
                    showarrow=False,
                    align='center',
                    valign='bottom'
                )

    # Personnalisation du layout du graphique
    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=dict(
            title=ytitle,
            titlefont_size=15,
            tickfont_size=14,
            range=[0, y_max]
        ),
        plot_bgcolor='white',
        showlegend=show_legend,
        height=height,
        width=width,
        font=dict(family='Arial', size=13, color='black')
    )

    # Affichage du graphique interactif dans Streamlit
    st.plotly_chart(fig)
def graph_line_chart_comparaison_day(list_dfs: List[pd.DataFrame], days: List[str], key_prefix: Optional[str] = None,
                                     colors: Optional[List[str]] = None, title: Optional[str] = None,
                                     ytitle: Optional[str] = None, type_text: Optional[str] = None,
                                     alpha_list: Optional[List[float]] = None, show_values: Optional[List[bool]] = None,
                                     label_names: Optional[List[str]] = None, show_legend: bool = False,
                                     height: int = 400, width: int = 700) -> None:
    # Initialisation des listes d'opacité et d'affichage des valeurs si non spécifiées
    if alpha_list is None:
        alpha_list = [0.7] * len(list_dfs)

    if show_values is None:
        show_values = [True] * len(list_dfs)

    # Fonction pour mettre à jour le graphique en fonction du jour sélectionné
    def update_graph(day: int) -> None:
        fig = go.Figure(layout=dict(
            title=dict(
                text=f'{title} <br> Comparaison des {days[day]}s',
                x=0.5,
                xanchor='center'
            )
        ))

        y_max = 0  # Initialisation de la valeur maximale de l'axe y

        for i in range(len(list_dfs)):
            # Extraction des données pour un jour spécifique
            list_one_day, list_date = function_comparaison_day(list_dfs[i], day)

            # Mise à jour de la valeur maximale de l'axe y
            y_max = max(list_one_day) * 1.2 if max(list_one_day) * 1.2 > y_max else y_max

            # Ajout de la série de données au graphique
            fig.add_trace(go.Scatter(
                x=list_date, y=list_one_day,
                mode='lines+markers',
                name=label_names[i] if label_names else f'Série {i + 1}',
                line=dict(color=colors[i] if colors else None, width=4),
                marker=dict(color=colors[i], size=10),
                opacity=alpha_list[i]
            ))

            # Ajout des annotations de valeurs sur le graphique si nécessaire
            if show_values[i]:
                for j, v in enumerate(list_one_day):
                    if type_text == 'pourcentage':
                        annotation_text = f'{v:.2f}%'
                    elif type_text == 'minutes':
                        annotation_text = minutes_to_hours(v)
                    elif type_text == 'heures':
                        annotation_text = hours_to_days(v)
                    else:
                        annotation_text = f'{v}'

                    fig.add_annotation(
                        x=j, y=v * 1.05,
                        text=annotation_text,
                        showarrow=False,
                        align='center',
                        valign='bottom'
                    )

        # Personnalisation du layout du graphique
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(list_date))),
                ticktext=list_date,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=ytitle,
                titlefont_size=15,
                tickfont_size=14,
                range=[0, y_max]
            ),
            plot_bgcolor='white',
            showlegend=show_legend,
            height=height,
            width=width,
            font=dict(family='Arial', size=13, color='black')
        )

        # Affichage du graphique interactif dans Streamlit
        st.plotly_chart(fig)

    # Interface utilisateur avec Streamlit pour sélectionner le jour
    day = st.selectbox('', options=days, index=0, key=f'{key_prefix}_day_selectbox')
    update_graph(days.index(day))
def graph_line_chart_comparaison_week(dates: List[pd.DataFrame], values_list: List[pd.Series], first_data_date: str,
                                      colors: Optional[List[str]] = None, title: Optional[str] = None,
                                      ytitle: Optional[str] = None, type_text: Optional[str] = None,
                                      alpha_list: Optional[List[float]] = None, show_values: Optional[List[bool]] = None,
                                      label_names: Optional[List[str]] = None, show_legend: bool = False,
                                      height: int = 400, width: int = 700) -> None:
    # Calcul de la valeur maximale de l'axe y pour ajuster l'échelle
    y_max = max([series.max() * 1.2 for series in values_list])

    # Initialisation des listes d'opacité et d'affichage des valeurs si non spécifiées
    if alpha_list is None:
        alpha_list = [0.7] * len(values_list)

    if show_values is None:
        show_values = [True] * len(values_list)

    # Détermination du texte du titre en fonction de sa longueur
    if len(title) > 20:
        text = f'<br> Suivi par semaine depuis le {first_data_date}'
    else:
        text = f' - Suivi par semaine <br> depuis le {first_data_date}'

    # Initialisation de la figure Plotly avec le titre
    fig = go.Figure(layout=dict(
        title=dict(
            text=f'{title}{text}',
            x=0.5,
            xanchor='center'
        )
    ))

    # Ajout de chaque série de données au graphique
    for i in range(len(values_list)):
        fig.add_trace(go.Scatter(
            x=values_list[i].index, y=values_list[i].values,
            mode='lines+markers',
            name=label_names[i] if label_names else f'Série {i + 1}',
            line=dict(color=colors[i] if colors else None, width=4),
            marker=dict(color=colors[i], size=10),
            opacity=alpha_list[i]
        ))

        # Ajout des annotations de valeurs sur le graphique si nécessaire
        if show_values[i]:
            for j, v in enumerate(values_list[i]):
                if type_text == 'pourcentage':
                    annotation_text = f'{v:.2f}%'
                elif type_text == 'minutes':
                    annotation_text = minutes_to_hours(v)
                elif type_text == 'heures':
                    annotation_text = hours_to_days(v)
                else:
                    annotation_text = f'{v}'

                fig.add_annotation(
                    x=j, y=v * 1.05,
                    text=annotation_text,
                    showarrow=False,
                    align='center',
                    valign='bottom'
                )

    # Personnalisation du layout du graphique
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(dates))),
            ticktext=dates,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=ytitle,
            titlefont_size=15,
            tickfont_size=14,
            range=[0, y_max]
        ),
        plot_bgcolor='white',
        showlegend=show_legend,
        height=height,
        width=width,
        font=dict(family='Arial', size=13, color='black')
    )

    # Affichage du graphique interactif dans Streamlit
    st.plotly_chart(fig)
def graph_network(info_current: List[pd.Series], info_name: List[str], values: pd.Series,
                  start_date: datetime, end_date: datetime, start_time: time, end_time: time,
                  title: Optional[str] = None, colors: Optional[List[str]] = None) -> None:
    # Définir les couleurs par défaut si aucune couleur n'est fournie
    if colors is None:
        colors = ["royalblue"] * len(values)

    def update_graph(selected_values: pd.Series) -> None:
        # Modifier les couleurs pour les valeurs maximales et minimales
        if max(values) in selected_values.values:
            colors[len(selected_values) - 1] = 'gold'
        if min(values) in selected_values.values:
            colors[0] = 'skyblue'

        # Ajuster les valeurs sélectionnées et les trier
        selected_values = (selected_values + 70) * 0.1
        selected_values.sort_values(ascending=True, inplace=True)

        # Initialisation de la figure Plotly avec le titre
        fig = go.Figure(layout=dict(
            title=dict(
                text=f'{title} <br> du {start_date} au {end_date} entre {str(start_time)[:5]} et {str(end_time)[:5]}' if title is not None else None,
                x=0.5,
                xanchor='center'
            )
        ))

        max_networks = len(values)

        # Ajout des barres horizontales
        fig.add_trace(go.Bar(
            x=selected_values.values,
            y=selected_values.index,
            orientation='h',
            marker=dict(color=colors[:len(selected_values)]),
            width=0.05 / (max_networks - len(selected_values)) if max_networks != len(selected_values) else 0.05,
        ))

        # Ajout des annotations et des cercles pour chaque valeur sélectionnée
        for i, (index, value) in enumerate(selected_values.items()):

            # Ajouter une annotation sur la barre principale
            fig.add_annotation(
                x=value, y=i,
                text=f'<b>{selected_values[i] * 10 - 70:.1f}%</b>',
                showarrow=False,
                font=dict(size=16, color='black'),
                align='center',
                valign='middle'
            )

            # Ajouter une annotation avec le nom du réseau
            fig.add_annotation(
                x=0 + len(index) / 30, y=i + 0.3,
                text=f'<b>{index}</b>',
                showarrow=False,
                font=dict(size=20, color='black'),
                align='left',
                valign='middle'
            )

            # Ajouter un cercle blanc sur la barre principale
            fig.add_trace(go.Scatter(
                x=[value],
                y=[index],
                mode='markers',
                marker=dict(size=60, color='white', line=dict(color='black', width=1))
            ))

            # Ajouter des cercles blancs et des annotations pour les informations courantes
            for j, interval in enumerate([2 / 8, 4 / 8, 6 / 8]):
                circle_x = value * interval

                fig.add_trace(go.Scatter(
                    x=[circle_x],
                    y=[index],
                    mode='markers',
                    marker=dict(size=15, color='white', line=dict(color='black', width=1))
                ))

                current_info_value = info_current[j][index]
                fig.add_annotation(
                    x=circle_x, y=i - 0.3,
                    text=f'<b>{current_info_value}</b>',
                    showarrow=False,
                    font=dict(size=13, color='black'),
                    align='center',
                    valign='middle'
                )

            # Ajouter des cercles colorés pour les intervalles
            for interval in [1 / 8, 3 / 8, 5 / 8, 7 / 8]:
                circle_x = value * interval

                fig.add_trace(go.Scatter(
                    x=[circle_x],
                    y=[index],
                    mode='markers',
                    marker=dict(size=15, color=colors[i], line=dict(color='black', width=0))
                ))

        # Mise en forme de la figure
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=100 + 130 * len(selected_values),
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
            showlegend=False
        )

        # Affichage du graphique interactif dans Streamlit
        st.plotly_chart(fig)

    def legend(selected_values: pd.Series, info_name: List[str]) -> None:
        # Initialisation de la figure pour la légende
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=selected_values.values,
            y=selected_values.index,
            orientation='h',
            marker=dict(color='gray'),
            width=0.03,
        ))

        fig.add_annotation(
            x=selected_values[0], y=0,
            text='<b>% de trajet<br>perturbé</b>',
            showarrow=False,
            font=dict(size=12, color='black'),
            align='center',
            valign='middle'
        )

        # Ajouter un cercle blanc
        fig.add_trace(go.Scatter(
            x=[selected_values[0]],
            y=[0],
            mode='markers',
            marker=dict(size=70, color='white', line=dict(color='black', width=1))
        ))

        # Ajouter des cercles blancs et des annotations pour la légende
        for j, interval in enumerate([2 / 8, 4 / 8, 6 / 8]):
            circle_x = selected_values[0] * interval

            fig.add_trace(go.Scatter(
                x=[circle_x],
                y=[0],
                mode='markers',
                marker=dict(size=15, color='white', line=dict(color='black', width=1))
            ))

            fig.add_annotation(
                x=circle_x, y=-0.2,
                text=f'<b>{info_name[j]}</b>',
                showarrow=False,
                font=dict(size=13, color='black'),
                align='center',
                valign='middle'
            )

        # Ajouter des cercles colorés pour les intervalles
        for interval in [1 / 8, 3 / 8, 5 / 8, 7 / 8]:
            circle_x = selected_values[0] * interval

            fig.add_trace(go.Scatter(
                x=[circle_x],
                y=[0],
                mode='markers',
                marker=dict(size=15, color='gray', line=dict(color='black', width=0))
            ))

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300,
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
            showlegend=False
        )

        # Affichage de la légende dans Streamlit
        st.plotly_chart(fig)

    # Interface utilisateur pour sélectionner les réseaux
    selected_network = st.multiselect('Sélectionnez des réseaux', values.index, default=list(values.index)[0:5])

    # Affichage de la légende sous un expander
    with st.expander('Legend', expanded=False):
        legend(pd.Series(values[selected_network][0]), info_name)

    # Mise à jour du graphique avec les valeurs sélectionnées
    update_graph(pd.Series(values[selected_network]))
def graph_bar_chart_gare(df: pd.DataFrame, start: datetime, end: datetime) -> None:
    # Cette fonction met à jour les données en fonction de l'échelle géographique sélectionnée.
    def update_graph_scale(scale_name: str, list_values_names: List[str]) -> pd.DataFrame:
        # Création d'un dictionnaire pour l'agrégation des colonnes
        agg_dict = {value_name: 'sum' for value_name in list_values_names if 'mean' or 'avg' not in value_name}
        agg_dict.update({value_name: 'mean' for value_name in list_values_names if 'mean' or 'avg' in value_name})

        # Agrégation des données en fonction de l'échelle géographique sélectionnée
        if scale_name == 'Gare':
            agg_dict.update({'Gare': 'first'})
            data = df.groupby('Gare').agg(agg_dict)
        elif scale_name == 'Commune':
            agg_dict.update({'Commune': 'first'})
            data = df.groupby('Commune').agg(agg_dict)
        elif scale_name == 'Département':
            agg_dict.update({'Département': 'first'})
            data = df.groupby('Département').agg(agg_dict)
        elif scale_name == 'Région':
            st.warning("Problèmes de données")
            data = None
        elif scale_name == 'Région SNCF':
            agg_dict.update({'Région SNCF': 'first'})
            data = df.groupby('Région SNCF').agg(agg_dict)
        return data

    # Cette fonction met à jour et affiche le graphique en fonction des données sélectionnées.
    def update_graph_names(selected_data: pd.DataFrame, value_name: str, type_text: str = None,
                           title: str = None) -> None:
        # Tri des données sélectionnées
        selected_data.sort_values(by=value_name, ascending=True, inplace=True)
        max_values_names = len(selected_data)

        fig = go.Figure()

        # Ajout des barres au graphique
        fig.add_trace(go.Bar(
            x=selected_data[value_name],
            y=selected_data[selected_scale],
            orientation='h',
            marker=dict(color='skyblue'),
            width=0.3 / (max_values_names - len(selected_names)) if max_values_names != len(selected_names) else 0.3,
        ))

        # Ajout des annotations pour chaque valeur
        for j, v in enumerate(selected_data[value_name]):
            if type_text == 'pourcentage':
                annotation_text = f'{v:.2f}%'
            elif type_text == 'secondes':
                annotation_text = seconds_to_minutes(v * 60)
            elif type_text == 'minutes':
                annotation_text = minutes_to_hours(v * 60)
            elif type_text == 'heures':
                annotation_text = hours_to_days(v)
            else:
                annotation_text = f'{v}'

            fig.add_annotation(x=v + max(selected_data[value_name]) / 15, y=j,
                               text=f'<b>{annotation_text}</b>',
                               showarrow=False,
                               align='left',
                               valign='bottom')

        # Détermination du texte du titre
        if 'Retard' in title:
            text = f'causé par perturbation'
        else:
            text = 'des perturbations'

        # Mise à jour du layout du graphique
        fig.update_layout(
            title=dict(
                text=f'{title} {text} - {selected_scale} <br> du {start} au {end}' if title is not None else None,
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(visible=False),
            height=100 + 70 * len(selected_names),
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family='Arial', size=11, color='black')
        )

        # Affichage du graphique interactif dans Streamlit
        st.plotly_chart(fig)

    # Liste des noms de valeurs à utiliser pour l'agrégation
    list_values_names = ['avg_arrival_delay_per_disruption', 'mean_duration_disruption', 'median_duration_disruption']
    list_type_text = ['secondes', "minutes", "minutes"]

    # Sélection de l'échelle géographique
    selected_scale = st.selectbox('Choisir une échelle géographique',
                                  ['Gare', 'Commune', 'Département', 'Région', 'Région SNCF'])

    # Vérification de la validité des données pour l'échelle sélectionnée
    if update_graph_scale(selected_scale, list_values_names) is None:
        return None

    # Mise à jour des données en fonction de l'échelle sélectionnée
    data = update_graph_scale(selected_scale, list_values_names).iloc[1:]

    # Sélection des noms pour l'échelle sélectionnée
    selected_names = st.multiselect(f'Sélectionnez des {selected_scale.lower()}s', data[selected_scale],
                                    default=data[selected_scale][0:5])

    # Définition des noms des onglets
    list_tab_names = ['Retard moyen', 'Durée moyenne', 'Durée médiane']
    tab1, tab2, tab3 = st.tabs(list_tab_names)
    list_tabs = [tab1, tab2, tab3]

    # Affichage des graphiques dans les onglets
    for tab, value_name, type_text, tab_name in zip(list_tabs, list_values_names, list_type_text, list_tab_names):
        with tab:
            update_graph_names(data[data[selected_scale].isin(selected_names)], value_name, type_text, tab_name)
def graph_bar_chart_perturbation_gare(df_non_detail: pd.DataFrame, df_detail: pd.DataFrame,
                                      start: datetime, end: datetime) -> None:
    # Cette fonction met à jour les données en fonction de l'échelle géographique sélectionnée.
    def update_graph_scale(df: pd.DataFrame, scale_name: str, coll_to_agg: List[str]) -> pd.DataFrame:
        # Création d'un dictionnaire pour l'agrégation des colonnes
        agg_dict = {value_name: 'sum' for value_name in coll_to_agg}

        # Agrégation des données en fonction de l'échelle géographique sélectionnée
        if scale_name == 'Gare':
            agg_dict.update({'Gare': 'first'})
            data = df.groupby('Code UIC').agg(agg_dict)
        elif scale_name == 'Commune':
            agg_dict.update({'Commune': 'first'})
            data = df.groupby('Commune').agg(agg_dict)
        elif scale_name == 'Département':
            agg_dict.update({'Département': 'first'})
            data = df.groupby('Département').agg(agg_dict)
        elif scale_name == 'Région':
            st.warning("Problèmes de données")
            data = None
        elif scale_name == 'Région SNCF':
            agg_dict.update({'Région SNCF': 'first'})
            data = df.groupby('Région SNCF').agg(agg_dict)

        return data

    # Cette fonction met à jour et retourne le graphique en fonction des données sélectionnées.
    def update_graph_names(selected_data: pd.DataFrame, list_names: List[str], detail: bool = True):
        # Transposition des données sélectionnées
        T_data = selected_data[list_names].T
        data = T_data.sort_values(by=T_data.columns[0], ascending=True)

        # Initialisation de la figure Plotly
        fig = go.Figure()

        # Ajout des barres au graphique
        fig.add_trace(go.Bar(
            x=data[data.columns[0]].values,
            y=data[data.columns[0]].index,
            orientation='h',
            marker=dict(color='skyblue'),
            width=0.3,
        ))

        # Ajout des annotations pour chaque valeur
        for j, v in enumerate(data[data.columns[0]]):
            fig.add_annotation(x=v+max(data[data.columns[0]])/15, y=j,
                               text=f'<b>{v}</b>',
                               showarrow=False,
                               align='left',
                               valign='bottom')

        # Détermination du texte du titre
        if detail:
            text = f'Détail {selected_type}'
        else:
            text = 'Nombre de perturbation par type'

        # Mise à jour du layout du graphique
        fig.update_layout(
            title=dict(
                text=f'{text} - {selected_scale} - {selected_name} <br> du {start} au {end}',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(visible=False),
            height=75*len(list_names) if len(list_names) > 5 else 150*len(list_names),
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family='Arial', size=11, color='black')
        )
        return fig

    # Liste des types de perturbations
    list_values = ['Non Connu',
                   'assistance passagers',
                   'conditions externes',
                   'conditions operationnelles',
                   'incivilite',
                   'problemes techniques']

    # Sélection de l'échelle géographique
    selected_scale = st.selectbox('Choisir une échelle géographique',
                                  ['Gare', 'Commune', 'Département', 'Région', 'Région SNCF'],
                                  key='1')

    # Mise à jour des données en fonction de l'échelle sélectionnée
    data = update_graph_scale(df_non_detail, selected_scale, list_values)
    if data is None:
        return None

    # Sélection du nom pour l'échelle sélectionnée
    if selected_scale == 'Gare':
        selected_data = data[selected_scale].dropna().sort_values(ascending=True)
        selected_name = st.selectbox(f'Sélectionnez des {selected_scale.lower()}s', selected_data, key='2')
    else:
        selected_name = st.selectbox(f'Sélectionnez des {selected_scale.lower()}s', data[selected_scale].iloc[1:], key='2')

    # Création des onglets pour les graphiques détaillés et non détaillés
    tab1, tab2 = st.tabs(['non detaillé', 'detaillé'])

    # Affichage du graphique non détaillé dans le premier onglet
    with tab1:
        fig = update_graph_names(data[data[selected_scale] == selected_name], list_values, detail=False)
        st.plotly_chart(fig)

    # Affichage du graphique détaillé dans le deuxième onglet
    with tab2:
        selected_type = st.selectbox('Choisir une échelle', list_values[1:], key='3')
        if selected_type == 'problemes techniques':
            data_detail = update_graph_scale(df_detail, selected_scale, PROBLEMES_TECHNIQUES)
            fig_detail = update_graph_names(data_detail[data_detail[selected_scale] == selected_name],
                                            PROBLEMES_TECHNIQUES)
        elif selected_type == 'conditions operationnelles':
            data_detail = update_graph_scale(df_detail, selected_scale, CONDITIONS_OPERATIONNELLES)
            fig_detail = update_graph_names(data_detail[data_detail[selected_scale] == selected_name], CONDITIONS_OPERATIONNELLES)
        elif selected_type == 'conditions externes':
            data_detail = update_graph_scale(df_detail, selected_scale, CONDITIONS_EXTERNES)
            fig_detail = update_graph_names(data_detail[data_detail[selected_scale] == selected_name], CONDITIONS_EXTERNES)
        elif selected_type == 'incivilite':
            data_detail = update_graph_scale(df_detail, selected_scale, INCIVILITE)
            fig_detail = update_graph_names(data_detail[data_detail[selected_scale] == selected_name], INCIVILITE)
        elif selected_type == 'assistance passagers':
            data_detail = update_graph_scale(df_detail, selected_scale, ASSISTANCE_PASSAGERS)
            fig_detail = update_graph_names(data_detail[data_detail[selected_scale] == selected_name], ASSISTANCE_PASSAGERS)

        st.plotly_chart(fig_detail)
def graph_disruption_across_day(disruption_by_ten: List[int], hours_intervals_name: List[str],
                                start: datetime = None, end: datetime = None, new_graph: bool = True,
                                fig = None, color: str = 'royalblue', fill: bool = True, name: str = None) -> None:
    # Créer une nouvelle figure si new_graph est True
    if new_graph:
        fig = go.Figure(layout=dict(
            title=dict(
                text = f'Nombre de perturbation au fil de la journée <br> du {start} au {end}',
                x=0.5,
                xanchor='center'
            )))

    # Ajouter une trace de ligne au graphique
    fig.add_trace(go.Line(
        x=list(range(len(disruption_by_ten))),  # Position x des points
        y=disruption_by_ten,  # Valeurs y des points
        fill='tozeroy' if fill else None,  # Remplir la zone sous la ligne si fill est True
        mode='lines+markers',  # Afficher des lignes et des marqueurs
        line=dict(color=color, width=2),  # Style de la ligne
        marker=dict(size=3, color=color),  # Style des marqueurs
        name=name  # Nom de la trace
    ))

    # Si un nouveau graphique a été créé, ajouter les titres et labels des axes
    if new_graph:
        fig.update_layout(
            xaxis_title='Heures',  # Titre de l'axe x
            yaxis_title='Nombre de perturbations en cours',  # Titre de l'axe y
            xaxis=dict(
                tickvals=list(range(len(hours_intervals_name))),  # Valeurs des ticks sur l'axe x
                ticktext=hours_intervals_name,  # Texte des ticks sur l'axe x
                range=[24, len(hours_intervals_name) - 1]  # Plage de l'axe x
            ),
            template='plotly_white',  # Template de style de graphique
            height=400,  # Hauteur du graphique
            width=2000,  # Largeur du graphique
        )

        # Afficher le graphique avec Streamlit
        st.plotly_chart(fig)
def graph_disruption_across_day_type(data: pd.DataFrame, start: datetime, end: datetime, colors: List[str] = None) -> None:
    # Définition des couleurs par défaut si aucune n'est spécifiée
    if colors is None:
        colors = ['royalblue', 'pink', 'gold', 'skyblue', 'purple', 'red']

    # Obtenir les intervalles de dix minutes et les noms des intervalles horaires
    ten_intervals, hours_intervals_name = interval_hour()

    def get_data(list_perturbation, list_choice) -> dict:
        # Initialisation de la liste des perturbations par tranche de dix minutes pour chaque choix
        list_disruption_by_ten = []
        for choice in list_choice:
            # Filtrage des données selon le choix de cause de perturbation
            df = data[data['cause_delay'].isin(choice)]
            # Calcul du nombre de perturbations par tranche de dix minutes
            disruption_by_ten = disruption_across_day(ten_intervals, df)
            list_disruption_by_ten.append(disruption_by_ten)

        # Création d'un dictionnaire associant chaque type de perturbation à ses données de perturbation par tranche de dix minutes
        disruption_by_ten_dict = {type_perturbation: list_disruption_by_ten[i] for i, type_perturbation in enumerate(list_perturbation)}

        return disruption_by_ten_dict

    def update_graph(disruption_by_ten_dict, types_names, detail: bool = True) -> None:
        # Initialisation de la figure Plotly
        fig = go.Figure()

        # Boucle sur les types de perturbation et les couleurs correspondantes
        for type, color in zip(types_names, colors):
            # Appel de la fonction pour générer le graphique de perturbation par tranche de dix minutes
            graph_disruption_across_day(disruption_by_ten_dict[type], hours_intervals_name,
                                        new_graph=False, fig=fig, fill=False, color=color, name=type)

            # Définition du texte de titre en fonction du mode détaillé ou non
            if detail:
                text = f'Détail {selected_type}'
            else:
                text = 'Nombre de perturbation au fil de la journée par type'

            # Mise à jour de la mise en page du graphique
            fig.update_layout(
                title=dict(
                    text=f'{text} <br> du {start} au {end}',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Heures',
                yaxis_title='Nombre de perturbations en cours',
                xaxis=dict(
                    tickvals=list(range(len(hours_intervals_name))),
                    ticktext=hours_intervals_name,
                    range=[24, len(hours_intervals_name) - 1]
                ),
                template='plotly_white',
                height=400,
                width=2000,
                legend=dict(
                    title='Types de perturbations',
                    title_font=dict(size=11, color='black'),
                    font=dict(size=12, color='black'),
                    orientation='h',
                    x=0.5,
                    xanchor='center',
                    y=0.8,
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.5)',
                    bordercolor='black',
                    borderwidth=1
                )
            )

        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)

    # Liste des types de perturbation
    types_de_perturbation = [
        'Non Connu',
        'assistance passagers',
        'conditions externes',
        'conditions operationnelles',
        'incivilite',
        'problemes techniques'
    ]

    # Listes des causes de perturbation détaillées
    list_no_detail = [
        INCIVILITE,
        PROBLEMES_TECHNIQUES,
        CONDITIONS_OPERATIONNELLES,
        CONDITIONS_EXTERNES,
        ASSISTANCE_PASSAGERS,
        ['Non connu']
    ]

    # Liste des valeurs de types de perturbation
    list_values = [
        'incivilite',
        'problemes techniques',
        'conditions operationnelles',
        'conditions externes',
        'assistance passagers',
        'Non Connu'
    ]

    # Récupération des données de perturbation par tranche de dix minutes
    disruption_by_ten_dict = get_data(types_de_perturbation, list_no_detail)

    # Création de deux onglets dans l'interface Streamlit
    tab1, tab2 = st.tabs(['Non detaillé', 'détaillé'])

    # Onglet pour les données non détaillées
    with tab1:
        selected_types = st.multiselect("Selectionnez des types de perturbation", disruption_by_ten_dict.keys(), default='incivilite')
        update_graph(disruption_by_ten_dict, selected_types, detail=False)

    # Onglet pour les données détaillées
    with tab2:
        selected_type = st.selectbox("Selectionnez un type de perturbation", list(disruption_by_ten_dict.keys())[1:])
        for v, l in zip(list_values[:-1], list_no_detail[:-1]):
            if selected_type == v:
                try:
                    disruption_by_ten_dict_detail = get_data(data[data.isin(l)]['cause_delay'].dropna().unique(), [[x] for x in l])
                    selected_types_detail = st.multiselect("Selectionnez des causes de perturbation", l, l[0])
                    update_graph(disruption_by_ten_dict_detail, selected_types_detail)
                except KeyError as e:
                    problematic_keys = e.args
                    for problematic_key in problematic_keys:
                        if problematic_key in selected_types_detail:
                            selected_types_detail.remove(problematic_key)
                    update_graph(disruption_by_ten_dict_detail, selected_types_detail)
                    for e in e.args:
                        st.info(f'La cause : {e}  n\'a pas eu lieu sur la période séléctioné')
def graph_cause_delay_info(cause_delay_detail: pd.DataFrame, start: datetime, end: datetime) -> None:
    # Fonction pour obtenir les données agrégées en fonction des colonnes spécifiées et du regroupement
    def get_data(list_col_name: List[str], group_by: str, list_causes: List[str] = None) -> pd.DataFrame:
        # Dictionnaire pour agréger les colonnes spécifiées
        agg_dict = {col_name: 'sum' for col_name in list_col_name if 'mean' not in col_name}  # Somme des colonnes non moyennes
        agg_dict.update({col_name: 'mean' for col_name in list_col_name if 'mean' in col_name})  # Moyenne des colonnes moyennes
        agg_dict.update({group_by: 'first'})  # Premier élément du groupe par

        # Agrégation des données selon le group_by spécifié
        data = cause_delay_detail.groupby(group_by).agg(agg_dict)

        # Filtrage des données si une liste de causes spécifiques est fournie
        if list_causes is not None:
            data = data[data[group_by].isin(list_causes)]
        return data

    # Fonction pour dessiner le graphique
    def draw_graph(data: pd.DataFrame, col_name: str, group_by: str, title: str, type_text: str = None, selected_type: str = None) -> None:
        # Conversion des valeurs timedelta en secondes, minutes ou heures si nécessaire
        if type(data[col_name][0]) == pd.Timedelta:
            if type_text == 'secondes':
                data[col_name] = data[col_name].dt.total_seconds()
            elif type_text == 'minutes':
                data[col_name] = data[col_name].dt.total_seconds() / 60
            elif type_text == 'heures':
                data[col_name] = data[col_name].dt.total_seconds() / 3600

        # Tri des données par ordre croissant selon la colonne spécifiée
        data.sort_values(by=col_name, ascending=True, inplace=True)
        fig = go.Figure()

        # Ajout de la trace de type Bar dans la figure
        fig.add_trace(go.Bar(
            x=data[col_name],
            y=data[group_by],
            orientation='h',
            marker=dict(color='skyblue'),
            width=0.3,
            opacity=0.9
        ))

        # Ajout des annotations sur les barres du graphique
        for j, v in enumerate(data[col_name].sort_values(ascending=True)):
            if type_text == 'pourcentage':
                annotation_text = f'{v:.2f}%'
            elif type_text == 'secondes':
                annotation_text = seconds_to_minutes(v * 60)  # Conversion secondes en minutes
            elif type_text == 'minutes':
                annotation_text = minutes_to_hours(v)  # Conversion minutes en heures
            elif type_text == 'heures':
                annotation_text = hours_to_days(v)  # Conversion heures en jours
            else:
                annotation_text = f'{v}'

            # Ajout des annotations sur les barres
            fig.add_annotation(x=v + max(data[col_name]) / 15, y=j,
                               text=f'<b>{annotation_text}</b>',
                               showarrow=False,
                               align='left',
                               valign='bottom')

        # Construction du titre du graphique en fonction du type sélectionné ou du type global
        if selected_type is not None:
            text = f' - Détail {selected_type}'
        else:
            if 'Retard' in title:
                text = ' causé par chaque type de perturbation'
            elif 'Nombre' in title:
                text = ' de perturbation par type de perturbation'
            else:
                text = ' des perturbations par type de perturbation'

        # Mise à jour de la mise en page du graphique
        fig.update_layout(
            title=dict(
                text=f'{title}{text} <br> du {start} au {end}',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(visible=False),
            height=100 + 75 * len(data[col_name]),  # Calcul de la hauteur du graphique en fonction du nombre de données
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(family='Arial', size=11, color='black')
        )

        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)

    # Fonction pour mettre à jour le graphique en fonction du type sélectionné
    def update_graph(df: pd.DataFrame, list_col_name: List[str], list_type_text: List[None | str] = None) -> None:
        list_tab_names = ['Nombre', 'Retard moyen', 'Durée moyenne']
        tab1, tab2, tab3 = st.tabs(list_tab_names)

        list_tabs = [tab1, tab2, tab3]
        for tab, col_name, type_text, tab_name in zip(list_tabs, list_col_name, list_type_text, list_tab_names):
            with tab:
                col1, col2 = st.columns(2)
                with col1:
                    # Appel de la fonction draw_graph pour dessiner le graphique pour chaque type de données
                    draw_graph(df, col_name, group_by='group_delay', title=tab_name, type_text=type_text)

                with col2:
                    # Sélection du type de perturbation spécifique
                    selected_type = st.selectbox('Selectionnez un type de perturbation',
                                                 cause_delay_detail['group_delay'].unique()[1:],
                                                 index=0,
                                                 key=f'tab {col_name}')

                    # Filtrage des données détaillées en fonction du type de perturbation sélectionné
                    if selected_type == 'problemes techniques':
                        data_detail = get_data(list_col_name, 'cause_delay', PROBLEMES_TECHNIQUES)
                        draw_graph(data_detail, col_name, group_by='cause_delay', title=tab_name, type_text=type_text,
                                   selected_type=selected_type)
                    elif selected_type == 'conditions operationnelles':
                        data_detail = get_data(list_col_name, 'cause_delay', CONDITIONS_OPERATIONNELLES)
                        draw_graph(data_detail, col_name, group_by='cause_delay', title=tab_name, type_text=type_text,
                                   selected_type=selected_type)
                    elif selected_type == 'conditions externes':
                        data_detail = get_data(list_col_name, 'cause_delay', CONDITIONS_EXTERNES)
                        draw_graph(data_detail, col_name, group_by='cause_delay', title=tab_name, type_text=type_text,
                                   selected_type=selected_type)
                    elif selected_type == 'incivilite':
                        data_detail = get_data(list_col_name, 'cause_delay', INCIVILITE)
                        draw_graph(data_detail, col_name, group_by='cause_delay', title=tab_name, type_text=type_text,
                                   selected_type=selected_type)
                    elif selected_type == 'assistance passagers':
                        data_detail = get_data(list_col_name, 'cause_delay', ASSISTANCE_PASSAGERS)
                        draw_graph(data_detail, col_name, group_by='cause_delay', title=tab_name, type_text=type_text,
                                   selected_type=selected_type)

    # Liste des noms de colonnes pour les données à agréger et à afficher
    list_col_name = [
        'count',  # Nombre de perturbations
        'mean_arrival_delay',  # Retard moyen à l'arrivée
        'mean_duration_disruption'  # Durée moyenne de la perturbation
    ]

    # Liste des types de texte correspondant aux différentes métriques (secondes, minutes, aucune pour count)
    list_type_text = [None, "secondes", "minutes"]

    # Obtention des données agrégées initiales pour affichage
    data = get_data(list_col_name, group_by='group_delay')

    # Mise à jour du graphique en fonction des données agrégées et des paramètres spécifiés
    update_graph(data, list_col_name, list_type_text)
#############################################################################################



########################## CREATIONS DES PAGES ###########################################
def get_data_page1_1(df_past: pd.DataFrame, df_disruption: pd.DataFrame, week_start: datetime) -> tuple:
    """
    Génère des données statistiques sur les perturbations pour la page 1.

    Args:
    - df_past (pd.DataFrame): DataFrame contenant les données passées.
    - df_disruption (pd.DataFrame): DataFrame contenant les détails des perturbations.
    - week_start (datetime): Date de début de la semaine.

    Returns:
    - Tuple: Dates, séries de données statistiques quotidiennes, DataFrames passés, valeurs statistiques globales.
    """

    dates = pd.concat([df_past['data_date'], pd.Series(week_start)], ignore_index=True)

    mean_time_disruption_minutes = int((df_disruption['duration_disruption'].dt.total_seconds() / 60).mean())
    median_time_disruption_minutes = int((df_disruption['duration_disruption'].dt.total_seconds() / 60).median())
    quantile_25_time_disruption_minutes = int((df_disruption['duration_disruption'].dt.total_seconds() / 60).quantile(0.25))
    quantile_75_time_disruption_minutes = int((df_disruption['duration_disruption'].dt.total_seconds() / 60).quantile(0.75))

    mean_time_disruption_minutes_daily = round(df_disruption.groupby('data_date')['duration_disruption'].mean().dt.total_seconds() / 60)
    median_time_disruption_minutes_daily = round(df_disruption.groupby('data_date')['duration_disruption'].median().dt.total_seconds() / 60)
    quantile_25_time_disruption_minutes_daily = df_disruption.groupby('data_date')['duration_disruption'].quantile(0.25).dt.total_seconds() / 60
    quantile_75_time_disruption_minutes_daily = df_disruption.groupby('data_date')['duration_disruption'].quantile(0.75).dt.total_seconds() / 60

    df_mean_time_disruption_minutes_one_day = df_past['mean_time_disruption_minutes_daily']
    df_median_time_disruption_minutes_one_day = df_past['median_time_disruption_minutes_daily']
    df_quantile_25_time_disruption_minutes_one_day = df_past['quantile_25_time_disruption_minutes_daily']
    df_quantile_75_time_disruption_minutes_one_day = df_past['quantile_75_time_disruption_minutes_daily']

    valeurs_mean = df_past['mean_time_disruption_minutes']
    valeurs_median = df_past['median_time_disruption_minutes']
    valeurs_quantile_25 = df_past['quantile_25_time_disruption_minutes']
    valeurs_quantile_75 = df_past['quantile_75_time_disruption_minutes']

    return (dates,
            [mean_time_disruption_minutes_daily, median_time_disruption_minutes_daily,
             quantile_25_time_disruption_minutes_daily, quantile_75_time_disruption_minutes_daily],
            [df_mean_time_disruption_minutes_one_day, df_median_time_disruption_minutes_one_day,
             df_quantile_25_time_disruption_minutes_one_day, df_quantile_75_time_disruption_minutes_one_day],
            [valeurs_mean, valeurs_median, valeurs_quantile_25, valeurs_quantile_75],
            mean_time_disruption_minutes,
            median_time_disruption_minutes,
            quantile_25_time_disruption_minutes,
            quantile_75_time_disruption_minutes)
def get_data_page1_2(df_vehicle: pd.DataFrame, df_disruption: pd.DataFrame, df_past: pd.DataFrame) -> tuple:
    """
    Génère des données statistiques sur les retards pour la page 1.

    Args:
    - df_vehicle (pd.DataFrame): DataFrame contenant les données des véhicules.
    - df_disruption (pd.DataFrame): DataFrame contenant les détails des perturbations.
    - df_past (pd.DataFrame): DataFrame contenant les données passées.

    Returns:
    - Tuple: Statistiques quotidiennes sur les retards, DataFrames passés sur les retards, valeurs statistiques globales sur les retards.
    """

    nb_disruption = df_vehicle['id_disruption'].nunique()
    total_delay = int(df_disruption['arrival_delay'].sum() / 60)
    mean_delay_by_disruption = round(total_delay / nb_disruption * 60)

    total_delay_daily = round(df_disruption.groupby('data_date')['arrival_delay'].sum() / 60)
    nb_disruption_daily = df_vehicle.groupby('data_date')['id_disruption'].nunique()
    mean_delay_by_disruption_daily = round(total_delay_daily / nb_disruption_daily * 60)

    df_total_delay_day = df_past['total_delay_daily']
    df_mean_delay_day = df_past['mean_delay_by_disruption_daily']

    valeurs_total = df_past['total_delay']
    valeurs_moyen = round(df_past['total_delay'] / df_past['nb_disruption'] * 60)

    return (total_delay_daily,
            nb_disruption_daily,
            mean_delay_by_disruption_daily,
            df_total_delay_day,
            df_mean_delay_day,
            valeurs_total,
            valeurs_moyen,
            total_delay,
            mean_delay_by_disruption)
def get_data_page1_3(df_vehicle: pd.DataFrame, df_past: pd.DataFrame) -> tuple:
    """
    Génère des données statistiques sur les véhicules pour la page 1.

    Args:
    - df_vehicle (pd.DataFrame): DataFrame contenant les données des véhicules.
    - df_past (pd.DataFrame): DataFrame contenant les données passées.

    Returns:
    - Tuple: Statistiques globales et quotidiennes sur les véhicules.
    """

    nb_disruption = df_vehicle['id_disruption'].nunique()
    nb_vehicle_journeys = df_vehicle['vehicle_id'].nunique()
    per_disrupted = nb_disruption / nb_vehicle_journeys * 100

    nb_vehicle_journeys_daily = df_vehicle.groupby('data_date')['vehicle_id'].nunique()
    nb_disruption_daily = df_vehicle.groupby('data_date')['id_disruption'].nunique()
    per_disrupted_daily = nb_disruption_daily / nb_vehicle_journeys_daily * 100

    df_journeys_one_day = df_past['nb_vehicle_journeys_daily']
    df_disruption_one_day = df_past['nb_disruption_daily']
    df_per_disruption_one_day = df_past['per_disrupted_daily']

    value_nb_journeys = df_past['nb_vehicle_journeys']
    value_nb_disruption = df_past['nb_disruption']
    value_per_disruption = df_past['per_disrupted']

    return (nb_disruption,
            nb_vehicle_journeys,
            per_disrupted,
            nb_vehicle_journeys_daily,
            nb_disruption_daily,
            per_disrupted_daily,
            df_journeys_one_day,
            df_disruption_one_day,
            df_per_disruption_one_day,
            value_nb_journeys,
            value_nb_disruption,
            value_per_disruption)
def page1(df_vehicle:pd.DataFrame, df_disruption: pd.DataFrame,df_past:pd.DataFrame, week_start: datetime,
          week_end: datetime, days:List[str], is_default:bool, start_time:time, end_time: time) -> None:

    (dates,
     Series_list,
     dfs_list,
     values_list,
     mean_time_disruption_minutes,
     median_time_disruption_minutes,
     quantile_25_time_disruption_minutes,
     quantile_75_time_disruption_minutes) = get_data_page1_1(df_past, df_disruption, week_start)
    (total_delay_daily,
     nb_disruption_daily,
     mean_delay_by_disruption_daily,
     df_total_delay_day,
     df_mean_delay_day,
     valeurs_total,
     valeurs_moyen,
     total_delay,
     mean_delay_by_disruption) = get_data_page1_2(df_vehicle, df_disruption, df_past)
    (nb_disruption,
     nb_vehicle_journeys,
     per_disrupted,
     nb_vehicle_journeys_daily,
     nb_disruption_daily,
     per_disrupted_daily,
     df_journeys_one_day,
     df_disruption_one_day,
     df_per_disruption_one_day,
     value_nb_journeys,
     value_nb_disruption,
     value_per_disruption) = get_data_page1_3(df_vehicle, df_past)

    first_tab_name = 'Analyse de la semaine' if is_default else f'Analyse du {week_start} au {week_end}'
    list_tab_names = [first_tab_name, 'Comparaison des jours', 'Comparaison des semaines', 'A venir']
    tab1, tab2, tab3, tab4 = st.tabs(list_tab_names)

    with tab4:
        st.info('Possibilité de sauvgarder différentes période dans une table pour les comparer à venir')

    with tab1:
        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

        with col1:
            metrics_disruprion(df_past, nb_vehicle_journeys, 'nb_vehicle_journeys', week_start, week_end, is_default,
                               label_name='Nombre de trajet', delta_color='off')
        with col2:
            metrics_disruprion(df_past, nb_disruption, 'nb_disruption', week_start, week_end, is_default, label_name='Nombre de perturbation')
        with col3:
            metrics_disruprion(df_past, per_disrupted, 'per_disrupted', week_start, week_end, is_default,
                               label_name='Pourcentage de perturbation', type_text='pourcentage')

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            grah_line_chart([nb_vehicle_journeys_daily], week_start, week_end, days, colors=['skyblue'], title='Nombre de trajet', ytitle='Nombre de trajet', is_default=is_default)
        with col2:
            grah_line_chart([nb_disruption_daily], week_start, week_end, days, colors=['pink'], title='Nombre de perturbation', ytitle='Nombre de perturbation', is_default=is_default)
        with col3:
            grah_line_chart([per_disrupted_daily], week_start, week_end, days, colors=['gold'], title='Pourcentage de perturbation',
                      ytitle='Pourcentage de perturbation', type_text='pourcentage', is_default=is_default)

        col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")

        with col1:
            grah_line_chart(Series_list, week_start, week_end, days=days, colors=['skyblue', 'pink', 'gray', 'gray'],
                            title='Temps moyen et median des perturbations', ytitle='Temps (minutes)',
                            label_names=['Moyenne', 'Median', 'Quantile 25% ', 'Quantile 75%'], show_legend=True,
                            alpha_list=[0.7, 0.7, 0.2, 0.2], show_values=[True, True, False, False], width=1800, height=600, type_text='minutes', is_default=is_default)
        with col2:
            metrics_disruprion(df_past, mean_time_disruption_minutes, 'mean_time_disruption_minutes', week_start, week_end, is_default,
                               label_name='Temps moyen des perturbations', type_text='minutes')

            metrics_disruprion(df_past, median_time_disruption_minutes, 'median_time_disruption_minutes', week_start, week_end, is_default,
                               label_name='Temps median des perturbations',type_text='minutes')

            metrics_disruprion(df_past, quantile_25_time_disruption_minutes, 'quantile_25_time_disruption_minutes', week_start, week_end, is_default,
                               label_name='Valeur du Quantile 25%', type_text='minutes')

            metrics_disruprion(df_past, quantile_75_time_disruption_minutes, 'quantile_75_time_disruption_minutes', week_start, week_end, is_default,
                               label_name='Valeur du Quantile 75%', type_text='minutes')


        col1, col2, col3 = st.columns([0.3, 0.35, 0.35])

        with col1:
            metrics_disruprion(df_past, total_delay, 'total_delay', week_start, week_end, is_default,
                               label_name='Retard total de retard causé par les perturbations', type_text='heures')
            metrics_disruprion(df_past, mean_delay_by_disruption, 'mean_delay_by_disruption', week_start, week_end, is_default,
                               label_name='Retard moyen causé par perturbation enregistré', type_text='minutes')

        with col2:
            grah_line_chart([total_delay_daily], week_start, week_end, days, colors=['skyblue'],
                            title='Retard total causé par les perturbations sur l\'ensemble du réseau \n',
                            ytitle='Temps total de retard (heures)', width=1000, height=350, type_text='heures', is_default=is_default)
        with col3:
            grah_line_chart([mean_delay_by_disruption_daily], week_start, week_end, days, colors=['pink'],
                            title='Retard moyen causé par perturbation enregistré ',
                            ytitle='Temps moyen de retard (minutes)', width=1000, height=350, type_text='minutes', is_default=is_default)

    with tab2:
        if not is_default:
            st.warning('Ces graphiques ne dépendent pas de la date choisi')

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

        with col1:
            metrics_disruprion(df_past, nb_vehicle_journeys, 'nb_vehicle_journeys', week_start, week_end, is_default=True,
                               label_name='Nombre de trajet', delta_color='off')
        with col2:
            metrics_disruprion(df_past, nb_disruption, 'nb_disruption', week_start, week_end, is_default=True,
                               label_name='Nombre de perturbation')
        with col3:
            metrics_disruprion(df_past, per_disrupted, 'per_disrupted', week_start, week_end, is_default=True,
                               label_name='Pourcentage de perturbation', type_text='pourcentage')

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            graph_line_chart_comparaison_day([df_journeys_one_day], days=days, key_prefix='trajets', colors=['skyblue'], title='Nombre de trajet', ytitle='Nombre de trajet')
        with col2:
            graph_line_chart_comparaison_day([df_disruption_one_day], days=days, key_prefix='perturbations', colors=['pink'], title='Nombre de perturbation', ytitle='Nombre de perturbation')
        with col3:
            graph_line_chart_comparaison_day([df_per_disruption_one_day], days=days, key_prefix='per_perturbations', colors=['gold'], title='Pourcentage de perturbation', ytitle='Pourcentage de perturbation', type_text='pourcentage')

        col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")

        with col1:
            graph_line_chart_comparaison_day(dfs_list, days=days,
                                             colors=['skyblue', 'pink', 'gray', 'gray'],
                                             title='Temps moyen et median des perturbations', ytitle='Temps (minutes)',
                                             label_names=['Moyenne', 'Median', 'Quantile 25% ', 'Quantile 75%'],
                                             show_legend=True, alpha_list=[0.7, 0.7, 0.2, 0.2],
                                             show_values=[True, True, False, False], width=1800, height=700, type_text='minutes')

        with col2:
            metrics_disruprion(df_past, mean_time_disruption_minutes, 'mean_time_disruption_minutes', week_start,
                               week_end, is_default=True,
                               label_name='Temps moyen des perturbations', type_text='minutes')

            metrics_disruprion(df_past, median_time_disruption_minutes, 'median_time_disruption_minutes', week_start,
                               week_end, is_default=True,
                               label_name='Temps median des perturbations', type_text='minutes')

            metrics_disruprion(df_past, quantile_25_time_disruption_minutes, 'quantile_25_time_disruption_minutes',
                               week_start, week_end, is_default=True,
                               label_name='Valeur du Quantile 25%', type_text='minutes')

            metrics_disruprion(df_past, quantile_75_time_disruption_minutes, 'quantile_75_time_disruption_minutes',
                               week_start, week_end, is_default=True,
                               label_name='Valeur du Quantile 75%', type_text='minutes')

        col1, col2, col3 = st.columns([0.3, 0.35, 0.35])

        with col1:
            metrics_disruprion(df_past, total_delay, 'total_delay', week_start, week_end, is_default= True,
                               label_name='Retard total de retard causé par les perturbations', type_text='heures')
            metrics_disruprion(df_past, mean_delay_by_disruption, 'mean_delay_by_disruption', week_start, week_end,
                               is_default=True,
                               label_name='Retard moyen causé par perturbation enregistré',
                               type_text='minutes')

        with col2:
            graph_line_chart_comparaison_day([df_total_delay_day], days=days,
                                                 key_prefix='Temps total', colors=['skyblue'],
                                                 title='Retard total causé par les perturbations sur l\'ensemble du réseau',
                                                 ytitle='Temps total de retard (heures)', width=1000, height=350, type_text='heures')
        with col3:
            graph_line_chart_comparaison_day([df_mean_delay_day], days=days,
                                                 key_prefix='Temps moyen', colors=['pink'],
                                                 title='Retard moyen causé par perturbation enregistré ',
                                                 ytitle='Temps moyen de retard (minutes)', width=1000, height=350, type_text='minutes')

    with tab3:
        if not is_default:
            st.warning('Ces graphiques ne dépendent pas de la date choisi')

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

        with col1:
            metrics_disruprion(df_past, nb_vehicle_journeys, 'nb_vehicle_journeys', week_start, week_end,
                               is_default=True,
                               label_name='Nombre de trajet', delta_color='off')
        with col2:
            metrics_disruprion(df_past, nb_disruption, 'nb_disruption', week_start, week_end, is_default=True,
                               label_name='Nombre de perturbation')
        with col3:
            metrics_disruprion(df_past, per_disrupted, 'per_disrupted', week_start, week_end, is_default=True,
                               label_name='Pourcentage de perturbation', type_text='pourcentage')

        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            graph_line_chart_comparaison_week(dates, [value_nb_journeys], dates[0], colors=['skyblue'], title='Nombre de trajet', ytitle='Nombre de trajet')
        with col2:
            graph_line_chart_comparaison_week(dates, [value_nb_disruption], dates[0], colors=['pink'], title='Nombre de perturbation', ytitle='Nombre de perturbation')
        with col3:
            graph_line_chart_comparaison_week(dates, [value_per_disruption], dates[0], colors=['gold'], title='Pourcentage de perturbation', ytitle='Pourcentage de perturbation', type_text='pourcentage')

        col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")

        with col1:
            graph_line_chart_comparaison_week(dates, values_list, dates[0], colors=['skyblue', 'pink', 'gray', 'gray'],
                                              title='Temps moyen et median des perturbations', ytitle='Temps (minutes)',
                                              label_names=['Moyenne', 'Median', 'Quantile 25% ', 'Quantile 75%'],
                                              show_legend=True, alpha_list=[0.7, 0.7, 0.2, 0.2],
                                              show_values=[True, True, False, False], width=1800, height=700, type_text='minutes')

        with col2:
            metrics_disruprion(df_past, mean_time_disruption_minutes, 'mean_time_disruption_minutes', week_start,
                               week_end, is_default=True,
                               label_name='Temps moyen des perturbations', type_text='minutes')

            metrics_disruprion(df_past, median_time_disruption_minutes, 'median_time_disruption_minutes', week_start,
                               week_end, is_default=True,
                               label_name='Temps median des perturbations', type_text='minutes')

            metrics_disruprion(df_past, quantile_25_time_disruption_minutes, 'quantile_25_time_disruption_minutes',
                               week_start, week_end, is_default=True,
                               label_name='Valeur du Quantile 25%', type_text='minutes')

            metrics_disruprion(df_past, quantile_75_time_disruption_minutes, 'quantile_75_time_disruption_minutes',
                               week_start, week_end, is_default=True,
                               label_name='Valeur du Quantile 75%', type_text='minutes')

        col1, col2, col3 = st.columns([0.3, 0.35, 0.35])

        with col1:
            metrics_disruprion(df_past, total_delay, 'total_delay', week_start, week_end, is_default=True,
                               label_name='Retard total de retard causé par les perturbations', type_text='heures')
            metrics_disruprion(df_past, mean_delay_by_disruption, 'mean_delay_by_disruption', week_start, week_end,
                               is_default=True,
                               label_name='Retard moyen causé par perturbation enregistré',
                               type_text='minutes')

        with col2:
            graph_line_chart_comparaison_week(dates, [valeurs_total], dates[0], colors=['skyblue'],
                                                  title='Retard total causé par les perturbations sur l\'ensemble du réseau',
                                                  ytitle='Temps total de retard (heures)', width=1000, height=350, type_text='heures')
        with col3:
            graph_line_chart_comparaison_week(dates, [valeurs_moyen], dates[0], colors=['pink'],
                                                  title='Retard moyen causé par perturbation enregistré ',
                                                  ytitle='Temps moyen de retard (minutes)', width=1000, height=350, type_text='minutes')
def get_data_page2(df_merge: pd.DataFrame) -> Tuple[pd.Series]:
    """
    Récupère et traite les données sur les perturbations pour la page 2.

    Args:
    - df_merge (pd.DataFrame): DataFrame contenant les données fusionnées.

    Returns:
    - Tuple: Séries de données sur les trajets, les trajets perturbés, le retard moyen par train, le pourcentage de perturbation par réseau.
    """

    # Nombre de véhicules par réseau
    vehicle_by_network = df_merge.groupby('network_name')['vehicle_id'].count()

    # Nombre de véhicules perturbés par réseau
    disrupted_vehicle_by_network = df_merge[df_merge['id_disruption'].notna()].groupby('network_name')['vehicle_id'].count()

    # Gestion du cas où aucun trajet du réseau n'a été perturbé
    for name in vehicle_by_network.index:
        if name not in disrupted_vehicle_by_network.index:
            disrupted_vehicle_by_network[name] = 0

    # Calcul du pourcentage de véhicules perturbés par réseau
    per_disrupted_vehicle_by_network = (disrupted_vehicle_by_network / vehicle_by_network * 100).where(disrupted_vehicle_by_network >= 1, 0).sort_values(ascending=True)

    # Calcul du retard total à l'arrivée des gares pour chaque réseau
    total_delay_by_network = df_merge.groupby('network_name')['arrival_delay'].sum()

    # Calcul du retard moyen par train pour chaque réseau
    avg_delay_by_network = total_delay_by_network / vehicle_by_network

    # Gestion du cas où il n'y a pas de retard pour certains réseaux
    for i in range(len(disrupted_vehicle_by_network)):
        if disrupted_vehicle_by_network.index[i] not in total_delay_by_network.index:
            avg_delay_by_network[i] = 0

    # Conversion en chaîne de caractères indiquant les retards moyens en secondes
    for i in range(len(avg_delay_by_network)):
        if avg_delay_by_network[i] != 0:
            avg_delay_by_network[i] = str(int(avg_delay_by_network[i] * 60)) + " secondes"
        else:
            avg_delay_by_network[i] = "0 seconde"

    return (vehicle_by_network,
            disrupted_vehicle_by_network,
            avg_delay_by_network,
            per_disrupted_vehicle_by_network)
def page2(df_merge: pd.DataFrame, start: datetime, end: datetime, start_time:time, end_time: time) -> None:

    (vehicle_by_network,
    disrupted_vehicle_by_network,
    avg_delay_by_network,
    per_disrupted_vehicle_by_network) = get_data_page2(df_merge)

    info_current = [vehicle_by_network, disrupted_vehicle_by_network, avg_delay_by_network]
    info_name = ["Nombre de trajet<br>", "Nombre de trajet<br>perturbé", "Retard moyen<br>par train"]

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        try:
            graph_network(info_current, info_name, per_disrupted_vehicle_by_network, start, end, start_time, end_time, title='Information sur les perturbations par réseau')
        except IndexError:
            st.warning('Sélectionnez des réseaux')

    with col2:
        list_tab_names = ['Trajet', 'Perturbation', 'Retard', 'Pourcentage perturbation']
        tab1, tab2, tab3, tab4 = st.tabs(list_tab_names)
        with tab1:
            st.dataframe(vehicle_by_network.sort_values(ascending=False),
                             column_config={
                                 "network_name": "Noms",
                                 "vehicle_id": f'Nombre de trajet entre {str(start_time)[:5]} et {str(end_time)[:5]}'
                             },
                             use_container_width=True,
                             height=400)
        with tab2:
            st.dataframe(disrupted_vehicle_by_network.sort_values(ascending=False),
                             column_config={
                                 "network_name": "Noms",
                                 "vehicle_id": f'Nombre de perturbation entre {str(start_time)[:5]} et {str(end_time)[:5]}'
                             },
                             use_container_width=True,
                             height=400)
        with tab3:
            avg_delay_by_network_ws = avg_delay_by_network.apply(lambda x: int(x.split(' ')[0])).sort_values(ascending=False)
            st.dataframe(avg_delay_by_network_ws.sort_values(ascending=False),
                             column_config={
                                 "network_name": "Noms",
                                 "0": f'Retard moyen par train entre {str(start_time)[:5]} et {str(end_time)[:5]} en secondes'
                             },
                             use_container_width=True,
                             height=400)
        with tab4:
            st.dataframe(per_disrupted_vehicle_by_network.sort_values(ascending=False),
                             column_config={
                                 "network_name": "Noms",
                                 "vehicle_id": st.column_config.NumberColumn(
                                     f'Pourcentage de train perturbé entre {str(start_time)[:5]} et {str(end_time)[:5]}',
                                     format="%.2f%%"
                                 )
                             },
                             use_container_width=True,
                             height=400)
        st.info('Détail des causes de perturbation par réseau à venir')
def get_data_page3_col1(df_merge: pd.DataFrame, df_sp: pd.DataFrame) -> pd.DataFrame:
    """
    Récupère les informations de la colonne 1 pour la page 3.

    Args:
    - df_merge (pd.DataFrame): DataFrame contenant les données fusionnées.
    - df_sp (pd.DataFrame): DataFrame contenant les informations supplémentaires sur les arrêts.

    Returns:
    - pd.DataFrame: DataFrame contenant les informations consolidées par arrêt avec les informations de correspondance SP.
    """

    # Calcul du retard total à l'arrivée par arrêt
    delay_by_stop = df_merge.groupby('id_stop')['arrival_delay'].sum()

    # Calcul du retard moyen par perturbation par arrêt
    mean_time_by_stop = df_merge.groupby('id_stop')['duration_disruption'].mean()

    # Calcul du retard médian par perturbation par arrêt
    median_time_by_stop = df_merge.groupby('id_stop')['duration_disruption'].median()

    # Nombre total de perturbations par arrêt
    most_impacted_stop = df_merge.groupby('id_stop')['id_disruption'].count()

    # Concaténation des données calculées
    df_stop_info = pd.concat(
        [delay_by_stop, round(delay_by_stop / most_impacted_stop), mean_time_by_stop, median_time_by_stop], axis=1)

    # Renommage des colonnes pour plus de clarté
    df_stop_info.columns = ['sum_arrival_delay', 'avg_arrival_delay_per_disruption', 'mean_duration_disruption',
                            'median_duration_disruption']

    # Réinitialisation de l'index pour pouvoir manipuler les données plus facilement
    df_stop_info.reset_index(inplace=True)

    # Extraction du code UIC à partir de l'identifiant d'arrêt et conversion en entier
    df_stop_info['Code UIC'] = df_stop_info['id_stop'].apply(
        lambda x: re.search(r'\d{8}', str(x)).group() if re.search(r'\d{8}', str(x)) else None).astype(int)

    # Jointure avec les informations supplémentaires sur les arrêts (df_sp) en utilisant le Code UIC
    df_stop_info = pd.merge(df_stop_info.drop('id_stop', axis=1), df_sp, on='Code UIC', how='left')

    # Conversion des durées de perturbation moyenne et médiane en heures
    df_stop_info['mean_duration_disruption'] = round(df_stop_info['mean_duration_disruption'].dt.total_seconds() / 3600,
                                                     2)
    df_stop_info['median_duration_disruption'] = round(
        df_stop_info['median_duration_disruption'].dt.total_seconds() / 3600, 2)

    # Gestion des valeurs manquantes dans les colonnes des informations SP
    df_stop_info['Nom de la gare'].fillna('', inplace=True)
    df_stop_info['Commune'].fillna('', inplace=True)
    df_stop_info['Département'].fillna('', inplace=True)
    df_stop_info['Code UIC'].fillna('', inplace=True)
    df_stop_info['Région SNCF'].fillna('', inplace=True)
    df_stop_info['LIBELLE'].fillna('', inplace=True)

    # Renommage final des colonnes pour une meilleure lisibilité
    df_stop_info.rename(columns={'Nom de la gare': 'Gare', 'LIBELLE': 'Région'}, inplace=True)

    return df_stop_info
def get_data_page3_col2(df_merge: pd.DataFrame, df_sp: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Récupère les informations de la colonne 2 pour la page 3.

    Args:
    - df_merge (pd.DataFrame): DataFrame contenant les données fusionnées.
    - df_sp (pd.DataFrame): DataFrame contenant les informations supplémentaires sur les arrêts.

    Returns:
    - Tuple[pd.DataFrame]: Tuple de DataFrames contenant les informations sur les perturbations par arrêt et les détails de perturbation par arrêt.
    """

    # Nombre total de perturbations par arrêt
    most_impacted_stop = df_merge.groupby('id_stop')['id_disruption'].count()

    # Groupement des causes de retard par arrêt
    cause_group_by_stop = pd.DataFrame(
        df_merge.groupby(['group_delay', 'id_stop'])['cause_delay'].count().reset_index())

    # Pivotement pour obtenir un tableau plus lisible des causes de retard par arrêt
    cause_group_by_stop_piv = cause_group_by_stop.pivot_table(index='id_stop', columns='group_delay',
                                                              values='cause_delay', aggfunc='sum',
                                                              fill_value=0).reset_index()

    # Jointure avec les informations supplémentaires sur les arrêts (df_sp) en utilisant le Code UIC
    df_stop_disruption_info = pd.merge(cause_group_by_stop_piv, df_merge[['name_stop', 'id_stop']],
                                       on='id_stop').drop_duplicates(subset=['id_stop'])

    # Ajout du total de perturbations par arrêt
    df_stop_disruption_info['total'] = most_impacted_stop.values

    # Extraction du code UIC à partir de l'identifiant d'arrêt et conversion en entier
    df_stop_disruption_info['Code UIC'] = df_stop_disruption_info['id_stop'].apply(
        lambda x: re.search(r'\d{8}', str(x)).group() if re.search(r'\d{8}', str(x)) else None).astype(int)

    # Jointure avec les informations supplémentaires sur les arrêts (df_sp) en utilisant le Code UIC
    df_stop_disruption_info = pd.merge(df_stop_disruption_info.drop('id_stop', axis=1), df_sp,
                                       on='Code UIC', how='left')

    # Gestion des valeurs manquantes dans les colonnes des informations SP
    df_stop_disruption_info['Commune'].fillna('', inplace=True)
    df_stop_disruption_info['Département'].fillna('', inplace=True)
    df_stop_disruption_info['Code UIC'].fillna('', inplace=True)
    df_stop_disruption_info['Région SNCF'].fillna('', inplace=True)
    df_stop_disruption_info['LIBELLE'].fillna('', inplace=True)
    df_stop_disruption_info['Région SNCF'].fillna('', inplace=True)

    # Ajout de colonnes pour les différentes causes de perturbation si elles ne sont pas déjà présentes
    for col in (
            INCIVILITE + PROBLEMES_TECHNIQUES + CONDITIONS_OPERATIONNELLES + CONDITIONS_EXTERNES + ASSISTANCE_PASSAGERS):
        if col not in df_stop_disruption_info.columns:
            df_stop_disruption_info[col] = 0

    # Renommage final des colonnes pour une meilleure lisibilité
    df_stop_disruption_info.rename(columns={'Nom de la gare': 'Gare', 'LIBELLE': 'Région'}, inplace=True)

    # Génération du DataFrame détaillé sur les causes de perturbation par arrêt
    cause_by_stop = pd.DataFrame(df_merge.groupby(['id_stop', 'cause_delay'])['cause_delay'].count())
    cause_by_stop.rename(columns={'cause_delay': 'count'}, inplace=True)
    cause_by_stop.reset_index(inplace=True)

    cause_by_stop_piv = cause_by_stop.pivot_table(
        index='id_stop',
        columns='cause_delay',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    df_stop_disruption_info_detail = pd.merge(cause_by_stop_piv, df_merge[['name_stop', 'id_stop']],
                                              on='id_stop').drop_duplicates(subset=['id_stop'])

    df_stop_disruption_info_detail['total'] = most_impacted_stop.values

    df_stop_disruption_info_detail['Code UIC'] = df_stop_disruption_info_detail['id_stop'].apply(
        lambda x: re.search(r'\d{8}', str(x)).group() if re.search(r'\d{8}', str(x)) else None).astype(int)

    df_stop_disruption_info_detail = pd.merge(df_stop_disruption_info_detail.drop('id_stop', axis=1), df_sp,
                                              on='Code UIC', how='left')

    # Gestion des valeurs manquantes dans les colonnes des informations SP
    df_stop_disruption_info_detail['Commune'].fillna('', inplace=True)
    df_stop_disruption_info_detail['Département'].fillna('', inplace=True)
    df_stop_disruption_info_detail['Code UIC'].fillna('', inplace=True)
    df_stop_disruption_info_detail['LIBELLE'].fillna('', inplace=True)
    df_stop_disruption_info_detail['Région SNCF'].fillna('', inplace=True)

    # Renommage final des colonnes pour une meilleure lisibilité
    df_stop_disruption_info_detail.rename(columns={'Nom de la gare': 'Gare', 'REG': 'Région'}, inplace=True)

    return df_stop_disruption_info, df_stop_disruption_info_detail
def page3(df_merge: pd.DataFrame, df_sp: pd.DataFrame, start: datetime, end: datetime,
          start_time:time, end_time: time) -> None:
    df_stop_info = get_data_page3_col1(df_merge, df_sp)
    df_stop_disruption_info, df_stop_disruption_info_detail = get_data_page3_col2(df_merge, df_sp)

    col1, col2 = st.columns(2)

    with col1:
        graph_bar_chart_gare(df_stop_info, start, end)
    with col2:
        graph_bar_chart_perturbation_gare(df_stop_disruption_info, df_stop_disruption_info_detail, start, end)

    with st.expander('A venir', expanded=False):
        st.info('Possibilité de selectionnez les gares/communes/ect. les plus impactées directement')
        st.info('Carte couleur pour une vue globale')
def get_data_page4_col1(df_disruption_unique: pd.DataFrame) -> Tuple[List[datetime], List[str], pd.DataFrame]:
    """
    Récupère les données de la colonne 1 pour la page 4.

    Args:
    - df_disruption_unique (pd.DataFrame): DataFrame contenant les données uniques de perturbation.

    Returns:
    - Tuple[List[datetime], List[str], pd.DataFrame]: Tuple contenant les intervalles de temps, les noms des intervalles et les données de perturbation par intervalle de temps.
    """

    # Appel de la fonction pour obtenir les intervalles de temps de 10 heures
    ten_intervals, hours_intervals_name = interval_hour()

    # Calcul des perturbations à travers la journée par intervalle de 10 heures
    disruption_by_ten = disruption_across_day(ten_intervals, df_disruption_unique)

    return ten_intervals, hours_intervals_name, disruption_by_ten
def get_data_page4_col2(df_disruption_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Récupère les données de la colonne 2 pour la page 4.

    Args:
    - df_disruption_unique (pd.DataFrame): DataFrame contenant les données uniques de perturbation.

    Returns:
    - pd.DataFrame: DataFrame contenant les détails de causes de retard par date, type de groupe de retard et cause de retard.
    """

    # Calcul du nombre de perturbations par date, type de groupe de retard et cause de retard
    cause_delay_detail = pd.DataFrame(
        df_disruption_unique.groupby(['data_date', 'group_delay', 'cause_delay'])['cause_delay'].count())

    cause_delay_detail.rename(columns={'cause_delay': 'count'}, inplace=True)
    cause_delay_detail.reset_index(inplace=True)

    # Calcul du retard total d'arrivée par date, type de groupe de retard et cause de retard
    delay_by_cause = df_disruption_unique.groupby(['data_date', 'group_delay', 'cause_delay'])[
        'arrival_delay'].sum().reset_index()

    # Calcul du retard moyen d'arrivée par perturbation par date, type de groupe de retard et cause de retard
    delay_by_cause['mean_arrival_delay'] = delay_by_cause['arrival_delay'] / cause_delay_detail['count']

    # Calcul de la durée totale de perturbation par date, type de groupe de retard et cause de retard
    duration_by_cause = df_disruption_unique.groupby(['data_date', 'group_delay', 'cause_delay'])[
        'duration_disruption'].sum().reset_index()

    # Calcul de la durée moyenne de perturbation par perturbation par date, type de groupe de retard et cause de retard
    duration_by_cause['mean_duration_disruption'] = pd.to_timedelta(
        duration_by_cause['duration_disruption'] / cause_delay_detail['count']).dt.floor('s')

    # Fusion des données calculées
    cause_delay_detail = pd.merge(cause_delay_detail, delay_by_cause, on=['data_date', 'group_delay', 'cause_delay'])
    cause_delay_detail = pd.merge(cause_delay_detail, duration_by_cause, on=['data_date', 'group_delay', 'cause_delay'])

    return cause_delay_detail
def page4(df_disruption_unique: pd.DataFrame, start: datetime, end: datetime, start_time:time, end_time: time) -> None:

    (ten_intervals,
     hours_intervals_name,
     disruption_by_ten) = get_data_page4_col1(df_disruption_unique)
    cause_delay_detail = get_data_page4_col2(df_disruption_unique)

    col1, col2 = st.columns(2)

    with col1:
        graph_disruption_across_day(disruption_by_ten, hours_intervals_name, start, end)
        st.info('Nombre de trains en circulation et pourcentage de train perturbé à venir')

    with col2:
        graph_disruption_across_day_type(df_disruption_unique, start, end)

    graph_cause_delay_info(cause_delay_detail, start, end)
def get_data_page5(df_merge: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Récupère les données de la page 5.

    Args:
    - df_merge (pd.DataFrame): DataFrame contenant les données fusionnées.

    Returns:
    - Tuple[pd.DataFrame]: Tuple contenant deux DataFrames : df_route_disruption_info et df_route_disruption_info_detail.
      df_route_disruption_info contient les informations agrégées par route.
      df_route_disruption_info_detail contient les détails des causes de perturbation par route.
    """

    # Nombre de trajet par route
    vehicle_by_route = df_merge.groupby('route_name')['vehicle_id'].count()

    # Nombre de perturbation par route
    disruption_by_route = df_merge.groupby('route_name')['id_disruption'].count()

    # Pourcentage de trajet perturbé par route
    per_disruption_by_route = disruption_by_route / vehicle_by_route * 100
    per_disruption_by_route = per_disruption_by_route.replace([np.inf, -np.inf], 0).dropna()

    # Création du DataFrame df_route_disruption contenant les informations agrégées par route
    df_route_disruption = pd.DataFrame({
        'vehicle_count': vehicle_by_route,
        'disruption_count': disruption_by_route,
        'percentage_disrupted': per_disruption_by_route
    }).reset_index()

    # Calcul du nombre de causes de perturbation par groupe de retard et par route
    cause_group_by_route = df_merge.groupby(['group_delay', 'route_name'])['cause_delay'].count().reset_index()

    # Pivotage des données pour avoir les groupes de retard en colonnes
    cause_group_by_route_piv = cause_group_by_route.pivot_table(index='route_name', columns='group_delay',
                                                                values='cause_delay', aggfunc='sum',
                                                                fill_value=0).reset_index()

    # Fusion des informations agrégées par route avec les données pivotées par groupe de retard
    df_route_disruption_info = pd.merge(df_route_disruption, cause_group_by_route_piv, on='route_name').sort_values(
        by='percentage_disrupted', ascending=False)

    # Calcul du nombre de causes de perturbation détaillées par route
    cause_by_route = pd.DataFrame(df_merge.groupby(['route_name', 'cause_delay'])['cause_delay'].count())
    cause_by_route.rename(columns={'cause_delay': 'count'}, inplace=True)
    cause_by_route.reset_index(inplace=True)

    # Pivotage des données pour avoir les causes de perturbation détaillées en colonnes
    cause_by_route_piv = cause_by_route.pivot_table(index='route_name', columns='cause_delay', values='count',
                                                    aggfunc='sum', fill_value=0).reset_index()

    # Fusion des informations agrégées par route avec les données pivotées par causes de perturbation détaillées
    df_route_disruption_info_detail = pd.merge(df_route_disruption, cause_by_route_piv, on='route_name').sort_values(
        by='percentage_disrupted', ascending=False)

    return df_route_disruption_info, df_route_disruption_info_detail
def page5(df_merge: pd.DataFrame, start_time:time, end_time: time) -> None:
    df_route_disruption_info, df_route_disruption_info_detail = get_data_page5(df_merge)

    st.info('A venir')
############################################################################################


#################################### MAIN #################################################
def main() -> None:
    (df_disruption,
     df_vehicle,
     df_lines,
     df_disruption_unique,
     df_merge,
     df_past,
     df_ref,
     df_sp) = load_and_clean_data()

    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    selected_dates = None
    start = (datetime.now() - timedelta(days=datetime.now().weekday()) - timedelta(days=7)).strftime('%d/%m/%Y')
    end = (datetime.now() + timedelta(days=(-1 - datetime.now().weekday()))).strftime('%d/%m/%Y')

    time_input_start= time(0, 00)
    time_input_end = time(23, 59, 59, 59)

    with st.sidebar:
        selected = option_menu(
            menu_title="Tableaux de bord",
            options=[
                "Analyse générale",
                "Détail des causes",
                "Détail géographique",
                "Détail par réseau",
                "Détail par route"
            ]
        )
        is_default = st.toggle("Dates par défaut", value=True)
        if not is_default:
            selected_dates = st.date_input(
                "Choisissez vos dates",
                (pd.to_datetime(df_vehicle['data_date'].iloc[0]),
                 pd.to_datetime(df_vehicle['data_date'].iloc[len(df_vehicle) - 1])),
                pd.to_datetime(df_vehicle['data_date'].iloc[0]),
                pd.to_datetime(df_vehicle['data_date'].iloc[len(df_vehicle) - 1]),
                key='Choose date for data'
            )
            start, end = selected_dates
            start = start.strftime('%d/%m/%Y')
            end = end.strftime('%d/%m/%Y')

            col1, col2 = st.columns(2)

            with col1:
                time_input_start = st.time_input(
                    label="",
                    value=time(0, 00),
                    key="start",
                    step=1800,
                )
            with col2:
                time_input_end = st.time_input(
                    label="",
                    value=time(23, 59),
                    key="end",
                    step=1800
                )
        if time_input_start >= time_input_end:
            st.warning("L'heure de début doit etre inferieur à l'heure de fin")

    (df_merge, df_vehicle) = get_data_selected_date(df_merge, df_vehicle, selected_dates=selected_dates,
                                                  start_time=time_input_start, end_time=time_input_end)
    (df_disruption,
    df_disruption_unique) = get_data_selected_date2(df_disruption, df_disruption_unique, selected_dates=selected_dates)

    if selected == "Analyse générale":
        page1(df_vehicle, df_disruption, df_past, start, end, days, is_default, time_input_start, time_input_end)
    elif selected == "Détail par réseau":
        page2(df_merge, start, end, time_input_start, time_input_end)
    elif selected == "Détail géographique":
        page3(df_merge, df_sp, start, end, time_input_start, time_input_end)
    elif selected == "Détail des causes":
        page4(df_disruption_unique, start, end, time_input_start, time_input_end)
    elif selected =="Détail par route":
        page5(df_merge, time_input_start, time_input_end)
############################################################################################


if __name__ == "__main__":
    main()