from sqlalchemy import create_engine
from email.message import EmailMessage
from Info import password
import smtplib
import ssl
import os
import time
import requests

def fetch_page(url, headers, params):
    """Effectue une requête API pour une page spécifique et retourne les données."""
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur lors de la récupération des données de la page {params['start_page']}: {response.status_code}")
        return None

def send_error_email(error, table_name):
    """
    Cette fonction envoie un email pour notifier d'une erreur survenue lors d'opérations
    sur une base de données. Les informations nécessaires pour l'envoi de l'email sont récupérées
    depuis les variables d'environnement.

    Parameters:
    error (str): Description de l'erreur survenue.
    table_name (str): Nom de la table de la base de données concernée par l'erreur.
    """

    # Récupération des informations d'envoi depuis les variables d'environnement.
    sender_email = os.getenv('gmail_sncf_project')
    email_password = password

    receiver_email = os.getenv('my_gmail')

    # Préparation de l'email
    subject = f"NOTIFICATION D'ERREUR {table_name}"
    body = f"""\
    Une erreur s'est produite sur la table {table_name} :
    {error}
    """

    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)


    try:
        # Établissement d'une connexion sécurisée avec le serveur SMTP de Gmail.
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, email_password)  # Authentification.
            smtp.sendmail(sender_email, receiver_email, em.as_string())  # Envoi de l'email.
        print("L'email d'erreur a été envoyé avec succès.")  # Confirmation d'envoi.
    except Exception as e:
        # Gestion des erreurs éventuelles lors de l'envoi.
        print(f"Erreur lors de l'envoi de l'email : {e}")

def to_csv_file(df, file_path):
    """
    Enregistre les données d'un DataFrame Pandas dans un fichier CSV spécifique.

    Cette fonction prend en entrée un DataFrame Pandas (`df`) et le chemin d'un fichier CSV (`file_path`)
    où les données doivent être ajoutées. Si le fichier existe déjà, les nouvelles données sont ajoutées à la suite.
    Si le fichier n'existe pas, il est créé avec les en-têtes.

    Args:
        df (DataFrame): Le DataFrame contenant les données à insérer dans le fichier CSV.
        file_path (str): Le chemin du fichier CSV où les données doivent être ajoutées.
    """

    try:
        # Vérifie si le fichier existe
        if os.path.isfile(file_path):
            # Si le fichier existe, ajoute les données sans les en-têtes
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Si le fichier n'existe pas, crée le fichier avec les en-têtes
            df.to_csv(file_path, mode='w', header=True, index=False)
        print(f'Data successfully written to {file_path}.')
    except Exception as e:
        # En cas d'erreur, affichage du message d'erreur
        print(f"An error occurred: {e}")

def make_request_with_retry(url, headers, params=None, max_retries=5):
    """
    Tente une requête HTTP avec une logique de nouvelle tentative en cas d'erreur 503.

    Args:
        url (str): URL de la requête.
        headers (dict): En-têtes de la requête.
        params (dict, optional): Paramètres de la requête. Defaults to None.
        max_retries (int, optional): Nombre maximal de tentatives. Defaults to 5.

    Returns:
        requests.models.Response: Réponse de la requête.
    """
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 503:  # Service temporairement indisponible
            wait = 2 ** attempt  # Délai exponentiel
            time.sleep(wait)
            continue
        return response  # Retourne la réponse en cas de succès ou d'autre erreur que 503
    return response  # Retourne la dernière réponse après toutes les tentatives