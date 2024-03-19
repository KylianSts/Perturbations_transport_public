from sqlalchemy import create_engine
from email.message import EmailMessage
from Info import password
import smtplib
import ssl
import os

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

def to_sql_database(df, database_sql, table_name):
    """
    Enregistre les données d'un DataFrame Pandas dans une table SQL spécifique.

    Cette fonction prend en entrée un DataFrame Pandas (`df`), le nom d'une base de données SQL (`database_sql`)
    et le nom d'une table (`table_name`) où les données doivent être insérées. Elle utilise SQLAlchemy pour
    établir une connexion à la base de données et insérer les données. Si une erreur survient pendant l'insertion,
    un email d'erreur est envoyé avec les détails.

    Args:
        df (DataFrame): Le DataFrame contenant les données à insérer dans la table SQL.
        database_sql (str): Le nom de la base de données SQL cible.
        table_name (str): Le nom de la table SQL dans laquelle les données doivent être insérées.
    """

    # Configuration de la connexion à la base de données SQL.
    dialect = 'mysql+pymysql'  # Définit le dialecte SQL et le driver à utiliser.
    user = 'root'  # Nom d'utilisateur pour se connecter à la base de données.
    password = os.getenv('mdp_mySQL')  # Récupère le mot de passe de la base de données depuis une variable d'environnement.
    host = 'localhost'  # Adresse de l'hôte où la base de données est située.
    port = 3306  # Port utilisé pour la connexion à la base de données.

    # Construction de l'URL de connexion pour SQLAlchemy.
    engine_url = f"{dialect}://{user}:{password}@{host}:{port}/{database_sql}"
    engine = create_engine(engine_url)  # Création de l'objet moteur pour la connexion.

    try:
        # Tentative d'écriture des données du DataFrame à la table SQL spécifiée.
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f'Data successfully written to table {table_name}.')
    except Exception as e:
        # En cas d'erreur, affichage du message d'erreur et envoi d'un email d'erreur.
        print(f"An error occurred: {e}")
        send_error_email(e, table_name)  # Appel de la fonction pour envoyer un email d'erreur.
