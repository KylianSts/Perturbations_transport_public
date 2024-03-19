-- Création de la base de donnée qui contient les informations sur le réseau ile-de-france
CREATE DATABASE SNCF_IDF;
USE SNCF_IDF;

-- Création de la table contenant les informations sur les perturbations
CREATE TABLE disruptions_idf (
    id_disruption VARCHAR(100) PRIMARY KEY,
    disruption_start DATETIME,
    disruption_end DATETIME,
    line_code VARCHAR(100),
    vehicle_type VARCHAR(50),
    network_id VARCHAR(50),
    impacted_stop_from_id VARCHAR(50),
    impacted_stop_to_id VARCHAR(50),
    data_date DATE,
	INDEX idx_disruption_time (disruption_start, disruption_end)
);

-- Création de la table contenant les informations sur l'esemble des trajets 
CREATE TABLE vehicle_journeys_idf (
    vehicle_id VARCHAR(100) PRIMARY KEY,
    line_id VARCHAR(100),
    vehicle_name VARCHAR(100),
    first_stop_id VARCHAR(50),
    last_stop_id VARCHAR(50),
    id_disruption VARCHAR(100),
    data_date DATE,
    FOREIGN KEY (id_disruption) REFERENCES disruptions_idf(id_disruption) ON DELETE SET NULL ON UPDATE CASCADE
);

-- Création de la table contenant les informations sur les différentes lignes du réseau ile-de-france (hors bus)
CREATE TABLE pt_lines_idf (
    line_id VARCHAR(100) PRIMARY KEY,
    line_code VARCHAR(100),
    line_color VARCHAR(20),
    vehicle_type VARCHAR(50),
    opening_time TIME,
    closing_time TIME,
    network_id VARCHAR(50),
    network_name VARCHAR(100)
);
