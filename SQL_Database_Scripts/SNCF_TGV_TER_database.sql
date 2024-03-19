-- Création de la base de donnée qui contient les informations sur les réseaux TGV et TER
CREATE DATABASE SNCF_TGV_TER;
USE SNCF_TGV_TER;

-- Création de la table contenant les informations sur les perturbations des TGV et TER
CREATE TABLE disruptions_tgv_ter (
    id_disruption CHAR(100),
    vehicle_id CHAR(100),
    train_type VARCHAR(50),
    id_stop CHAR(100),
    name_stop VARCHAR(100),
    lon FLOAT,
    lat FLOAT,
    disruption_start DATETIME,
    disruption_end DATETIME,
    arrival_delay FLOAT,
    departure_delay FLOAT,
    cause_delay VARCHAR(100),
    data_date DATE,
    PRIMARY KEY (vehicle_id, id_disruption, id_stop),
	INDEX idx_disruption_time (disruption_start, disruption_end)
);

-- Création de la table contenant les informations sur l'esemble des trajets (TGV et TER)
CREATE TABLE vehicle_journeys_tgv_ter(
    vehicle_id CHAR(100) PRIMARY KEY,
    route_id CHAR(100),
    time_begin TIME,
    time_end TIME,
    train_type VARCHAR(50), 
    id_disruption CHAR(100),
    data_date DATE
);

-- Création de la table contenant les informations sur les différentes routes des TGV et TER
CREATE TABLE pt_lines_tgv_ter (
    route_id CHAR(100) PRIMARY KEY,
    route_name VARCHAR(100),
    train_type VARCHAR(50),
    network_name VARCHAR(100),
    opening_time TIME,
    closing_time TIME
);