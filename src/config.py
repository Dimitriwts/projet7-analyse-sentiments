"""
Configuration centrale du projet.

Tous les chemins de fichiers et toutes les constantes de modelisation sont
definis ICI, et nulle part ailleurs. Les notebooks, les scripts et l'API
importent ce module plutot que de recopier des chemins en dur.

Pourquoi ? Parce qu'une experimentation n'est reproductible que si l'on sait
exactement avec quels parametres elle a tourne. En centralisant, on evite le
grand classique du projet de data science : deux notebooks qui utilisent en
douce deux tailles d'echantillon differentes, et des scores incomparables.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. CHEMINS DU PROJET
# ---------------------------------------------------------------------------
# Path(__file__) = ce fichier (src/config.py)
#   .resolve()   = son chemin absolu
#   .parent      = le dossier src/
#   .parent      = la racine du projet
#
# On procede ainsi plutot qu'avec un chemin ecrit en dur : le projet
# fonctionne alors quel que soit l'endroit ou il est clone, et quel que soit
# le dossier depuis lequel on lance Python.
RACINE_PROJET = Path(__file__).resolve().parent.parent

DOSSIER_DONNEES = RACINE_PROJET / "data"
DOSSIER_DONNEES_BRUTES = DOSSIER_DONNEES / "raw"
DOSSIER_DONNEES_PREPAREES = DOSSIER_DONNEES / "processed"

DOSSIER_MODELES = RACINE_PROJET / "models"
DOSSIER_API = RACINE_PROJET / "api"
DOSSIER_ARTEFACTS_API = DOSSIER_API / "artefacts"
DOSSIER_DOCS = RACINE_PROJET / "docs"

# Fichier brut attendu : le dataset Sentiment140 tel qu'il est distribue.
# Si votre fichier porte un autre nom, le chargeur (src/data_loader.py) le
# detectera automatiquement dans data/raw/ et vous le signalera.
FICHIER_DATASET_BRUT = DOSSIER_DONNEES_BRUTES / "training.1600000.processed.noemoticon.csv"

# Echantillon nettoye, produit une seule fois par le notebook d'exploration
# puis reutilise tel quel par les trois notebooks de modelisation.
# C'est la garantie que les trois approches sont comparees sur EXACTEMENT
# les memes tweets.
FICHIER_DONNEES_PREPAREES = DOSSIER_DONNEES_PREPAREES / "tweets_prepares.parquet"


# ---------------------------------------------------------------------------
# 2. REPRODUCTIBILITE
# ---------------------------------------------------------------------------
# Une seule graine aleatoire pour tout le projet : melange des donnees,
# decoupage train/test, initialisation des poids des reseaux de neurones.
# Relancer un notebook doit redonner le meme resultat.
GRAINE_ALEATOIRE = 42


# ---------------------------------------------------------------------------
# 3. STRUCTURE DU DATASET SENTIMENT140
# ---------------------------------------------------------------------------
# Le fichier CSV n'a PAS de ligne d'en-tete : on doit donner les noms nous-memes.
# Les 6 colonnes, dans l'ordre, sont documentees par les auteurs du dataset.
COLONNES_DATASET = ["polarite", "identifiant", "date", "requete", "utilisateur", "texte"]

# La colonne "polarite" vaut 0 (negatif) ou 4 (positif) dans le fichier brut.
# On la convertit en label binaire classique 0/1.
POLARITE_NEGATIVE_BRUTE = 0
POLARITE_POSITIVE_BRUTE = 4

# Encodage du fichier : le dataset contient des caracteres non-UTF8
# (accents mal encodes, caracteres de controle). latin-1 accepte tous les
# octets sans lever d'erreur, c'est l'encodage recommande pour ce fichier.
ENCODAGE_DATASET = "latin-1"


# ---------------------------------------------------------------------------
# 4. TAILLES D'ECHANTILLON PAR APPROCHE
# ---------------------------------------------------------------------------
# Le dataset complet fait 1,6 million de tweets. Les trois approches n'ont
# pas du tout le meme cout de calcul, donc pas la meme taille d'echantillon.
# Ces valeurs sont volontairement calibrees pour que chaque entrainement
# tienne en quelques minutes sur un processeur, sans carte graphique.
#
# Ces chiffres seront ajustes apres une mesure reelle du temps de calcul
# (voir le notebook 03) : on ne devine pas, on mesure.

# Approche 1 - Modele classique (TF-IDF + regression logistique).
# Tres peu couteux : on peut se permettre l'integralite du dataset.
TAILLE_ECHANTILLON_CLASSIQUE = None  # None = tout le dataset

# Approche 2 - Modele avance sur mesure (word embeddings + reseau recurrent).
TAILLE_ECHANTILLON_AVANCE = 300_000

# Approche 3 - Modele BERT.
# De loin le plus couteux : un modele pre-entraine de 66 millions de parametres.
TAILLE_ECHANTILLON_BERT = 50_000

# Part des donnees reservee au test final, identique pour les trois approches.
PROPORTION_TEST = 0.2


# ---------------------------------------------------------------------------
# 5. PARAMETRES DU MODELE AVANCE (reseau de neurones sur mesure)
# ---------------------------------------------------------------------------
# Nombre de mots conserves dans le vocabulaire. Au-dela, les mots trop rares
# apportent surtout du bruit et alourdissent inutilement le modele.
TAILLE_VOCABULAIRE = 20_000

# Longueur maximale d'un tweet en nombre de mots. Les tweets plus courts sont
# completes par des zeros, les plus longs sont tronques.
# 40 couvre la tres grande majorite des tweets (voir notebook 01).
LONGUEUR_MAX_SEQUENCE = 40

# Dimension des vecteurs de mots (word embeddings).
DIMENSION_EMBEDDING = 200

# Token utilise pour tout mot absent du vocabulaire.
TOKEN_MOT_INCONNU = "<INCONNU>"


# ---------------------------------------------------------------------------
# 6. PARAMETRES DU MODELE BERT
# ---------------------------------------------------------------------------
# DistilBERT : version allegee de BERT (40 % plus petite, 60 % plus rapide,
# environ 97 % des performances). Le choix raisonnable sans carte graphique.
NOM_MODELE_BERT = "distilbert-base-uncased"

# Longueur maximale en tokens BERT. Le cout de calcul de BERT croit avec le
# carre de cette valeur : la garder basse est le principal levier de rapidite.
LONGUEUR_MAX_BERT = 64


# ---------------------------------------------------------------------------
# 7. SUIVI DES EXPERIMENTATIONS (MLflow)
# ---------------------------------------------------------------------------
# Adresse du serveur MLflow local, lance par la commande documentee dans le
# README. On passe par un serveur (et non par de simples fichiers) car le
# Model Registry - la "centralisation du stockage des modeles" - exige une
# base de donnees comme support.
URI_SUIVI_MLFLOW = "http://127.0.0.1:5000"

# Un nom d'experience par approche : les runs restent ranges et comparables.
EXPERIENCE_CLASSIQUE = "01_modele_classique"
EXPERIENCE_AVANCE = "02_modele_avance"
EXPERIENCE_BERT = "03_modele_bert"

# Nom sous lequel le modele retenu est enregistre dans le Model Registry.
# C'est ce modele-la que l'API sert en production.
NOM_MODELE_ENREGISTRE = "analyse-sentiment-air-paradis"
