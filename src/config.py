"""
Toutes les constantes du projet au même endroit.

Les notebooks, les scripts et l'API importent ce fichier au lieu de recopier
des chemins et des réglages un peu partout.

Pourquoi je fais ça : si la taille du vocabulaire est écrite en dur dans trois
notebooks et que j'en modifie deux, le troisième continue de tourner avec
l'ancienne valeur et je compare des scores qui ne veulent plus rien dire.
En centralisant, je change une ligne et tout le projet suit.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. LES CHEMINS DU PROJET
# ---------------------------------------------------------------------------
# Path(__file__) désigne ce fichier, c'est-à-dire src/config.py.
#   .resolve()  donne son chemin complet
#   .parent     remonte au dossier src/
#   .parent     remonte encore, à la racine du projet
#
# Je passe par là plutôt que d'écrire "C:/Users/..." en dur, pour deux raisons :
# le projet marche sur n'importe quelle machine, et il marche quel que soit le
# dossier depuis lequel je lance Python.
RACINE_PROJET = Path(__file__).resolve().parent.parent

DOSSIER_DONNEES = RACINE_PROJET / "data"
DOSSIER_DONNEES_BRUTES = DOSSIER_DONNEES / "raw"
DOSSIER_DONNEES_PREPAREES = DOSSIER_DONNEES / "processed"

DOSSIER_MODELES = RACINE_PROJET / "models"
DOSSIER_API = RACINE_PROJET / "api"
DOSSIER_ARTEFACTS_API = DOSSIER_API / "artefacts"
DOSSIER_DOCS = RACINE_PROJET / "docs"

# Le fichier de données tel qu'on le télécharge. Si le vôtre porte un autre
# nom, le chargeur (src/data_loader.py) va quand même le trouver et vous
# le dire.
FICHIER_DATASET_BRUT = DOSSIER_DONNEES_BRUTES / "training.1600000.processed.noemoticon.csv"

# Le fichier de travail, produit une seule fois par le notebook 01 et relu
# ensuite par les trois notebooks de modélisation. C'est ce qui garantit que
# les trois approches travaillent sur exactement les mêmes tweets.
FICHIER_DONNEES_PREPAREES = DOSSIER_DONNEES_PREPAREES / "tweets_prepares.parquet"


# ---------------------------------------------------------------------------
# 2. LA GRAINE ALÉATOIRE
# ---------------------------------------------------------------------------
# Beaucoup d'opérations en apprentissage automatique tirent au sort : quels
# tweets vont dans le jeu de test, dans quel ordre on les présente au modèle,
# avec quelles valeurs de départ on initialise un réseau de neurones.
#
# Une "graine" (seed en anglais) est un nombre qui sert de point de départ au
# tirage au sort. Si je fixe la graine, le tirage donne toujours le même
# résultat. Sans ça, je relance un notebook et j'obtiens un score légèrement
# différent, sans savoir si c'est mon changement qui a agi ou juste le hasard.
#
# 42 est une valeur arbitraire, c'est une convention très répandue.
GRAINE_ALEATOIRE = 42


# ---------------------------------------------------------------------------
# 3. À QUOI RESSEMBLE LE FICHIER DE DONNÉES
# ---------------------------------------------------------------------------
# Le fichier CSV (Comma-Separated Values, un tableau où les colonnes sont
# séparées par des virgules) n'a pas de ligne de titre. Il faut donc donner
# les noms des colonnes soi-même, dans l'ordre où elles apparaissent.
COLONNES_DATASET = ["polarite", "identifiant", "date", "requete", "utilisateur", "texte"]

# Dans le fichier d'origine, la colonne "polarite" vaut 0 pour un tweet négatif
# et 4 pour un tweet positif. Je convertis ensuite en 0 et 1, qui est la façon
# habituelle de coder deux classes.
POLARITE_NEGATIVE_BRUTE = 0
POLARITE_POSITIVE_BRUTE = 4

# L'encodage, c'est la façon dont les caractères sont traduits en octets dans
# le fichier. Ce fichier n'est pas en UTF-8 (l'encodage moderne standard) et
# contient des octets qui feraient planter la lecture. L'encodage latin-1
# accepte tous les octets sans broncher, c'est celui qui est recommandé pour
# ce jeu de données.
ENCODAGE_DATASET = "latin-1"


# ---------------------------------------------------------------------------
# 4. COMBIEN DE TWEETS POUR CHAQUE APPROCHE
# ---------------------------------------------------------------------------
# Le jeu de données complet fait 1,6 million de tweets. Les trois approches ne
# coûtent pas du tout le même temps de calcul, donc je ne leur donne pas la
# même quantité de données. Ces valeurs sont calibrées pour que chaque
# entraînement tienne en quelques minutes sur un processeur ordinaire, sans
# carte graphique dédiée.

# Approche 1, le modèle classique. Très rapide, je peux tout lui donner.
TAILLE_ECHANTILLON_CLASSIQUE = None  # None veut dire "tout le jeu de données"

# Approche 2, le réseau de neurones sur mesure.
TAILLE_ECHANTILLON_AVANCE = 300_000

# Approche 3, BERT. De loin la plus lourde : c'est un modèle déjà entraîné qui
# contient 66 millions de paramètres.
#
# Cette taille est le résultat direct d'une mesure faite avant d'écrire le
# notebook, et pas d'une estimation. Voici le débit de DistilBERT sur ce
# processeur, sans carte graphique :
#
#     BERT figé, calculs "aller" seulement   32,7 tweets par seconde
#     BERT affiné, "aller" et "retour"        7,2 tweets par seconde
#
# Autrement dit, affiner BERT sur 50 000 tweets aurait demandé près de
# 4 heures par passage sur les données. Avec 10 000 tweets, le notebook
# complet tient en 45 minutes, ce qui reste utilisable.
#
# Ce n'est pas un pis-aller : le coût de calcul fait partie de la réponse
# qu'attend le client. La question posée est de savoir s'il faut investir
# dans ce type de modèle, et un modèle 4,5 fois plus cher à entraîner qui
# n'apporterait qu'un point de performance ne vaudrait pas l'investissement.
TAILLE_ECHANTILLON_BERT = 10_000

# Part des tweets mise de côté pour le test final. La même pour les trois
# approches, sinon la comparaison ne veut rien dire.
PROPORTION_TEST = 0.2


# ---------------------------------------------------------------------------
# 5. RÉGLAGES DU MODÈLE AVANCÉ (le réseau de neurones sur mesure)
# ---------------------------------------------------------------------------
# Le vocabulaire, c'est la liste des mots que le modèle connaît. Je la limite
# aux 20 000 mots les plus fréquents. Au-delà, on tombe sur des mots vus deux
# ou trois fois en tout : ils apportent surtout du bruit et alourdissent le
# modèle pour rien.
TAILLE_VOCABULAIRE = 20_000

# Un réseau de neurones récurrent lit le tweet mot par mot, mais il a besoin
# que tous les tweets fassent la même longueur. Les plus courts sont complétés
# par des zéros, les plus longs sont coupés.
#
# Valeur choisie sur mesure, pas au hasard (voir le notebook 01). Voici la
# part des tweets nettoyés qui tiennent entièrement, selon le seuil :
#     24 mots  ->  94,40 %
#     28 mots  ->  99,46 %
#     32 mots  ->  99,985 %   <- ce que je retiens
#     40 mots  ->  99,998 %
# Le temps de calcul d'un réseau récurrent est proportionnel à cette longueur.
# Passer de 40 à 32 fait gagner 20 % de temps et ne coupe la fin que de
# 15 tweets sur 100 000.
LONGUEUR_MAX_SEQUENCE = 32

# Un "embedding" (plongement lexical en français, mais tout le monde dit
# embedding) est la façon de représenter un mot par une liste de nombres.
# Au lieu de dire "le mot terrible est le numéro 4271", on dit "le mot
# terrible, c'est ce vecteur de 200 nombres". L'intérêt : deux mots de sens
# proche ont des vecteurs proches, donc le modèle peut généraliser de "awful"
# à "terrible" même s'il a peu vu le second.
# 200 est un compromis courant entre finesse et taille du modèle.
DIMENSION_EMBEDDING = 200

# Quand l'API reçoit un mot qui n'était pas dans le vocabulaire d'entraînement,
# il faut bien lui donner quelque chose. On le remplace par ce jeton spécial.
# OOV veut dire "out of vocabulary", hors vocabulaire.
TOKEN_MOT_INCONNU = "<INCONNU>"


# ---------------------------------------------------------------------------
# 6. RÉGLAGES DU MODÈLE BERT
# ---------------------------------------------------------------------------
# BERT (Bidirectional Encoder Representations from Transformers) est un modèle
# de langue publié par Google en 2018. Il a été entraîné sur des milliards de
# mots à deviner des mots masqués dans des phrases, et il a ainsi appris le
# fonctionnement de l'anglais. On le récupère déjà entraîné et on l'adapte à
# notre tâche : c'est ce qu'on appelle le transfert d'apprentissage.
#
# DistilBERT est une version allégée de BERT : 40 % plus petite, 60 % plus
# rapide, et elle conserve environ 97 % des performances. Sans carte graphique,
# c'est le seul choix raisonnable.
NOM_MODELE_BERT = "distilbert-base-uncased"

# Longueur maximale du texte donné à BERT, en jetons. Le coût de calcul de
# BERT augmente comme le carré de cette valeur : la garder basse est le
# principal levier pour que ça tourne vite.
LONGUEUR_MAX_BERT = 64


# ---------------------------------------------------------------------------
# 7. SUIVI DES EXPÉRIMENTATIONS AVEC MLFLOW
# ---------------------------------------------------------------------------
# MLflow est l'outil qui garde la trace de chaque entraînement : les réglages
# utilisés, les scores obtenus, le temps passé, et le modèle lui-même. Sans
# lui, au bout de quinze essais je ne sais plus lequel était le meilleur ni
# avec quels réglages je l'avais obtenu.
#
# Adresse du serveur MLflow qui tourne en local. La commande pour le lancer
# est dans le README.
URI_SUIVI_MLFLOW = "http://127.0.0.1:5000"

# Une "expérience" MLflow est simplement un dossier de résultats. J'en fais
# une par approche pour que les essais restent rangés.
EXPERIENCE_CLASSIQUE = "01_modele_classique"
EXPERIENCE_AVANCE = "02_modele_avance"
EXPERIENCE_BERT = "03_modele_bert"

# Nom sous lequel le modèle retenu est rangé dans le Model Registry de MLflow.
# Le Model Registry est le catalogue des modèles : il leur donne un numéro de
# version et permet de désigner celui qui part en production.
NOM_MODELE_ENREGISTRE = "analyse-sentiment-air-paradis"
