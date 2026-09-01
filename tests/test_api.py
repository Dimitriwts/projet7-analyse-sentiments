"""
Les tests de l'API.

Le livrable demande des tests unitaires automatisés, exécutés par le pipeline
de déploiement avant chaque mise en ligne. Si l'un d'eux échoue, l'API n'est
pas déployée et la version en production reste celle qui marchait.

Ces tests se répartissent en quatre groupes :

  1. La synchronisation du nettoyage entre l'entraînement et la production.
     C'est le test le plus important du projet.
  2. Les artefacts du modèle sont présents et cohérents.
  3. Les routes de l'API répondent ce qu'on attend.
  4. Les entrées invalides sont refusées proprement, sans faire tomber le
     serveur.

Pour les lancer :
    .venv\\Scripts\\python.exe -m pytest
"""

import json
import sys
from pathlib import Path

import pytest

RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_API = RACINE_PROJET / "api"
DOSSIER_ARTEFACTS = DOSSIER_API / "artefacts"

# L'API importe `preprocessing` sans préfixe, parce qu'en production ce fichier
# est à côté d'elle. Pour que les tests puissent l'importer de la même façon,
# on ajoute le dossier api/ au chemin de recherche de Python.
if str(DOSSIER_API) not in sys.path:
    sys.path.insert(0, str(DOSSIER_API))


# ===========================================================================
# GROUPE 1 : le nettoyage est bien le même des deux côtés
# ===========================================================================
# C'est le test le plus important du projet, alors je prends le temps de dire
# pourquoi.
#
# Le tweet doit être nettoyé exactement de la même façon à l'entraînement et
# en production. Si les deux versions divergent, le modèle reçoit en ligne un
# texte qui ne ressemble pas à ce qu'il a appris, et ses réponses se dégradent
# sans qu'aucune erreur ne s'affiche nulle part. On appelle ça le
# training/serving skew, le décalage entre l'entraînement et le service.
#
# Le fichier api/preprocessing.py est une copie de src/preprocessing.py. Une
# copie, ça se met à jour d'un côté et pas de l'autre. Ce test rend cette
# situation impossible : le déploiement échoue avant la mise en ligne.

def test_le_nettoyage_est_identique_entre_src_et_api():
    """Les deux fichiers de nettoyage doivent etre rigoureusement identiques."""
    fichier_source = RACINE_PROJET / "src" / "preprocessing.py"
    fichier_api = DOSSIER_API / "preprocessing.py"

    assert fichier_source.exists(), "src/preprocessing.py est introuvable"
    assert fichier_api.exists(), "api/preprocessing.py est introuvable"

    contenu_source = fichier_source.read_bytes()
    contenu_api = fichier_api.read_bytes()

    assert contenu_source == contenu_api, (
        "src/preprocessing.py et api/preprocessing.py ont diverge.\n"
        "Le modele recevrait en production un texte nettoye differemment de "
        "celui sur lequel il a ete entraine.\n"
        "Pour corriger : copiez src/preprocessing.py vers api/preprocessing.py."
    )


def test_les_deux_modules_donnent_le_meme_resultat():
    """
    Verifie le comportement, et pas seulement le contenu des fichiers.

    Deux fichiers identiques donneront forcement le meme resultat, mais ce
    test protege aussi contre le cas ou l'un des deux importerait une version
    differente d'une bibliotheque.
    """
    from preprocessing import nettoyer_tweet as nettoyer_cote_api
    from src.preprocessing import nettoyer_tweet as nettoyer_cote_source

    exemples = [
        "@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz",
        "Great service &amp; friendly crew :)",
        "I do NOT recommend this airline",
        "",
        "12345 !!! ???",
    ]
    for exemple in exemples:
        assert nettoyer_cote_api(exemple) == nettoyer_cote_source(exemple)


# ===========================================================================
# GROUPE 2 : les artefacts du modèle sont là et cohérents
# ===========================================================================
# L'API a besoin de trois fichiers. S'il en manque un, ou si les métadonnées
# ne correspondent pas au modèle, mieux vaut le savoir avant le déploiement.

def test_les_trois_artefacts_sont_presents():
    """Le modele, le vocabulaire et les metadonnees doivent etre versionnes."""
    for nom in ("modele.tflite", "vocabulaire.json", "metadonnees.json"):
        fichier = DOSSIER_ARTEFACTS / nom
        assert fichier.exists(), (
            f"{nom} est absent de api/artefacts/.\n"
            "Ces fichiers sont produits par le notebook "
            "05_comparaison_et_export.ipynb."
        )
        assert fichier.stat().st_size > 0, f"{nom} est vide"


def test_les_metadonnees_contiennent_ce_qu_il_faut():
    """L'API lit ces cles au demarrage, elles doivent toutes exister."""
    metadonnees = json.loads(
        (DOSSIER_ARTEFACTS / "metadonnees.json").read_text(encoding="utf-8")
    )
    for cle in ("longueur_max_sequence", "index_mot_inconnu", "seuil_decision",
                "nom_modele", "architecture", "date_export"):
        assert cle in metadonnees, f"La cle {cle} manque dans metadonnees.json"

    assert 0 < metadonnees["seuil_decision"] < 1
    assert metadonnees["longueur_max_sequence"] > 0


def test_le_vocabulaire_est_coherent_avec_les_metadonnees():
    """Le jeton des mots inconnus doit exister dans le vocabulaire."""
    vocabulaire = json.loads(
        (DOSSIER_ARTEFACTS / "vocabulaire.json").read_text(encoding="utf-8")
    )
    metadonnees = json.loads(
        (DOSSIER_ARTEFACTS / "metadonnees.json").read_text(encoding="utf-8")
    )

    assert len(vocabulaire) > 1000, "Le vocabulaire est anormalement petit"
    assert metadonnees["index_mot_inconnu"] in vocabulaire.values(), (
        "L'index reserve aux mots inconnus n'existe pas dans le vocabulaire"
    )
    # L'index 0 est reserve au remplissage, aucun mot ne doit le porter.
    assert 0 not in vocabulaire.values(), (
        "L'index 0 est reserve au remplissage, il ne doit designer aucun mot"
    )


# ===========================================================================
# GROUPE 3 : les routes de l'API répondent correctement
# ===========================================================================
# TestClient de FastAPI simule des requetes HTTP sans lancer de vrai serveur.
# C'est ce qui permet de tester l'API dans un pipeline automatique, sans avoir
# a demarrer quoi que ce soit.

@pytest.fixture(scope="module")
def client():
    """Un client de test, partage par tous les tests de ce fichier."""
    from fastapi.testclient import TestClient

    from main import app

    # Le `with` declenche le demarrage de l'application, donc le chargement du
    # modele. Sans lui, les routes repondraient avant que le modele soit pret.
    with TestClient(app) as client_de_test:
        yield client_de_test


def test_la_page_d_accueil_repond(client):
    """La racine doit decrire l'API et le modele servi."""
    reponse = client.get("/")
    assert reponse.status_code == 200
    contenu = reponse.json()
    assert "modele" in contenu
    assert "documentation" in contenu


def test_la_route_de_sante_repond(client):
    """Azure interroge cette route pour verifier que l'application vit."""
    reponse = client.get("/sante")
    assert reponse.status_code == 200
    contenu = reponse.json()
    assert contenu["statut"] == "ok"
    assert contenu["modele_charge"] is True


def test_la_prediction_renvoie_le_format_attendu(client):
    """La reponse doit contenir tous les champs annonces."""
    reponse = client.post("/prediction", json={"tweet": "this flight was great"})
    assert reponse.status_code == 200

    contenu = reponse.json()
    for champ in ("tweet", "tweet_nettoye", "sentiment",
                  "probabilite_positif", "seuil_decision"):
        assert champ in contenu

    assert contenu["sentiment"] in ("positif", "negatif")
    assert 0.0 <= contenu["probabilite_positif"] <= 1.0


@pytest.mark.parametrize(
    "tweet, sentiment_attendu",
    [
        ("this flight was absolutely terrible, worst experience", "negatif"),
        ("amazing crew, best flight ever, thank you so much", "positif"),
        ("i do not like this airline at all", "negatif"),
        ("the service was excellent and the staff very friendly", "positif"),
    ],
)
def test_le_modele_predit_le_bon_sentiment(client, tweet, sentiment_attendu):
    """
    Verifie que le modele deploye fait bien son travail.

    Ces quatre cas sont volontairement tranches. Le but n'est pas de mesurer
    la performance du modele, ce que font les notebooks, mais de detecter un
    accident : un mauvais fichier deploye, un vocabulaire desynchronise, une
    sortie inversee. Sur des phrases aussi nettes, un modele sain ne se trompe
    pas.
    """
    reponse = client.post("/prediction", json={"tweet": tweet})
    assert reponse.status_code == 200
    assert reponse.json()["sentiment"] == sentiment_attendu


def test_la_negation_inverse_bien_la_prediction(client):
    """
    Le coeur de ce qui distingue ce modele du modele classique.

    Deux phrases qui ne different que par une negation doivent donner deux
    reponses opposees. Si ce test echoue, c'est que le modele deploye n'est
    pas le bon.
    """
    positif = client.post("/prediction", json={"tweet": "this flight was good"}).json()
    negatif = client.post("/prediction", json={"tweet": "this flight was not good"}).json()

    assert positif["probabilite_positif"] > negatif["probabilite_positif"], (
        "La negation ne fait pas baisser le score : le modele deploye est "
        "probablement le mauvais."
    )


# ===========================================================================
# GROUPE 4 : les entrées invalides sont refusées proprement
# ===========================================================================
# En production, l'API recoit ce qu'on veut bien lui envoyer. Elle doit
# repondre une erreur claire, jamais tomber.

def test_un_tweet_vide_est_refuse(client):
    """Le code 422 signifie que la requete est bien formee mais inexploitable."""
    reponse = client.post("/prediction", json={"tweet": ""})
    assert reponse.status_code == 422


def test_un_tweet_sans_mot_exploitable_est_refuse(client):
    """Apres nettoyage il ne reste rien : on le dit au lieu de deviner."""
    reponse = client.post("/prediction", json={"tweet": "@user1 @user2 http://x.com"})
    assert reponse.status_code == 422
    assert "nettoye" in reponse.json()["detail"].lower()


def test_une_requete_sans_le_champ_tweet_est_refusee(client):
    """FastAPI refuse tout seul une requete qui ne suit pas le format annonce."""
    reponse = client.post("/prediction", json={"texte": "mauvais nom de champ"})
    assert reponse.status_code == 422


def test_un_tweet_trop_long_est_refuse(client):
    """La limite de 1000 caracteres evite qu'on sature le serveur."""
    reponse = client.post("/prediction", json={"tweet": "a" * 5000})
    assert reponse.status_code == 422


def test_les_caracteres_speciaux_ne_font_pas_tomber_l_api(client):
    """
    Emojis, alphabets non latins, ponctuation seule : l'API doit tenir.

    Elle repond soit une prediction, soit une erreur 422 si le tweet devient
    vide apres nettoyage. Jamais une erreur 500, qui signifierait que le
    serveur a plante.
    """
    entrees_difficiles = [
        "日本語のツイート",
        "flight 😡😡😡 delayed",
        "!!!???...",
        "a",
        "   spaces   everywhere   ",
    ]
    for tweet in entrees_difficiles:
        reponse = client.post("/prediction", json={"tweet": tweet})
        assert reponse.status_code in (200, 422), (
            f"Reponse inattendue {reponse.status_code} pour {tweet!r}"
        )
