"""
L'API de prédiction de sentiment.

Elle reçoit le texte d'un tweet et renvoie le sentiment prédit par le modèle
sur mesure avancé, celui entraîné dans le notebook 03 et exporté par le
notebook 05.

Ce fichier est le seul code qui tourne en production. Il est volontairement
court et sans dépendance lourde, pour deux raisons.

D'abord la contrainte matérielle : le serveur Azure gratuit est limité à 1 Go
de mémoire. TensorFlow pèse à lui seul environ 600 Mo à l'installation, il ne
rentre pas. On utilise donc le modèle converti au format TensorFlow Lite, dont
le moteur d'exécution ne pèse que quelques mégaoctets.

Ensuite la fiabilité : moins il y a de code en production, moins il y a de
choses qui peuvent casser. Tout ce qui pouvait être fait en amont, dans les
notebooks, l'a été.

Pour lancer l'API en local :
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload

La documentation interactive est alors sur http://127.0.0.1:8000/docs
"""

import json
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ce fichier importe `preprocessing` sans prefixe, parce qu'en production les
# deux fichiers sont cote a cote a la racine du serveur : le pipeline de
# deploiement n'envoie que le contenu du dossier api/, pas le projet entier.
#
# Mais en developpement on lance l'API depuis la racine du projet, avec
# `uvicorn api.main:app`, et Python cherche alors les modules a partir de la
# racine. Sans la ligne ci-dessous, l'import echouerait localement tout en
# marchant en production, ce qui est le pire des cas : on ne s'en apercevrait
# qu'une fois deploye.
#
# On ajoute donc le dossier de ce fichier au chemin de recherche de Python.
# Ca ne change rien en production, ou il y est deja, et ca fait marcher les
# deux facons de lancer l'API.
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import nettoyer_tweet  # noqa: E402

# ---------------------------------------------------------------------------
# 1. LES FICHIERS DU MODÈLE
# ---------------------------------------------------------------------------
# Les trois fichiers produits par le notebook 05. Ils sont versionnés avec le
# code, contrairement aux données : le pipeline de déploiement envoie le
# dossier tel quel sur le serveur, donc tout doit s'y trouver.
DOSSIER_ARTEFACTS = Path(__file__).parent / "artefacts"

FICHIER_MODELE = DOSSIER_ARTEFACTS / "modele.tflite"
FICHIER_VOCABULAIRE = DOSSIER_ARTEFACTS / "vocabulaire.json"
FICHIER_METADONNEES = DOSSIER_ARTEFACTS / "metadonnees.json"


# ---------------------------------------------------------------------------
# 2. CE QUI EST CHARGÉ UNE SEULE FOIS, AU DÉMARRAGE
# ---------------------------------------------------------------------------
# Charger le modèle prend une fraction de seconde, mais le faire à chaque
# requête serait un gâchis considérable. On le charge donc une fois au
# démarrage du serveur et on garde le résultat en mémoire.
#
# Le dictionnaire ci-dessous sert de rangement pour ces objets partagés. On
# évite ainsi les variables globales éparpillées dans le fichier.
modele = {
    "interpreteur": None,
    "vocabulaire": None,
    "metadonnees": None,
    "index_entree": None,
    "index_sortie": None,
    "type_entree": None,
}

# Un interpréteur TensorFlow Lite ne supporte pas d'être utilisé par plusieurs
# requêtes en même temps : il écrit ses résultats dans des cases mémoire qui
# lui sont propres, et deux requêtes simultanées se marcheraient dessus.
#
# Ce verrou garantit qu'une seule prédiction se fait à la fois. Comme une
# prédiction prend moins d'une milliseconde, l'attente est imperceptible.
verrou_prediction = threading.Lock()


def charger_le_modele():
    """
    Charge le modèle, le vocabulaire et les métadonnées depuis le disque.

    Appelée une seule fois, au démarrage du serveur.

    Lève
    ----
    FileNotFoundError
        Si un des trois fichiers manque, avec un message explicite. Mieux vaut
        que le serveur refuse de démarrer plutôt qu'il accepte des requêtes
        auxquelles il ne saura pas répondre.
    """
    for fichier in (FICHIER_MODELE, FICHIER_VOCABULAIRE, FICHIER_METADONNEES):
        if not fichier.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {fichier}\n"
                "Les artefacts du modele sont produits par le notebook "
                "05_comparaison_et_export.ipynb."
            )

    interpreteur = Interpreter(model_path=str(FICHIER_MODELE))
    interpreteur.allocate_tensors()

    details_entree = interpreteur.get_input_details()[0]
    details_sortie = interpreteur.get_output_details()[0]

    modele["interpreteur"] = interpreteur
    modele["index_entree"] = details_entree["index"]
    modele["index_sortie"] = details_sortie["index"]
    modele["type_entree"] = details_entree["dtype"]
    modele["vocabulaire"] = json.loads(
        FICHIER_VOCABULAIRE.read_text(encoding="utf-8")
    )
    modele["metadonnees"] = json.loads(
        FICHIER_METADONNEES.read_text(encoding="utf-8")
    )


@asynccontextmanager
async def cycle_de_vie(application: FastAPI):
    """
    Ce qui se passe au démarrage et à l'arrêt du serveur.

    FastAPI appelle cette fonction une fois au lancement, exécute tout ce qui
    précède le `yield`, sert les requêtes, puis exécute ce qui suit à l'arrêt.
    """
    charger_le_modele()
    print(f"Modele charge : {modele['metadonnees']['nom_modele']}")
    print(f"Vocabulaire   : {len(modele['vocabulaire'])} mots")
    yield
    print("Arret du serveur.")


app = FastAPI(
    title="API de prediction de sentiment, Air Paradis",
    description=(
        "Predit si un tweet exprime un sentiment negatif ou positif. "
        "Modele sur mesure avance : embeddings GloVe Twitter et reseau de "
        "neurones recurrent bidirectionnel, converti au format TensorFlow Lite."
    ),
    version="1.0.0",
    lifespan=cycle_de_vie,
)


# ---------------------------------------------------------------------------
# 3. LE FORMAT DES ÉCHANGES
# ---------------------------------------------------------------------------
# Ces classes décrivent ce que l'API accepte en entrée et ce qu'elle renvoie.
# FastAPI s'en sert pour trois choses à la fois : valider automatiquement les
# requêtes, générer la documentation interactive, et refuser proprement une
# requête mal formée avec un message clair plutôt qu'une erreur 500.
class DemandeDePrediction(BaseModel):
    """Le tweet à analyser."""

    tweet: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Le texte brut du tweet, tel qu'il a ete publie.",
        examples=["@AirParadis my flight was delayed again, terrible service"],
    )


class ReponseDePrediction(BaseModel):
    """Le résultat de l'analyse."""

    tweet: str = Field(description="Le tweet recu, tel quel.")
    tweet_nettoye: str = Field(description="Le tweet apres nettoyage.")
    sentiment: str = Field(description="negatif ou positif.")
    probabilite_positif: float = Field(
        description="Probabilite que le tweet soit positif, entre 0 et 1."
    )
    seuil_decision: float = Field(
        description="Seuil au-dessus duquel le tweet est classe positif."
    )


# ---------------------------------------------------------------------------
# 4. LA PRÉDICTION
# ---------------------------------------------------------------------------
def texte_vers_sequence(texte_nettoye: str) -> list[int]:
    """
    Traduit un texte nettoyé en une suite de nombres de longueur fixe.

    C'est le maillon le plus délicat de toute la chaîne, alors je détaille.

    À l'entraînement, cette traduction était faite par le tokeniseur de Keras.
    Ici elle est refaite en trois lignes, parce que Keras est bien trop lourd
    pour le serveur. Les deux doivent donner exactement le même résultat,
    sinon le modèle reçoit en production des nombres différents de ceux qu'il
    a appris à interpréter, et se dégrade sans qu'aucune erreur ne s'affiche.

    Cette équivalence a été vérifiée nombre par nombre dans le notebook 05,
    sur 128 000 valeurs, sans aucune différence.

    Les trois étapes :
      1. Chaque mot devient son numéro. Un mot absent du vocabulaire reçoit le
         numéro réservé aux mots inconnus.
      2. On coupe si le tweet dépasse la longueur attendue par le modèle.
      3. On complète par des zéros s'il est plus court.
    """
    vocabulaire = modele["vocabulaire"]
    metadonnees = modele["metadonnees"]
    longueur_attendue = metadonnees["longueur_max_sequence"]
    index_inconnu = metadonnees["index_mot_inconnu"]

    numeros = [vocabulaire.get(mot, index_inconnu) for mot in texte_nettoye.split()]
    numeros = numeros[:longueur_attendue]
    numeros += [0] * (longueur_attendue - len(numeros))

    return numeros


def predire(texte_brut: str) -> dict:
    """
    Prédit le sentiment d'un tweet, du texte brut au résultat.

    Trois étapes : nettoyer, traduire en nombres, faire prédire le modèle.
    """
    texte_nettoye = nettoyer_tweet(texte_brut)

    # Un tweet qui ne contenait que des liens ou des mentions devient vide
    # après nettoyage. Le modèle n'a alors rien à se mettre sous la dent, et
    # il vaut mieux le dire clairement que de renvoyer une prédiction faite
    # sur du vide.
    if not texte_nettoye:
        raise HTTPException(
            status_code=422,
            detail=(
                "Le tweet ne contient aucun mot exploitable une fois nettoye. "
                "Les liens, les mentions et la ponctuation sont retires avant "
                "l'analyse."
            ),
        )

    sequence = np.array([texte_vers_sequence(texte_nettoye)], dtype=modele["type_entree"])

    with verrou_prediction:
        interpreteur = modele["interpreteur"]
        interpreteur.set_tensor(modele["index_entree"], sequence)
        interpreteur.invoke()
        probabilite = float(interpreteur.get_tensor(modele["index_sortie"]).ravel()[0])

    seuil = modele["metadonnees"]["seuil_decision"]

    return {
        "tweet": texte_brut,
        "tweet_nettoye": texte_nettoye,
        "sentiment": "positif" if probabilite >= seuil else "negatif",
        "probabilite_positif": round(probabilite, 4),
        "seuil_decision": seuil,
    }


# ---------------------------------------------------------------------------
# 5. LES ROUTES
# ---------------------------------------------------------------------------
@app.get("/", tags=["Informations"])
def accueil():
    """
    Page d'accueil : à quoi sert cette API et quel modèle elle sert.

    Utile pour vérifier d'un coup d'oeil, depuis un navigateur, que le bon
    modèle est bien déployé.
    """
    metadonnees = modele["metadonnees"]
    return {
        "message": "API de prediction de sentiment pour Air Paradis",
        "modele": metadonnees["nom_modele"],
        "architecture": metadonnees["architecture"],
        "date_export_du_modele": metadonnees["date_export"],
        "documentation": "/docs",
        "route_de_prediction": "POST /prediction",
    }


@app.get("/sante", tags=["Informations"])
def sante():
    """
    Vérifie que le serveur est vivant et que le modèle est bien chargé.

    Cette route existe pour Azure, qui l'interroge régulièrement. Si elle ne
    répond pas, la plateforme considère que l'application est en panne et la
    redémarre automatiquement.

    Elle ne fait aucun calcul lourd, pour rester instantanée.
    """
    modele_pret = modele["interpreteur"] is not None
    return {
        "statut": "ok" if modele_pret else "modele non charge",
        "modele_charge": modele_pret,
        "horodatage": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/prediction", response_model=ReponseDePrediction, tags=["Prediction"])
def prediction(demande: DemandeDePrediction):
    """
    Prédit le sentiment d'un tweet.

    Envoyez le texte brut du tweet, l'API se charge du nettoyage.

    Exemple de requête :
        {"tweet": "@AirParadis my flight was delayed again, terrible service"}

    Exemple de réponse :
        {
          "tweet": "@AirParadis my flight was delayed again, terrible service",
          "tweet_nettoye": "my flight was delayed again terrible service",
          "sentiment": "negatif",
          "probabilite_positif": 0.0412,
          "seuil_decision": 0.5
        }
    """
    return predire(demande.tweet)
