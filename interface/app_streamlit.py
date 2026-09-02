"""
L'interface de test de l'API.

C'est le livrable 4 du projet. La consigne demande une interface exécutée en
local qui permet de saisir un tweet, affiche la prédiction, demande à
l'utilisateur si elle est pertinente, et envoie une trace à Application
Insights quand elle ne l'est pas.

Un point important sur ce qu'elle teste. Elle n'utilise pas le modèle en
local : elle appelle l'API déployée sur Azure, par le réseau, exactement comme
le ferait un vrai client. C'est donc bien la chaîne de production qu'on
éprouve, avec son démarrage à froid et sa latence, et pas une copie de
laboratoire.

À quoi sert la validation par l'utilisateur, au-delà de la consigne. Un modèle
mis en production se dégrade avec le temps : le langage évolue, les sujets
changent, et les tweets de 2026 ne ressemblent plus à ceux de 2009 sur
lesquels il a été entraîné. On appelle ça la dérive. Le problème est qu'on ne
la voit pas venir : le modèle continue de répondre avec assurance, simplement
il se trompe plus souvent. Demander à l'utilisateur de signaler les erreurs
est le moyen le plus direct de la détecter, et c'est ce que ces traces
permettent.

Pour lancer l'interface :
    .venv\\Scripts\\streamlit.exe run interface/app_streamlit.py
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. LES RÉGLAGES
# ---------------------------------------------------------------------------
# Les réglages sont lus dans le fichier .env, à la racine du projet, qui n'est
# pas versionné. La chaîne de connexion Application Insights donne le droit
# d'écrire dans un service Azure : elle n'a rien à faire dans le dépôt.
RACINE_PROJET = Path(__file__).resolve().parent.parent
load_dotenv(RACINE_PROJET / ".env")

URL_API = os.getenv("URL_API", "http://127.0.0.1:8000").rstrip("/")
CHAINE_APPLICATION_INSIGHTS = os.getenv(
    "APPLICATIONINSIGHTS_CONNECTION_STRING", ""
).strip()

# Le message qui identifie une trace de prédiction invalidée. C'est sur lui
# que portera la règle d'alerte Azure, donc il doit être stable et unique.
MESSAGE_TRACE = "PREDICTION_INVALIDEE"

# L'API tourne sur un plan gratuit sans option "toujours allumé" : elle
# s'endort après vingt minutes et met une trentaine de secondes à se réveiller.
# Le délai d'attente doit en tenir compte.
DELAI_ATTENTE_SECONDES = 90


# ---------------------------------------------------------------------------
# 2. LA CONNEXION À APPLICATION INSIGHTS
# ---------------------------------------------------------------------------
@st.cache_resource
def preparer_supervision():
    """
    Prépare l'envoi des traces vers Application Insights.

    Le décorateur @st.cache_resource est indispensable ici. Streamlit réexécute
    tout le script à chaque clic de l'utilisateur ; sans ce cache, on
    reconfigurerait la connexion des dizaines de fois et on enverrait chaque
    trace en plusieurs exemplaires.

    Renvoie None si la chaîne de connexion n'est pas renseignée, auquel cas
    l'interface fonctionne quand même, sans envoyer de trace. Mieux vaut une
    interface utilisable qu'une interface qui refuse de démarrer.
    """
    if not CHAINE_APPLICATION_INSIGHTS:
        return None

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(
            connection_string=CHAINE_APPLICATION_INSIGHTS,
            logger_name="air_paradis",
            # On ne veut pas instrumenter automatiquement les requêtes HTTP de
            # Streamlit, qui noieraient nos traces sous du bruit. Seules nos
            # traces explicites nous intéressent.
            disable_offline_storage=False,
        )
        journal = logging.getLogger("air_paradis")
        journal.setLevel(logging.INFO)
        return journal
    except Exception as erreur:  # noqa: BLE001
        st.sidebar.error(f"Supervision indisponible : {erreur}")
        return None


def envoyer_trace(tweet, tweet_nettoye, sentiment_predit, probabilite):
    """
    Envoie à Application Insights la trace d'une prédiction jugée fausse.

    Ce qu'on met dans la trace compte autant que le fait de l'envoyer. On y
    joint le tweet, sa version nettoyée, ce que le modèle a répondu et avec
    quelle assurance. Sans ces informations, on saurait qu'il y a eu des
    erreurs mais on ne pourrait rien en faire.

    Avec elles, on peut répondre à des questions utiles : le modèle se
    trompe-t-il surtout sur les tweets courts, sur ceux qui contiennent de
    l'ironie, ou sur un vocabulaire qu'il ne connaissait pas ? Et se
    trompe-t-il en hésitant, avec une probabilité proche de 0,5, ou avec
    assurance, ce qui est bien plus inquiétant ?
    """
    journal = preparer_supervision()
    if journal is None:
        return False

    journal.warning(
        MESSAGE_TRACE,
        extra={
            "tweet": tweet[:500],
            "tweet_nettoye": tweet_nettoye,
            "sentiment_predit": sentiment_predit,
            "probabilite_positif": probabilite,
            # La distance au seuil de décision : 0 veut dire que le modèle
            # hésitait, 0,5 qu'il était certain de lui.
            "assurance": abs(probabilite - 0.5),
            "horodatage": datetime.now().isoformat(),
        },
    )

    # Les traces sont normalement envoyées par paquets, toutes les quelques
    # secondes. On force l'envoi immédiat pour que l'utilisateur voie sa trace
    # arriver dans Azure sans attendre, et pour que l'alerte se déclenche au
    # bon moment pendant une démonstration.
    try:
        from opentelemetry import _logs

        _logs.get_logger_provider().force_flush()
    except Exception:  # noqa: BLE001
        pass

    return True


# ---------------------------------------------------------------------------
# 3. L'APPEL À L'API
# ---------------------------------------------------------------------------
def demander_une_prediction(tweet):
    """
    Interroge l'API déployée et renvoie sa réponse.

    Renvoie un couple (réponse, message d'erreur). L'un des deux vaut toujours
    None. Cette façon de faire évite les exceptions qui remonteraient jusqu'à
    l'utilisateur sous forme de page rouge illisible.
    """
    try:
        reponse = requests.post(
            f"{URL_API}/prediction",
            json={"tweet": tweet},
            timeout=DELAI_ATTENTE_SECONDES,
        )
    except requests.exceptions.Timeout:
        return None, (
            f"L'API n'a pas repondu en {DELAI_ATTENTE_SECONDES} secondes. "
            "Elle etait peut-etre endormie, reessayez."
        )
    except requests.exceptions.RequestException as erreur:
        return None, f"Impossible de joindre l'API : {erreur}"

    if reponse.status_code == 200:
        return reponse.json(), None

    if reponse.status_code == 422:
        detail = reponse.json().get("detail", "")
        if isinstance(detail, list):  # erreur de validation de FastAPI
            detail = "; ".join(d.get("msg", "") for d in detail)
        return None, f"Tweet refuse par l'API : {detail}"

    return None, f"L'API a repondu {reponse.status_code} : {reponse.text[:200]}"


# ---------------------------------------------------------------------------
# 4. L'INTERFACE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Air Paradis, analyse de sentiment", page_icon="✈️")

st.title("Air Paradis, analyse du sentiment d'un tweet")
st.caption(
    "Interface de test de l'API deployee sur Azure. "
    "Elle interroge la vraie API par le reseau, pas une copie locale."
)

# --- La colonne latérale : l'état du système -------------------------------
with st.sidebar:
    st.header("Etat du systeme")
    st.write("**API interrogee**")
    st.code(URL_API, language=None)

    if st.button("Verifier que l'API repond"):
        with st.spinner("Interrogation..."):
            try:
                debut = datetime.now()
                sante = requests.get(f"{URL_API}/sante", timeout=DELAI_ATTENTE_SECONDES)
                duree = (datetime.now() - debut).total_seconds()
                if sante.status_code == 200 and sante.json().get("modele_charge"):
                    st.success(f"L'API repond, modele charge ({duree:.1f}s)")
                else:
                    st.warning(f"Reponse inattendue : {sante.status_code}")
            except Exception as erreur:  # noqa: BLE001
                st.error(f"Injoignable : {erreur}")

    st.divider()
    st.write("**Supervision**")
    if CHAINE_APPLICATION_INSIGHTS:
        st.success("Application Insights configure")
    else:
        st.warning(
            "Application Insights non configure. "
            "Renseignez APPLICATIONINSIGHTS_CONNECTION_STRING dans le fichier "
            ".env a la racine du projet."
        )

# --- La mémoire de la session ----------------------------------------------
# Streamlit réexécute tout le script à chaque interaction. st.session_state est
# le seul endroit où l'on peut conserver quelque chose d'un clic à l'autre.
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "historique" not in st.session_state:
    st.session_state.historique = []

# --- La saisie --------------------------------------------------------------
tweet_saisi = st.text_area(
    "Le tweet a analyser",
    height=100,
    placeholder="@AirParadis my flight was delayed again, terrible service",
)

if st.button("Analyser", type="primary", disabled=not tweet_saisi.strip()):
    with st.spinner(
        "Analyse en cours. Le premier appel peut prendre une trentaine de "
        "secondes, le temps que l'API se reveille."
    ):
        resultat, erreur = demander_une_prediction(tweet_saisi)

    if erreur:
        st.error(erreur)
        st.session_state.prediction = None
    else:
        st.session_state.prediction = resultat

# --- Le résultat ------------------------------------------------------------
if st.session_state.prediction:
    prediction = st.session_state.prediction
    probabilite = prediction["probabilite_positif"]
    est_negatif = prediction["sentiment"] == "negatif"

    st.divider()

    colonne_verdict, colonne_score = st.columns([2, 1])
    with colonne_verdict:
        if est_negatif:
            st.error(f"### Sentiment NEGATIF")
        else:
            st.success(f"### Sentiment POSITIF")
    with colonne_score:
        st.metric("Probabilite d'etre positif", f"{probabilite:.1%}")

    st.progress(probabilite)
    st.caption(
        f"Seuil de decision : {prediction['seuil_decision']:.0%}. "
        f"En dessous, le tweet est classe negatif."
    )

    with st.expander("Ce que l'API a reellement analyse"):
        st.write("**Tweet nettoye**")
        st.code(prediction["tweet_nettoye"], language=None)
        st.caption(
            "Les liens, les mentions et la ponctuation sont retires avant "
            "l'analyse, exactement comme pendant l'entrainement du modele."
        )

    # --- La validation par l'utilisateur ------------------------------------
    st.divider()
    st.write("**Cette prediction vous parait-elle juste ?**")

    colonne_oui, colonne_non = st.columns(2)

    with colonne_oui:
        if st.button("Oui, la prediction est correcte", use_container_width=True):
            st.session_state.historique.append(
                {"tweet": prediction["tweet"], "valide": True,
                 "sentiment": prediction["sentiment"]}
            )
            st.session_state.prediction = None
            st.success("Merci. Rien n'est envoye pour une prediction correcte.")
            st.rerun()

    with colonne_non:
        if st.button("Non, la prediction est fausse", use_container_width=True):
            envoyee = envoyer_trace(
                tweet=prediction["tweet"],
                tweet_nettoye=prediction["tweet_nettoye"],
                sentiment_predit=prediction["sentiment"],
                probabilite=probabilite,
            )
            st.session_state.historique.append(
                {"tweet": prediction["tweet"], "valide": False,
                 "sentiment": prediction["sentiment"]}
            )
            st.session_state.prediction = None
            if envoyee:
                st.warning(
                    "Signalement enregistre et trace envoyee a Application "
                    "Insights. Trois signalements en cinq minutes declenchent "
                    "une alerte par courriel."
                )
            else:
                st.error(
                    "Signalement enregistre localement, mais la trace n'a pas "
                    "pu etre envoyee : Application Insights n'est pas "
                    "configure."
                )
            st.rerun()

# --- L'historique de la session --------------------------------------------
if st.session_state.historique:
    st.divider()
    signalements = sum(1 for e in st.session_state.historique if not e["valide"])
    st.write(
        f"**Historique de la session** : {len(st.session_state.historique)} "
        f"analyses, dont {signalements} signalee(s) comme fausse(s)"
    )

    for entree in reversed(st.session_state.historique[-10:]):
        marque = "OK" if entree["valide"] else "SIGNALE"
        st.caption(f"[{marque}] {entree['sentiment']} : {entree['tweet'][:90]}")
