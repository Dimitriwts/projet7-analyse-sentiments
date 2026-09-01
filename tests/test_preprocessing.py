"""
Tests unitaires du nettoyage des tweets.

Ces tests sont executes automatiquement par le pipeline de deploiement continu
(GitHub Actions) AVANT tout deploiement : si l'un d'eux echoue, l'API n'est
pas mise en ligne.

C'est le filet de securite le plus important du projet. Le nettoyage du texte
est partage entre l'entrainement et la production ; une modification en
apparence anodine (retirer un caractere de plus, ajouter un mot vide) suffit
a degrader silencieusement les predictions en ligne, sans qu'aucune erreur ne
soit levee nulle part. Ces tests figent le comportement attendu.

Pour les lancer a la main :
    .venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import pytest

from src.preprocessing import (
    MOTS_A_PRESERVER,
    MOTS_VIDES,
    nettoyer_tweet,
    retirer_mots_vides,
)


# ===========================================================================
# GROUPE 1 : le nettoyage fait bien ce qu'on attend de lui
# ===========================================================================

def test_passage_en_minuscules():
    """Le modele ne doit pas voir "GREAT" et "great" comme deux mots differents."""
    assert nettoyer_tweet("GREAT Flight") == "great flight"


def test_suppression_des_adresses_web():
    """Un lien n'est ni positif ni negatif : il doit disparaitre."""
    resultat = nettoyer_tweet("nice trip http://t.co/abc123")
    assert resultat == "nice trip"
    assert "http" not in resultat


def test_suppression_des_mentions():
    """Un pseudo est unique : le garder gonflerait le vocabulaire pour rien."""
    resultat = nettoyer_tweet("@AirParadis thanks for the upgrade")
    assert resultat == "thanks for the upgrade"
    assert "@" not in resultat


def test_le_hashtag_perd_son_croisillon_mais_garde_son_mot():
    """Le mot d'un hashtag porte souvent un sentiment tres fort : on le garde."""
    assert nettoyer_tweet("flight was #terrible") == "flight was terrible"


def test_decodage_des_entites_html():
    """Le jeu de donnees contient "&amp;" au lieu de "&"."""
    assert nettoyer_tweet("crew &amp; food") == "crew and food"


def test_reduction_des_lettres_repetees():
    """"soooooo" et "sooo" doivent devenir le meme mot."""
    assert nettoyer_tweet("sooooo good") == "soo good"
    assert nettoyer_tweet("sooo good") == "soo good"


def test_suppression_de_la_ponctuation_et_des_chiffres():
    """Ni la ponctuation ni les chiffres ne portent le sentiment ici."""
    assert nettoyer_tweet("delayed 3 hours!!! ???") == "delayed hours"


def test_les_espaces_sont_normalises():
    """Un seul espace entre les mots, aucun en debut ni en fin."""
    assert nettoyer_tweet("   too    many   spaces   ") == "too many spaces"


# ===========================================================================
# GROUPE 2 : les cas limites ne font pas planter l'API
# ===========================================================================
# En production, l'API recoit du JSON envoye par un client exterieur. On ne
# maitrise pas ce qui arrive : ces tests garantissent qu'aucune entree bizarre
# ne provoque une erreur 500.

@pytest.mark.parametrize(
    "entree_inattendue",
    [
        "",             # chaine vide
        "   ",          # que des espaces
        None,           # valeur nulle envoyee en JSON
        12345,          # un nombre au lieu d'un texte
        [],             # une liste
        "@user http://x.com",   # il ne reste rien apres nettoyage
        "!!!???...",            # que de la ponctuation
        "日本語のツイート",        # alphabet non latin
    ],
)
def test_les_entrees_inattendues_renvoient_une_chaine_vide(entree_inattendue):
    """Aucune entree ne doit lever d'exception : au pire on renvoie ""."""
    resultat = nettoyer_tweet(entree_inattendue)
    assert isinstance(resultat, str)
    assert resultat == ""


def test_le_nettoyage_est_idempotent():
    """
    Nettoyer un texte deja nettoye ne doit rien changer.

    Cette propriete evite une classe entiere de bugs : si par megarde le
    nettoyage etait applique deux fois (une fois dans l'interface de test, une
    fois dans l'API), le resultat resterait identique.
    """
    texte = "@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz"
    une_fois = nettoyer_tweet(texte)
    deux_fois = nettoyer_tweet(une_fois)
    assert une_fois == deux_fois


# ===========================================================================
# GROUPE 3 : les negations survivent - LE test critique du projet
# ===========================================================================
# C'est ici que se joue la qualite du modele. Les listes de mots vides
# standard suppriment "not", "no", "never". Or supprimer la negation d'une
# phrase en inverse le sens : le modele apprend alors le contraire de ce
# qu'il faudrait.

def test_aucune_negation_dans_la_liste_des_mots_vides():
    """
    Verifie qu'aucun mot porteur de sentiment n'a atterri dans MOTS_VIDES.

    Ce test protege contre une modification future de la liste : si quelqu'un
    ajoute un jour "not" ou "never" aux mots vides, il casse ici et non en
    production trois semaines plus tard.
    """
    intersection = MOTS_VIDES & MOTS_A_PRESERVER
    assert intersection == frozenset(), (
        "Ces mots porteurs de sentiment sont traites comme des mots vides "
        f"et seraient supprimes : {sorted(intersection)}"
    )


@pytest.mark.parametrize("negation", ["not", "no", "never", "nothing", "cannot"])
def test_les_negations_survivent_au_retrait_des_mots_vides(negation):
    """"the flight was not good" ne doit pas devenir "flight good"."""
    texte = f"the flight was {negation} good at all"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    assert negation in resultat.split()


@pytest.mark.parametrize("intensificateur", ["very", "so", "too", "really"])
def test_les_intensificateurs_survivent(intensificateur):
    """"very bad" est plus negatif que "bad" : l'intensite compte."""
    texte = f"this is {intensificateur} bad"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    assert intensificateur in resultat.split()


def test_exemple_complet_de_negation():
    """Cas concret : le sens de la phrase doit rester lisible apres nettoyage."""
    texte = "I do NOT recommend this airline, worst experience ever"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    # La negation et les deux mots negatifs forts sont bien la.
    assert "not" in resultat
    assert "recommend" in resultat
    assert "worst" in resultat


# ===========================================================================
# GROUPE 4 : le retrait des mots vides fait bien son travail
# ===========================================================================

def test_les_mots_vides_sont_bien_retires():
    """Les articles et pronoms neutres doivent disparaitre."""
    resultat = retirer_mots_vides("the flight of the day was on time")
    assert "the" not in resultat.split()
    assert "of" not in resultat.split()
    assert "flight" in resultat.split()


def test_retirer_mots_vides_accepte_une_entree_invalide():
    """Meme robustesse que le nettoyage : jamais d'exception."""
    assert retirer_mots_vides(None) == ""
    assert retirer_mots_vides("") == ""
