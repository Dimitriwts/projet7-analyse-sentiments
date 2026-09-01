"""
Les tests du nettoyage des tweets.

Un test unitaire est un petit bout de code qui vérifie automatiquement qu'une
fonction fait bien ce qu'on attend d'elle. On les lance en une commande, et si
l'un d'eux échoue, on sait tout de suite quoi est cassé et où.

Ces tests sont lancés automatiquement par le pipeline de déploiement (GitHub
Actions) avant chaque mise en ligne. Si l'un d'eux échoue, l'API n'est pas
déployée.

C'est le filet de sécurité le plus important du projet. Le nettoyage du texte
est partagé entre l'entraînement et la production, donc une modification qui
a l'air anodine (enlever un caractère de plus, ajouter un mot vide) suffit à
dégrader les réponses en ligne sans qu'aucune erreur ne s'affiche. Ces tests
figent le comportement attendu.

Pour les lancer à la main :
    .venv\\Scripts\\python.exe -m pytest
"""

import pytest

from src.preprocessing import (
    MOTS_A_PRESERVER,
    MOTS_VIDES,
    nettoyer_tweet,
    retirer_mots_vides,
)


# ===========================================================================
# GROUPE 1 : le nettoyage fait bien ce qu'on lui demande
# ===========================================================================

def test_passage_en_minuscules():
    """Le modele ne doit pas voir GREAT et great comme deux mots differents."""
    assert nettoyer_tweet("GREAT Flight") == "great flight"


def test_suppression_des_adresses_web():
    """Un lien n'est ni positif ni negatif, il doit disparaitre."""
    resultat = nettoyer_tweet("nice trip http://t.co/abc123")
    assert resultat == "nice trip"
    assert "http" not in resultat


def test_suppression_des_mentions():
    """Un pseudo est unique, le garder gonflerait le vocabulaire pour rien."""
    resultat = nettoyer_tweet("@AirParadis thanks for the upgrade")
    assert resultat == "thanks for the upgrade"
    assert "@" not in resultat


def test_le_hashtag_perd_son_croisillon_mais_garde_son_mot():
    """Le mot d'un hashtag porte souvent un sentiment fort, on le garde."""
    assert nettoyer_tweet("flight was #terrible") == "flight was terrible"


def test_decodage_des_entites_html():
    """Le jeu de donnees contient &amp; a la place de &."""
    assert nettoyer_tweet("crew &amp; food") == "crew and food"


def test_reduction_des_lettres_repetees():
    """sooooo et sooo doivent devenir le meme mot."""
    assert nettoyer_tweet("sooooo good") == "soo good"
    assert nettoyer_tweet("sooo good") == "soo good"


def test_suppression_de_la_ponctuation_et_des_chiffres():
    """Ni la ponctuation ni les chiffres ne portent le sentiment ici."""
    assert nettoyer_tweet("delayed 3 hours!!! ???") == "delayed hours"


def test_les_espaces_sont_normalises():
    """Un seul espace entre les mots, aucun au debut ni a la fin."""
    assert nettoyer_tweet("   too    many   spaces   ") == "too many spaces"


# ===========================================================================
# GROUPE 2 : les entrees bizarres ne font pas planter l'API
# ===========================================================================
# En production, l'API recoit du JSON envoye par un client exterieur. On ne
# maitrise pas ce qui arrive. Ces tests garantissent qu'aucune entree bizarre
# ne provoque une erreur 500 (le code d'erreur qui veut dire "le serveur a
# planté").

@pytest.mark.parametrize(
    "entree_inattendue",
    [
        "",                      # chaine vide
        "   ",                   # que des espaces
        None,                    # valeur nulle envoyee en JSON
        12345,                   # un nombre au lieu d'un texte
        [],                      # une liste
        "@user http://x.com",    # il ne reste rien apres nettoyage
        "!!!???...",             # que de la ponctuation
        "日本語のツイート",         # un alphabet non latin
    ],
)
def test_les_entrees_inattendues_renvoient_une_chaine_vide(entree_inattendue):
    """Aucune entree ne doit lever d'exception, au pire on renvoie une chaine vide."""
    resultat = nettoyer_tweet(entree_inattendue)
    assert isinstance(resultat, str)
    assert resultat == ""


def test_le_nettoyage_est_idempotent():
    """
    Nettoyer un texte deja nettoye ne doit rien changer.

    Idempotent veut dire "qui donne le meme resultat qu'on l'applique une ou
    plusieurs fois". Cette propriete evite toute une classe de bugs : si par
    megarde le nettoyage etait applique deux fois, une fois dans l'interface
    et une fois dans l'API, le resultat resterait le meme.
    """
    texte = "@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz"
    une_fois = nettoyer_tweet(texte)
    deux_fois = nettoyer_tweet(une_fois)
    assert une_fois == deux_fois


# ===========================================================================
# GROUPE 3 : les negations survivent, le test le plus important du projet
# ===========================================================================
# C'est ici que se joue la qualite du modele. Les listes de mots vides toutes
# faites suppriment not, no, never. Or supprimer la negation d'une phrase en
# inverse le sens, et le modele apprend alors le contraire de ce qu'il faut.

def test_aucune_negation_dans_la_liste_des_mots_vides():
    """
    Verifie qu'aucun mot porteur de sentiment n'est traite comme un mot vide.

    Ce test protege contre une modification future : si quelqu'un ajoute un
    jour not ou never aux mots vides, ca casse ici, et pas en production trois
    semaines plus tard.
    """
    intersection = MOTS_VIDES & MOTS_A_PRESERVER
    assert intersection == frozenset(), (
        "Ces mots porteurs de sentiment sont traites comme des mots vides "
        f"et seraient supprimes : {sorted(intersection)}"
    )


@pytest.mark.parametrize("negation", ["not", "no", "never", "nothing", "cannot"])
def test_les_negations_survivent_au_retrait_des_mots_vides(negation):
    """the flight was not good ne doit pas devenir flight good."""
    texte = f"the flight was {negation} good at all"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    assert negation in resultat.split()


@pytest.mark.parametrize("intensificateur", ["very", "so", "too", "really"])
def test_les_intensificateurs_survivent(intensificateur):
    """very bad est plus negatif que bad, l'intensite compte."""
    texte = f"this is {intensificateur} bad"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    assert intensificateur in resultat.split()


def test_exemple_complet_de_negation():
    """Un cas concret : le sens de la phrase doit rester lisible apres nettoyage."""
    texte = "I do NOT recommend this airline, worst experience ever"
    resultat = retirer_mots_vides(nettoyer_tweet(texte))
    assert "not" in resultat
    assert "recommend" in resultat
    assert "worst" in resultat


# ===========================================================================
# GROUPE 4 : le retrait des mots vides fait bien son travail
# ===========================================================================

def test_les_mots_vides_sont_bien_retires():
    """Les articles et les pronoms neutres doivent disparaitre."""
    resultat = retirer_mots_vides("the flight of the day was on time")
    assert "the" not in resultat.split()
    assert "of" not in resultat.split()
    assert "flight" in resultat.split()


def test_retirer_mots_vides_accepte_une_entree_invalide():
    """Meme robustesse que le nettoyage, jamais d'exception."""
    assert retirer_mots_vides(None) == ""
    assert retirer_mots_vides("") == ""
