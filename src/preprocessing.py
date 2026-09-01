"""
Le nettoyage du texte des tweets.

C'est le fichier le plus important du projet, et voici pourquoi.

Il sert à deux moments :
  1. Pendant l'entraînement, pour nettoyer les 1,6 million de tweets.
  2. En production, dans l'API, pour nettoyer le tweet que l'utilisateur
     vient de taper.

Ces deux nettoyages doivent être exactement identiques. Si l'API nettoie ne
serait-ce qu'un tout petit peu différemment, le modèle reçoit en production un
texte qui ne ressemble pas à ce qu'il a vu à l'entraînement, et ses réponses
se dégradent sans qu'aucune erreur ne s'affiche nulle part. C'est un piège
classique de la mise en production, on l'appelle le "training/serving skew",
littéralement le décalage entre l'entraînement et le service.

Pour être sûr que les deux versions restent identiques, ce fichier est recopié
automatiquement dans le dossier api/ par le pipeline de déploiement, et un test
vérifie que les deux copies sont bien les mêmes.

Une contrainte technique importante : ce fichier n'importe que "re", le module
d'expressions régulières livré avec Python. Aucune bibliothèque extérieure.
La raison est que l'API tourne sur un serveur Azure gratuit limité à 1 Go de
mémoire. Chaque bibliothèque ajoutée (nltk, spacy, pandas) alourdit le
démarrage et rapproche du plafond. Nettoyer du texte ne demande aucune
bibliothèque, autant s'en passer.
"""

import re

# ---------------------------------------------------------------------------
# 1. LES MOTS VIDES
# ---------------------------------------------------------------------------
# Les mots vides (stop words en anglais) sont les mots trop courants pour
# porter du sens à eux seuls : les articles, les pronoms, les auxiliaires.
# Les retirer allège le vocabulaire du modèle classique.
#
# Attention, et c'est le point crucial de tout ce projet : les listes de mots
# vides toutes faites, comme celle de la bibliothèque NLTK, contiennent les
# négations "not", "no", "never", "nor". Or si je retire "not" de la phrase
# "this flight was not good", il reste "flight good", et le sens est
# exactement inversé. Le modèle apprendrait alors le contraire de ce qu'il
# faut.
#
# La liste ci-dessous est donc une liste classique dont j'ai retiré tout ce
# qui porte du sentiment. Je garde volontairement :
#   les négations        : not, no, never, nor, none, cannot, don't...
#   les intensificateurs : very, too, so, really, quite...
#   les mots d'opposition : but, however, although
#
# Ils sont peu nombreux mais ils changent complètement la polarité d'une
# phrase. Les supprimer coûterait plusieurs points de score.
MOTS_VIDES = frozenset(
    {
        # Articles et déterminants
        "a", "an", "the", "this", "that", "these", "those",
        "some", "any", "each", "every", "another", "other", "such",
        # Pronoms
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
        # Auxiliaires et verbes support
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing", "done",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        # Prépositions et conjonctions neutres
        "of", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "and", "or", "as", "if", "because",
        "until", "while", "since", "than", "then", "there", "here", "to",
        # Divers qui ne portent pas de sentiment
        "all", "both", "few", "own", "same",
        "now", "just", "only", "also", "well", "back", "even", "still",
        "get", "got", "go", "going", "gonna",
        "one", "two", "im", "u", "ur", "amp",
    }
)

# Les mots que je garde exprès, alors qu'ils sont très fréquents, parce qu'ils
# portent ou modifient le sentiment. Cette liste sert de documentation et de
# garde-fou : un test vérifie qu'aucun d'eux ne s'est glissé dans MOTS_VIDES.
MOTS_A_PRESERVER = frozenset(
    {
        # Les négations, elles inversent le sens.
        # Je liste les deux orthographes parce que le nettoyage garde
        # l'apostrophe et que sur Twitter les deux formes coexistent
        # massivement ("don't" et "dont").
        "not", "no", "never", "nor", "none", "nothing", "nobody", "cannot",
        "cant", "dont", "doesnt", "didnt", "isnt", "arent",
        "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt",
        "havent", "hasnt", "hadnt",
        "can't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
        "haven't", "hasn't", "hadn't",
        # Les intensificateurs, ils amplifient le sentiment.
        "very", "too", "so", "much", "many", "more", "most", "really",
        "quite", "totally", "absolutely",
        # Les mots qui atténuent ou qui opposent.
        "but", "however", "although", "though", "barely", "hardly", "less",
    }
)


# ---------------------------------------------------------------------------
# 2. LES EXPRESSIONS RÉGULIÈRES
# ---------------------------------------------------------------------------
# Une expression régulière est une façon de décrire un motif de texte à
# chercher. Par exemple "un @ suivi de lettres" pour repérer une mention.
#
# Je les compile une seule fois au chargement du fichier, et pas à chaque
# appel de la fonction. Sur 1,6 million de tweets, ça change le temps de
# calcul de plusieurs minutes.

# Les entités HTML. Le jeu de données a été aspiré du web sans être décodé,
# on y trouve donc "&amp;" à la place de "&", "&quot;" pour un guillemet.
# Je les remets en clair avant tout le reste.
ENTITES_HTML = {
    "&amp;": " and ",
    "&quot;": " ",
    "&lt;": " ",
    "&gt;": " ",
    "&nbsp;": " ",
    "&#39;": "'",
}

# Les adresses web. Un lien n'est ni positif ni négatif, et comme chaque lien
# est unique, les garder ferait exploser la taille du vocabulaire avec des
# milliers de mots vus une seule fois. Je les supprime.
MOTIF_URL = re.compile(r"https?://\S+|www\.\S+")

# Les mentions d'utilisateur, comme @AirParadis. Même raisonnement : chaque
# pseudo est unique.
MOTIF_MENTION = re.compile(r"@\w+")

# Les hashtags. Là je ne supprime que le croisillon et je garde le mot, parce
# que "#terrible" est une information de sentiment très forte.
MOTIF_HASHTAG = re.compile(r"#(\w+)")

# Les lettres répétées trois fois ou plus. Quand quelqu'un écrit "sooooo
# goooood", c'est pour insister. Je ramène à deux répétitions ("soo goood"),
# ce qui regroupe toutes les variantes du même mot tout en gardant la trace
# de l'insistance.
MOTIF_LETTRES_REPETEES = re.compile(r"(.)\1{2,}")

# Tout ce qui n'est ni une lettre, ni une apostrophe, ni un espace : les
# chiffres, la ponctuation, les caractères spéciaux, les emojis.
MOTIF_CARACTERES_NON_ALPHA = re.compile(r"[^a-z'\s]")

# Les espaces en trop, les tabulations, les retours à la ligne.
MOTIF_ESPACES_MULTIPLES = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 3. LA FONCTION DE NETTOYAGE
# ---------------------------------------------------------------------------
def nettoyer_tweet(texte: str) -> str:
    """
    Nettoie le texte brut d'un tweet et renvoie une version normalisée.

    J'applique exactement le même nettoyage aux trois approches (classique,
    réseau sur mesure, BERT). C'est la condition pour que la comparaison de
    leurs scores soit honnête : si chaque approche nettoyait à sa façon, je ne
    saurais plus si l'écart vient du modèle ou du prétraitement.

    Les étapes, dans l'ordre :
      1. Sécuriser l'entrée (valeur manquante, mauvais type).
      2. Passer en minuscules.
      3. Décoder les entités HTML.
      4. Supprimer les adresses web.
      5. Supprimer les mentions @utilisateur.
      6. Enlever le croisillon des hashtags en gardant le mot.
      7. Réduire les lettres répétées.
      8. Supprimer chiffres, ponctuation et caractères spéciaux.
      9. Normaliser les espaces.

    Paramètres
    ----------
    texte : str
        Le texte brut du tweet, soit tel qu'il sort du fichier CSV, soit tel
        qu'il a été tapé par l'utilisateur dans l'interface de test.

    Retourne
    --------
    str
        Le tweet nettoyé, en minuscules, sans ponctuation, mots séparés par un
        seul espace. Peut être une chaîne vide si le tweet ne contenait que des
        liens ou des mentions.

    Exemples
    --------
    >>> nettoyer_tweet("@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz")
    'my flight is soo late angry'

    >>> nettoyer_tweet("Great service &amp; friendly crew :)")
    'great service and friendly crew'

    >>> nettoyer_tweet("")
    ''
    """
    # Étape 1, sécuriser l'entrée.
    # L'API reçoit du JSON envoyé par un client extérieur : on peut très bien
    # y trouver None ou un nombre. Plutôt que de laisser une erreur remonter
    # jusqu'à l'utilisateur, je renvoie une chaîne vide, que la suite sait
    # gérer.
    if not isinstance(texte, str):
        return ""

    # Étape 2, les minuscules.
    # "GREAT", "Great" et "great" doivent être le même mot pour le modèle.
    texte = texte.lower()

    # Étape 3, les entités HTML.
    # À faire tôt : "&amp;" contient un "&" qui serait sinon supprimé à
    # l'étape 8, en laissant traîner un "amp" parasite.
    for entite, remplacement in ENTITES_HTML.items():
        texte = texte.replace(entite, remplacement)

    # Étape 4, les adresses web.
    texte = MOTIF_URL.sub(" ", texte)

    # Étape 5, les mentions.
    texte = MOTIF_MENTION.sub(" ", texte)

    # Étape 6, les hashtags.
    # Le r"\1" veut dire "remets ce que tu as capturé entre parenthèses",
    # c'est-à-dire le mot qui suivait le croisillon.
    texte = MOTIF_HASHTAG.sub(r"\1", texte)

    # Étape 7, les lettres répétées.
    # r"\1\1" : je garde deux exemplaires de la lettre capturée.
    texte = MOTIF_LETTRES_REPETEES.sub(r"\1\1", texte)

    # Étape 8, les caractères qui ne sont pas des lettres.
    # Je garde l'apostrophe, sinon "don't" devient "don" plus "t" et la
    # négation disparaît.
    texte = MOTIF_CARACTERES_NON_ALPHA.sub(" ", texte)

    # Étape 9, les espaces.
    # .strip() enlève les espaces au début et à la fin.
    texte = MOTIF_ESPACES_MULTIPLES.sub(" ", texte).strip()

    return texte


# ---------------------------------------------------------------------------
# 4. LE RETRAIT DES MOTS VIDES, EN OPTION
# ---------------------------------------------------------------------------
def retirer_mots_vides(texte: str) -> str:
    """
    Retire les mots vides d'un texte déjà nettoyé.

    J'ai séparé cette étape de nettoyer_tweet exprès, parce qu'elle n'est pas
    souhaitable pour toutes les approches :

      Modèle classique : le retrait est souvent bénéfique. Ce modèle ne voit
      que des mots isolés, sans leur ordre, donc les mots vides ne sont que
      du bruit qui dilue le vocabulaire.

      Modèle avancé et BERT : le retrait est plutôt nuisible. Ces modèles se
      servent de l'ordre et de la structure de la phrase, et cette structure
      est justement portée par les mots vides. En plus, les vecteurs de mots
      qu'ils utilisent ont été appris sur du texte complet, pas amputé.

    Le notebook 02 compare les deux variantes et enregistre les deux scores
    dans MLflow, pour que la décision se prenne sur une mesure et pas sur une
    intuition.

    Paramètres
    ----------
    texte : str
        Un texte déjà passé par nettoyer_tweet.

    Retourne
    --------
    str
        Le texte sans ses mots vides.

    Exemples
    --------
    >>> retirer_mots_vides("the flight was not good at all")
    'flight not good'
    """
    if not isinstance(texte, str):
        return ""

    mots_conserves = [mot for mot in texte.split() if mot not in MOTS_VIDES]
    return " ".join(mots_conserves)


# ---------------------------------------------------------------------------
# 5. VÉRIFICATION RAPIDE
# ---------------------------------------------------------------------------
# Ce bloc ne s'exécute que si je lance directement "python src/preprocessing.py".
# Il ne s'exécute jamais quand le fichier est importé par un notebook ou par
# l'API. C'est pratique pour vérifier le comportement d'un coup d'oeil.
if __name__ == "__main__":
    exemples = [
        "@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz",
        "Great service &amp; friendly crew :)",
        "I do NOT recommend this airline... worst experience ever",
        "@user1 @user2 http://spam.com",
        "",
    ]

    print("=" * 70)
    print("VERIFICATION DU NETTOYAGE DES TWEETS")
    print("=" * 70)
    for exemple in exemples:
        nettoye = nettoyer_tweet(exemple)
        sans_mots_vides = retirer_mots_vides(nettoye)
        print(f"\nBrut            : {exemple!r}")
        print(f"Nettoye         : {nettoye!r}")
        print(f"Sans mots vides : {sans_mots_vides!r}")
    print("\n" + "=" * 70)
