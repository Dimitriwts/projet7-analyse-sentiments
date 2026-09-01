"""
Nettoyage du texte des tweets.

=============================================================================
CE FICHIER EST LA SOURCE DE VERITE DU PRETRAITEMENT.
=============================================================================

Il est utilise a DEUX moments distincts :

  1. A l'entrainement, dans les notebooks, pour nettoyer les 1,6 million de
     tweets du jeu de donnees.
  2. En production, dans l'API, pour nettoyer le tweet que l'utilisateur
     vient de saisir.

Ces deux nettoyages doivent etre RIGOUREUSEMENT identiques. Si l'API nettoie
ne serait-ce qu'un peu differemment, le modele recoit en production un texte
qui ne ressemble pas a ce qu'il a vu a l'entrainement, et ses predictions se
degradent sans qu'aucune erreur ne soit levee. C'est un piege classique de la
mise en production, appele "training/serving skew" (litteralement : decalage
entre entrainement et service).

Pour garantir cette identite, le fichier est copie automatiquement dans le
dossier api/ par le pipeline de deploiement, et un test unitaire verifie que
les deux copies sont bien identiques (voir tests/test_preprocessing.py).

-----------------------------------------------------------------------------
CONTRAINTE TECHNIQUE : ce module n'importe QUE la bibliotheque standard
Python (le module `re` des expressions regulieres, et rien d'autre).

Pourquoi ? Parce que l'API tourne sur un serveur Azure gratuit limite a
1 Go de memoire. Chaque dependance ajoutee (nltk, spacy, pandas...) alourdit
le demarrage et rapproche du plafond. Un nettoyage de texte n'a besoin
d'aucune bibliotheque : autant s'en passer.
-----------------------------------------------------------------------------
"""

import re

# ---------------------------------------------------------------------------
# 1. MOTS VIDES ("stop words")
# ---------------------------------------------------------------------------
# Les mots vides sont les mots trop frequents pour porter du sens : articles,
# pronoms, auxiliaires. Les retirer allege le vocabulaire du modele classique.
#
# ATTENTION - POINT CRUCIAL POUR L'ANALYSE DE SENTIMENT :
# Les listes de mots vides standard (celle de NLTK par exemple) contiennent
# les negations : "not", "no", "never", "nor". Or "this flight was not good"
# devient "flight good" une fois ces mots retires : le sens est INVERSE.
#
# La liste ci-dessous est donc une liste standard EXPURGEE de tout ce qui
# porte du sentiment. On conserve volontairement :
#   - les negations      : not, no, never, nor, none, cannot, n't...
#   - les intensificateurs : very, too, so, more, most, really, quite...
#   - les contrastifs    : but, however, although
#
# Ces mots sont peu nombreux mais ils changent radicalement la polarite
# d'une phrase : les supprimer ferait perdre plusieurs points de performance.
MOTS_VIDES = frozenset(
    {
        # Articles et determinants
        "a", "an", "the", "this", "that", "these", "those",
        "some", "any", "each", "every", "another", "other", "such",
        # Pronoms personnels et possessifs
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        # Pronoms relatifs et interrogatifs
        "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
        # Auxiliaires et verbes support
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing", "done",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        # Prepositions et conjonctions neutres
        "of", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "and", "or", "as", "if", "because",
        "until", "while", "since", "than", "then", "there", "here", "to",
        # Divers non porteurs de sentiment
        "all", "both", "few", "own", "same",
        "now", "just", "only", "also", "well", "back", "even", "still",
        "get", "got", "go", "going", "gonna",
        "one", "two", "im", "u", "ur", "amp",
    }
)

# Mots explicitement PRESERVES malgre leur frequence, car ils portent ou
# modifient le sentiment. Cette liste sert de documentation et de garde-fou :
# le test unitaire verifie qu'aucun d'entre eux n'a atterri dans MOTS_VIDES.
MOTS_A_PRESERVER = frozenset(
    {
        # Negations : elles inversent la polarite.
        # Les deux orthographes sont listees car le nettoyage conserve
        # l'apostrophe, et sur Twitter les deux formes coexistent
        # massivement ("don't" et "dont").
        "not", "no", "never", "nor", "none", "nothing", "nobody", "cannot",
        "cant", "dont", "doesnt", "didnt", "isnt", "arent",
        "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt",
        "havent", "hasnt", "hadnt",
        "can't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
        "haven't", "hasn't", "hadn't",
        # Intensificateurs : ils amplifient la polarite
        "very", "too", "so", "much", "many", "more", "most", "really",
        "quite", "totally", "absolutely",
        # Attenuateurs et contrastifs : ils la nuancent ou la retournent
        "but", "however", "although", "though", "barely", "hardly", "less",
    }
)


# ---------------------------------------------------------------------------
# 2. EXPRESSIONS REGULIERES
# ---------------------------------------------------------------------------
# Les expressions regulieres sont compilees UNE SEULE FOIS au chargement du
# module, et non a chaque appel. Sur 1,6 million de tweets, la difference se
# compte en minutes.

# Entites HTML : le jeu de donnees Sentiment140 a ete aspire du web sans etre
# decode, on y trouve donc "&amp;" a la place de "&", "&quot;" pour un
# guillemet, etc. On les remet en clair AVANT tout autre traitement.
ENTITES_HTML = {
    "&amp;": " and ",
    "&quot;": " ",
    "&lt;": " ",
    "&gt;": " ",
    "&nbsp;": " ",
    "&#39;": "'",
}

# Adresses web. Elles n'apportent rien au sentiment (un lien n'est ni positif
# ni negatif) mais polluent enormement le vocabulaire, puisque chaque lien est
# unique. On les supprime.
MOTIF_URL = re.compile(r"https?://\S+|www\.\S+")

# Mentions d'utilisateurs (@AirParadis). Meme raisonnement : chaque pseudo est
# unique, cela creerait des milliers de mots vus une seule fois.
MOTIF_MENTION = re.compile(r"@\w+")

# Hashtags. Ici on ne supprime que le croisillon et on GARDE le mot :
# "#terrible" est une information de sentiment tres forte.
MOTIF_HASHTAG = re.compile(r"#(\w+)")

# Lettres repetees trois fois ou plus. "sooooo goooood" est ecrit ainsi pour
# l'emphase ; on ramene a deux repetitions ("soo goood") pour regrouper toutes
# les variantes d'un meme mot, tout en conservant la trace de l'emphase.
MOTIF_LETTRES_REPETEES = re.compile(r"(.)\1{2,}")

# Tout ce qui n'est ni une lettre, ni une apostrophe, ni un espace :
# chiffres, ponctuation, caracteres speciaux, emojis.
MOTIF_CARACTERES_NON_ALPHA = re.compile(r"[^a-z'\s]")

# Espaces multiples, tabulations, retours a la ligne.
MOTIF_ESPACES_MULTIPLES = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 3. FONCTION PRINCIPALE DE NETTOYAGE
# ---------------------------------------------------------------------------
def nettoyer_tweet(texte: str) -> str:
    """
    Nettoie le texte brut d'un tweet et renvoie une chaine normalisee.

    Ce nettoyage est volontairement le meme pour les trois approches
    (classique, avance sur mesure, BERT) : c'est la condition pour que la
    comparaison de leurs performances soit honnete. Si chaque approche
    nettoyait a sa facon, on ne saurait plus si l'ecart de score vient du
    modele ou du pretraitement.

    Les etapes, dans l'ordre :
      1. Securiser l'entree (valeur manquante, type inattendu).
      2. Passer en minuscules.
      3. Decoder les entites HTML.
      4. Supprimer les adresses web.
      5. Supprimer les mentions @utilisateur.
      6. Retirer le croisillon des hashtags en gardant le mot.
      7. Reduire les lettres repetees.
      8. Supprimer chiffres, ponctuation et caracteres speciaux.
      9. Normaliser les espaces.

    Parametres
    ----------
    texte : str
        Le texte brut du tweet, tel qu'il sort du fichier CSV ou tel qu'il a
        ete saisi par l'utilisateur dans l'interface de test.

    Retourne
    --------
    str
        Le tweet nettoye, en minuscules, sans ponctuation, mots separes par
        un espace unique. Peut etre une chaine vide si le tweet ne contenait
        que des liens ou des mentions.

    Exemples
    --------
    >>> nettoyer_tweet("@AirParadis my flight is soooo LATE!!! #angry http://t.co/xyz")
    'my flight is soo late angry'

    >>> nettoyer_tweet("Great service &amp; friendly crew :)")
    'great service and friendly crew'

    >>> nettoyer_tweet("")
    ''
    """
    # --- Etape 1 : securiser l'entree -------------------------------------
    # L'API recoit du JSON : on peut tres bien y trouver None ou un nombre.
    # Plutot que de laisser une exception remonter jusqu'a l'utilisateur, on
    # renvoie une chaine vide, que le reste du code sait gerer.
    if not isinstance(texte, str):
        return ""

    # --- Etape 2 : minuscules ---------------------------------------------
    # "GREAT", "Great" et "great" doivent etre le meme mot pour le modele.
    texte = texte.lower()

    # --- Etape 3 : entites HTML -------------------------------------------
    # A faire tot : "&amp;" contient un "&" qui serait sinon supprime a
    # l'etape 8, laissant trainer un "amp" parasite.
    for entite, remplacement in ENTITES_HTML.items():
        texte = texte.replace(entite, remplacement)

    # --- Etape 4 : adresses web -------------------------------------------
    texte = MOTIF_URL.sub(" ", texte)

    # --- Etape 5 : mentions d'utilisateurs --------------------------------
    texte = MOTIF_MENTION.sub(" ", texte)

    # --- Etape 6 : hashtags -----------------------------------------------
    # r"\1" veut dire "remets le groupe capture", c'est-a-dire le mot qui
    # suivait le croisillon : "#angry" devient "angry".
    texte = MOTIF_HASHTAG.sub(r"\1", texte)

    # --- Etape 7 : lettres repetees ---------------------------------------
    # r"\1\1" : on garde deux exemplaires de la lettre capturee.
    texte = MOTIF_LETTRES_REPETEES.sub(r"\1\1", texte)

    # --- Etape 8 : caracteres non alphabetiques ---------------------------
    # On conserve l'apostrophe pour ne pas casser "don't" en "don" + "t",
    # ce qui ferait disparaitre la negation.
    texte = MOTIF_CARACTERES_NON_ALPHA.sub(" ", texte)

    # --- Etape 9 : espaces ------------------------------------------------
    # .strip() retire les espaces en debut et fin de chaine.
    texte = MOTIF_ESPACES_MULTIPLES.sub(" ", texte).strip()

    return texte


# ---------------------------------------------------------------------------
# 4. RETRAIT DES MOTS VIDES (optionnel)
# ---------------------------------------------------------------------------
def retirer_mots_vides(texte: str) -> str:
    """
    Retire les mots vides d'un texte deja nettoye.

    Cette etape est SEPAREE de `nettoyer_tweet` a dessein, car elle n'est pas
    souhaitable pour toutes les approches :

      - Modele classique (TF-IDF) : le retrait est generalement benefique.
        Le modele ne voit que des mots isoles, sans ordre ; les mots vides
        n'y sont que du bruit qui dilue le vocabulaire.

      - Modele avance et BERT : le retrait est generalement NUISIBLE. Ces
        modeles exploitent l'ordre et la structure de la phrase. Les mots
        vides portent cette structure grammaticale, et les vecteurs de mots
        pre-entraines ont ete appris sur du texte complet, pas ampute.

    Le notebook du modele classique compare les deux variantes et enregistre
    les deux scores dans MLflow : la decision est prise sur la mesure, pas
    sur une intuition.

    Parametres
    ----------
    texte : str
        Un texte deja passe par `nettoyer_tweet`.

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
# 5. VERIFICATION RAPIDE
# ---------------------------------------------------------------------------
# Ce bloc ne s'execute que si l'on lance directement `python src/preprocessing.py`.
# Il n'est jamais execute quand le module est importe par un notebook ou par
# l'API. C'est un moyen commode de verifier le comportement en un coup d'oeil.
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
