"""
Chargement et préparation du jeu de données.

Je regroupe ici tout ce qui touche aux données, pour que les notebooks n'aient
plus qu'à appeler une fonction. Ça me sert à deux choses.

D'abord, les trois notebooks de modélisation partent exactement du même jeu de
données, découpé de la même façon. Sans ça, la comparaison des trois approches
ne voudrait rien dire.

Ensuite, le nettoyage des 1,6 million de tweets prend une bonne minute. Je ne
le fais qu'une fois : le résultat est sauvegardé sur le disque et les notebooks
suivants le rechargent en une seconde.

Le jeu de données s'appelle Sentiment140. Il contient 1 600 000 tweets en
anglais collectés en 2009, moitié positifs, moitié négatifs.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import nettoyer_tweet


# ---------------------------------------------------------------------------
# 0. UN PETIT OUTIL D'AFFICHAGE
# ---------------------------------------------------------------------------
def formater_nombre(valeur: int) -> str:
    """
    Écrit un nombre avec des espaces tous les trois chiffres.

    1596248 devient "1 596 248", c'est plus lisible dans les sorties.

    Je passe par une fonction plutôt que par un .replace(",", " ") appliqué à
    toute la phrase, parce que ce raccourci remplaçait aussi les virgules du
    texte autour du nombre et cassait la ponctuation.
    """
    return f"{valeur:,}".replace(",", " ")


# ---------------------------------------------------------------------------
# 1. TROUVER LE FICHIER DE DONNÉES
# ---------------------------------------------------------------------------
def trouver_fichier_brut() -> Path:
    """
    Cherche le fichier CSV du jeu de données dans data/raw/.

    Je cherche d'abord le nom officiel. S'il n'y est pas, je prends n'importe
    quel fichier .csv du dossier, parce que le fichier est parfois redistribué
    sous un autre nom.

    Retourne
    --------
    Path
        Le chemin du fichier trouvé.

    Lève
    ----
    FileNotFoundError
        Si aucun CSV n'est présent, avec un message qui explique quoi faire.
    """
    # Cas normal, le fichier porte son nom d'origine.
    if config.FICHIER_DATASET_BRUT.exists():
        return config.FICHIER_DATASET_BRUT

    # Sinon, je prends le premier CSV que je trouve.
    fichiers_csv = sorted(config.DOSSIER_DONNEES_BRUTES.glob("*.csv"))
    if fichiers_csv:
        return fichiers_csv[0]

    # Rien trouvé. J'explique quoi faire au lieu de lever une erreur sèche.
    # Je construis le message ligne par ligne puis je l'assemble. C'est plus
    # lisible qu'une longue concaténation, et ça évite un piège classique de
    # Python : dans 'abc' '=' * 70, la mise bout à bout des deux chaînes se
    # fait AVANT la multiplication, et on obtient 70 fois "abc=".
    lignes = [
        "",
        "=" * 70,
        "JEU DE DONNEES INTROUVABLE",
        "=" * 70,
        f"Aucun fichier .csv dans : {config.DOSSIER_DONNEES_BRUTES}",
        "",
        "A FAIRE :",
        "  1. Telecharger le jeu de donnees Sentiment140 (environ 80 Mo compresse).",
        "  2. Le decompresser.",
        "  3. Deposer le fichier CSV dans le dossier ci-dessus.",
        "",
        "Le fichier attendu s'appelle normalement :",
        "  training.1600000.processed.noemoticon.csv",
        "",
        "Un autre nom marche aussi : je prends le premier .csv du dossier.",
        "=" * 70,
    ]
    raise FileNotFoundError("\n".join(lignes))


# ---------------------------------------------------------------------------
# 2. LIRE LE FICHIER
# ---------------------------------------------------------------------------
def charger_donnees_brutes() -> pd.DataFrame:
    """
    Lit le fichier CSV et renvoie un tableau utilisable.

    Deux particularités du fichier sont gérées ici. Il n'a pas de ligne de
    titre, donc sans header=None pandas prendrait le premier tweet pour les
    noms de colonnes et on perdrait une ligne. Et il n'est pas en UTF-8, donc
    je précise l'encodage latin-1 sinon la lecture plante.

    Retourne
    --------
    pd.DataFrame
        Deux colonnes seulement :
          texte : le texte brut du tweet
          label : 0 pour négatif, 1 pour positif

    Je laisse tomber les quatre autres colonnes du fichier (identifiant, date,
    requête, utilisateur). Le projet demande de prédire le sentiment à partir
    du seul contenu du tweet, et l'API ne recevra qu'un texte : ni la date, ni
    l'auteur. Entraîner le modèle sur des informations qu'il n'aura pas en
    production n'aurait aucun sens.
    """
    chemin = trouver_fichier_brut()
    print(f"Lecture de : {chemin.name}")

    donnees = pd.read_csv(
        chemin,
        encoding=config.ENCODAGE_DATASET,
        header=None,
        names=config.COLONNES_DATASET,
    )

    # Je convertis la polarité d'origine (0 ou 4) en label 0 ou 1.
    # La comparaison donne True ou False, que .astype(int) transforme en 1 ou 0.
    # C'est plus rapide et plus court qu'un dictionnaire de correspondance.
    donnees["label"] = (donnees["polarite"] == config.POLARITE_POSITIVE_BRUTE).astype(int)

    resultat = donnees[["texte", "label"]].copy()

    print(f"  {formater_nombre(len(resultat))} tweets charges")
    print(f"  Negatifs : {formater_nombre((resultat['label'] == 0).sum())}")
    print(f"  Positifs : {formater_nombre((resultat['label'] == 1).sum())}")

    return resultat


# ---------------------------------------------------------------------------
# 3. PRÉPARER LES DONNÉES, UNE SEULE FOIS
# ---------------------------------------------------------------------------
def preparer_donnees(forcer: bool = False) -> pd.DataFrame:
    """
    Nettoie tous les tweets et sauvegarde le résultat sur le disque.

    Ça prend environ une minute sur 1,6 million de tweets, donc je ne le fais
    qu'une fois. Le résultat est écrit au format Parquet et les appels suivants
    se contentent de relire ce fichier.

    Parquet est un format de tableau compressé. Je le préfère au CSV parce
    qu'il est environ cinq fois plus léger, beaucoup plus rapide à relire, et
    qu'il garde le type de chaque colonne sans que j'aie à le redéclarer.

    Paramètres
    ----------
    forcer : bool
        Si True, refait le nettoyage même si le fichier existe déjà. Utile
        après avoir modifié src/preprocessing.py.

    Retourne
    --------
    pd.DataFrame
        Colonnes : texte (brut), label, texte_nettoye.
    """
    fichier_prepare = config.FICHIER_DONNEES_PREPAREES

    # Si le travail a déjà été fait, je ne le refais pas.
    if fichier_prepare.exists() and not forcer:
        print(f"Fichier prepare deja present : {fichier_prepare.name}")
        print("  (utiliser preparer_donnees(forcer=True) pour le regenerer)")
        return pd.read_parquet(fichier_prepare)

    donnees = charger_donnees_brutes()

    print("\nNettoyage des tweets en cours...")
    donnees["texte_nettoye"] = donnees["texte"].apply(nettoyer_tweet)

    # Je retire les tweets devenus vides.
    # Certains ne contenaient que des liens ou des mentions : une fois
    # nettoyés, il ne reste rien. Un texte vide n'apprend rien au modèle et
    # fausserait les scores.
    nombre_avant = len(donnees)
    donnees = donnees[donnees["texte_nettoye"].str.len() > 0].copy()
    print(f"  {formater_nombre(nombre_avant - len(donnees))} tweets vides, retires")

    # Je retire les doublons.
    # Environ 5 % des tweets nettoyés sont des textes strictement identiques :
    # des messages courts et banals comme "thanks" ou "good morning", et
    # surtout du spam de robots, le même message republié jusqu'à 1 500 fois.
    #
    # Pourquoi les enlever : à cause d'une fuite entre l'entraînement et le
    # test. Le découpage entre les deux se fait au hasard, donc si un même
    # texte apparaît 500 fois, il se retrouvera forcément des deux côtés. Le
    # modèle l'aura déjà vu à l'entraînement, il le reconnaît au lieu de le
    # comprendre, et le score du test devient trop beau. Le modèle paraît
    # meilleur qu'il ne le sera en production, où il ne verra que des tweets
    # nouveaux.
    #
    # keep="first" garde la première occurrence de chaque texte. Je perds 5 %
    # des lignes, ce qui n'a aucune importance avec 1,6 million de tweets, et
    # je gagne une évaluation honnête.
    nombre_avant = len(donnees)
    donnees = donnees.drop_duplicates(subset="texte_nettoye", keep="first").copy()
    print(f"  {formater_nombre(nombre_avant - len(donnees))} doublons, retires")

    print(f"  {formater_nombre(len(donnees))} tweets conserves")

    # Sauvegarde. index=False évite d'écrire la colonne d'index de pandas, qui
    # ne sert à rien ici.
    fichier_prepare.parent.mkdir(parents=True, exist_ok=True)
    donnees.to_parquet(fichier_prepare, index=False)
    print(f"\nSauvegarde : {fichier_prepare}")

    return donnees


# ---------------------------------------------------------------------------
# 4. PRENDRE UN ÉCHANTILLON
# ---------------------------------------------------------------------------
def charger_echantillon(
    taille: int | None = None,
    colonnes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Recharge les données préparées, éventuellement réduites à un échantillon.

    Le tirage est stratifié, c'est-à-dire que je prends autant de tweets
    négatifs que de positifs. Un tirage complètement au hasard donnerait des
    proportions un peu différentes à chaque taille, et les scores des trois
    approches seraient un peu moins comparables entre eux.

    Le tirage utilise la graine aléatoire du projet : deux appels avec la même
    taille renvoient exactement les mêmes tweets.

    Paramètres
    ----------
    taille : int ou None
        Nombre total de tweets voulu. None veut dire tout le jeu de données.
    colonnes : list ou None
        Les colonnes à charger. None veut dire toutes.

        Ce paramètre a beaucoup plus d'effet qu'il n'y paraît, alors je
        détaille. Voici ce que j'ai mesuré en chargeant le fichier préparé :

            toutes les colonnes            1068 Mo de mémoire
            texte_nettoye et label seuls     45 Mo de mémoire

        Vingt-quatre fois moins. La colonne "texte" contient le tweet brut,
        qui ne sert qu'à afficher des exemples lisibles dans l'analyse des
        erreurs. Les notebooks de modélisation n'en ont pas besoin.

        Pourquoi ça compte : quand la mémoire vive vient à manquer, Windows
        se met à écrire sur le disque à la place. Un disque est des milliers
        de fois plus lent que la mémoire, et l'entraînement d'un réseau passe
        alors de 1 minute à 15 minutes par passage, sans qu'aucune erreur ne
        s'affiche. Je l'ai vécu sur ce projet.

        Bref : dans un notebook qui charge aussi des embeddings d'un giga-
        octet, précisez ["texte_nettoye", "label"].

    Retourne
    --------
    pd.DataFrame
        Les colonnes demandées, parmi texte, label et texte_nettoye.
    """
    if not config.FICHIER_DONNEES_PREPAREES.exists():
        raise FileNotFoundError(
            "Les donnees n'ont pas encore ete preparees.\n"
            "Executez d'abord le notebook 01_exploration_donnees.ipynb, "
            "ou appelez preparer_donnees()."
        )

    donnees = pd.read_parquet(config.FICHIER_DONNEES_PREPAREES, columns=colonnes)

    if taille is None or taille >= len(donnees):
        print(f"Jeu de donnees complet : {formater_nombre(len(donnees))} tweets")
        return donnees

    # Tirage stratifié : je prends la moitié dans chaque classe.
    # J'utilise une boucle plutôt qu'un groupby : c'est une ligne de plus mais
    # on voit exactement ce qui se passe.
    taille_par_classe = taille // 2
    morceaux = []
    for valeur_label in (0, 1):
        tweets_de_la_classe = donnees[donnees["label"] == valeur_label]
        morceaux.append(
            tweets_de_la_classe.sample(
                n=min(taille_par_classe, len(tweets_de_la_classe)),
                random_state=config.GRAINE_ALEATOIRE,
            )
        )

    echantillon = (
        pd.concat(morceaux)
        # Je remélange, sinon tous les négatifs seraient devant tous les
        # positifs, ce qui perturberait l'entraînement par petits paquets.
        .sample(frac=1.0, random_state=config.GRAINE_ALEATOIRE)
        .reset_index(drop=True)
    )

    print(f"Echantillon stratifie : {formater_nombre(len(echantillon))} tweets")
    print(f"  Negatifs : {formater_nombre((echantillon['label'] == 0).sum())}")
    print(f"  Positifs : {formater_nombre((echantillon['label'] == 1).sum())}")

    return echantillon


# ---------------------------------------------------------------------------
# 5. DÉCOUPER EN ENTRAÎNEMENT ET TEST
# ---------------------------------------------------------------------------
def separer_train_test(donnees: pd.DataFrame, colonne_texte: str = "texte_nettoye"):
    """
    Sépare le jeu de données en un jeu d'entraînement et un jeu de test.

    Le découpage est stratifié (même proportion de positifs des deux côtés) et
    utilise la graine du projet, donc il est reproductible à l'identique.

    Le jeu de test ne sert qu'à la toute fin, pour mesurer la performance
    finale. Tous les réglages se font sur une partie du jeu d'entraînement.
    Sans cette discipline, on finit par choisir le modèle qui colle le mieux au
    jeu de test, et le score annoncé devient trop optimiste.

    Paramètres
    ----------
    donnees : pd.DataFrame
        Le jeu de données, avec au minimum colonne_texte et label.
    colonne_texte : str
        La colonne à utiliser en entrée. "texte_nettoye" par défaut. Le
        notebook BERT utilise "texte" parce que BERT a son propre
        prétraitement interne.

    Retourne
    --------
    tuple
        (X_train, X_test, y_train, y_test). Les X sont des séries de textes,
        les y des séries de labels 0 ou 1.
    """
    X = donnees[colonne_texte]
    y = donnees["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.PROPORTION_TEST,
        random_state=config.GRAINE_ALEATOIRE,
        stratify=y,
    )

    print(f"Entrainement : {formater_nombre(len(X_train))} tweets")
    print(f"Test         : {formater_nombre(len(X_test))} tweets")

    return X_train, X_test, y_train, y_test
