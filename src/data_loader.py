"""
Chargement et preparation du jeu de donnees Sentiment140.

Ce module regroupe tout ce qui touche aux donnees, pour que les notebooks
n'aient plus qu'a appeler une fonction. L'interet est double :

  - Les trois notebooks de modelisation partent EXACTEMENT du meme jeu de
    donnees, decoupe de la meme facon. C'est la condition pour que la
    comparaison des trois approches soit valable.
  - Le nettoyage des 1,6 million de tweets, qui prend une bonne minute, n'est
    fait qu'UNE FOIS. Le resultat est sauvegarde sur le disque et les
    notebooks suivants le rechargent instantanement.

Le jeu de donnees : Sentiment140, 1 600 000 tweets collectes en 2009 et
etiquetes automatiquement selon les emoticones qu'ils contenaient (les
emoticones ont ensuite ete retirees du texte). Parfaitement equilibre :
800 000 tweets negatifs, 800 000 positifs.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import nettoyer_tweet


# ---------------------------------------------------------------------------
# 1. LOCALISATION DU FICHIER BRUT
# ---------------------------------------------------------------------------
def trouver_fichier_brut() -> Path:
    """
    Localise le fichier CSV du jeu de donnees dans data/raw/.

    Cherche d'abord le nom officiel du dataset. S'il est absent, se rabat sur
    n'importe quel fichier .csv present dans le dossier - le fichier est
    parfois redistribue sous un autre nom.

    Retourne
    --------
    Path
        Le chemin du fichier trouve.

    Leve
    ----
    FileNotFoundError
        Si aucun fichier CSV n'est present, avec un message expliquant
        precisement quoi faire.
    """
    # Cas nominal : le fichier porte son nom d'origine.
    if config.FICHIER_DATASET_BRUT.exists():
        return config.FICHIER_DATASET_BRUT

    # Repli : on accepte n'importe quel CSV depose dans data/raw/.
    fichiers_csv = sorted(config.DOSSIER_DONNEES_BRUTES.glob("*.csv"))
    if fichiers_csv:
        return fichiers_csv[0]

    # Rien trouve : on explique quoi faire plutot que de lever une erreur seche.
    # Le message est construit ligne par ligne puis assemble : c'est plus lisible
    # qu'une longue concatenation, et cela evite un piege classique de Python
    # (dans "abc" "=" * 70, la concatenation des litteraux a lieu AVANT la
    # multiplication, et l'on obtient 70 repetitions de "abc=").
    lignes = [
        "",
        "=" * 70,
        "JEU DE DONNEES INTROUVABLE",
        "=" * 70,
        f"Aucun fichier .csv dans : {config.DOSSIER_DONNEES_BRUTES}",
        "",
        "A FAIRE :",
        "  1. Telecharger le jeu de donnees Sentiment140 (~80 Mo compresse).",
        "  2. Le decompresser.",
        "  3. Deposer le fichier CSV dans le dossier ci-dessus.",
        "",
        "Le fichier attendu s'appelle normalement :",
        "  training.1600000.processed.noemoticon.csv",
        "",
        "Un autre nom fonctionne aussi : ce chargeur prend le premier .csv",
        "qu'il trouve dans le dossier.",
        "=" * 70,
    ]
    raise FileNotFoundError("\n".join(lignes))


# ---------------------------------------------------------------------------
# 2. LECTURE DU FICHIER BRUT
# ---------------------------------------------------------------------------
def charger_donnees_brutes() -> pd.DataFrame:
    """
    Lit le fichier CSV brut et renvoie un tableau exploitable.

    Deux specificites du fichier Sentiment140 sont prises en charge ici :

      - Il n'a PAS de ligne d'en-tete. Sans `header=None`, pandas prendrait
        le premier tweet pour les noms de colonnes et on perdrait une ligne.
      - Il n'est pas en UTF-8. Certains octets feraient echouer la lecture ;
        l'encodage latin-1 les accepte tous.

    Retourne
    --------
    pd.DataFrame
        Deux colonnes seulement :
          - `texte`  : le texte brut du tweet
          - `label`  : 0 pour negatif, 1 pour positif

    Note : les quatre autres colonnes du fichier (identifiant, date, requete,
    utilisateur) sont ecartees. Le projet demande de predire le sentiment a
    partir du seul contenu du tweet : l'API ne recevra qu'un texte, elle
    n'aura ni la date ni l'auteur. Entrainer le modele sur des informations
    indisponibles en production n'aurait aucun sens.
    """
    chemin = trouver_fichier_brut()
    print(f"Lecture de : {chemin.name}")

    donnees = pd.read_csv(
        chemin,
        encoding=config.ENCODAGE_DATASET,
        header=None,
        names=config.COLONNES_DATASET,
    )

    # Conversion de la polarite brute (0 / 4) en label binaire (0 / 1).
    # On passe par une comparaison booleenne convertie en entier : c'est plus
    # rapide et plus lisible qu'un dictionnaire de correspondance.
    donnees["label"] = (donnees["polarite"] == config.POLARITE_POSITIVE_BRUTE).astype(int)

    resultat = donnees[["texte", "label"]].copy()

    print(f"  {len(resultat):,} tweets charges".replace(",", " "))
    print(f"  Negatifs : {(resultat['label'] == 0).sum():,}".replace(",", " "))
    print(f"  Positifs : {(resultat['label'] == 1).sum():,}".replace(",", " "))

    return resultat


# ---------------------------------------------------------------------------
# 3. PREPARATION (nettoyage) - a executer une seule fois
# ---------------------------------------------------------------------------
def preparer_donnees(forcer: bool = False) -> pd.DataFrame:
    """
    Nettoie l'integralite des tweets et sauvegarde le resultat sur le disque.

    Cette operation prend environ une minute sur 1,6 million de tweets. Elle
    n'est donc faite qu'une fois : le resultat est ecrit au format Parquet,
    et tous les appels suivants se contentent de relire ce fichier (moins
    d'une seconde).

    Le format Parquet est choisi plutot que le CSV parce qu'il est compresse
    (environ 5 fois plus leger), qu'il se relit beaucoup plus vite, et qu'il
    conserve les types des colonnes sans avoir a les redeclarer.

    Parametres
    ----------
    forcer : bool
        Si True, refait le nettoyage meme si le fichier prepare existe deja.
        Utile apres avoir modifie `src/preprocessing.py`.

    Retourne
    --------
    pd.DataFrame
        Colonnes : `texte` (brut), `texte_nettoye`, `label`.
    """
    fichier_prepare = config.FICHIER_DONNEES_PREPAREES

    # Si le travail a deja ete fait, on ne le refait pas.
    if fichier_prepare.exists() and not forcer:
        print(f"Fichier prepare deja present : {fichier_prepare.name}")
        print("  (utiliser preparer_donnees(forcer=True) pour le regenerer)")
        return pd.read_parquet(fichier_prepare)

    donnees = charger_donnees_brutes()

    print("\nNettoyage des tweets en cours...")
    donnees["texte_nettoye"] = donnees["texte"].apply(nettoyer_tweet)

    # Certains tweets ne contenaient QUE des liens ou des mentions : une fois
    # nettoyes, il ne reste rien. On les retire, car un texte vide n'apprend
    # rien au modele et fausserait les metriques.
    nombre_avant = len(donnees)
    donnees = donnees[donnees["texte_nettoye"].str.len() > 0].copy()
    nombre_retires = nombre_avant - len(donnees)

    print(f"  {nombre_retires:,} tweets vides apres nettoyage, retires".replace(",", " "))
    print(f"  {len(donnees):,} tweets conserves".replace(",", " "))

    # Sauvegarde. `index=False` evite d'ecrire la colonne d'index de pandas,
    # qui ne sert a rien ici.
    fichier_prepare.parent.mkdir(parents=True, exist_ok=True)
    donnees.to_parquet(fichier_prepare, index=False)
    print(f"\nSauvegarde : {fichier_prepare}")

    return donnees


# ---------------------------------------------------------------------------
# 4. CHARGEMENT D'UN ECHANTILLON
# ---------------------------------------------------------------------------
def charger_echantillon(taille: int | None = None) -> pd.DataFrame:
    """
    Recharge les donnees preparees, eventuellement reduites a un echantillon.

    L'echantillonnage est STRATIFIE : on prend autant de tweets negatifs que
    de positifs. Un tirage purement aleatoire donnerait des proportions
    legerement differentes a chaque taille, ce qui rendrait les scores des
    trois approches un peu moins comparables entre eux.

    Le tirage utilise la graine aleatoire du projet : deux appels avec la
    meme taille renvoient exactement les memes tweets.

    Parametres
    ----------
    taille : int ou None
        Nombre total de tweets voulu. None = tout le jeu de donnees.

    Retourne
    --------
    pd.DataFrame
        Colonnes : `texte`, `texte_nettoye`, `label`.
    """
    if not config.FICHIER_DONNEES_PREPAREES.exists():
        raise FileNotFoundError(
            "Les donnees n'ont pas encore ete preparees.\n"
            "Executez d'abord le notebook 01_exploration_donnees.ipynb, "
            "ou appelez preparer_donnees()."
        )

    donnees = pd.read_parquet(config.FICHIER_DONNEES_PREPAREES)

    if taille is None or taille >= len(donnees):
        print(f"Jeu de donnees complet : {len(donnees):,} tweets".replace(",", " "))
        return donnees

    # Echantillonnage stratifie : on tire la moitie dans chaque classe.
    # On procede classe par classe avec une simple boucle plutot qu'avec un
    # groupby : c'est plus long d'une ligne, mais on voit exactement ce qui
    # se passe, et cela n'utilise aucune API pandas en cours d'obsolescence.
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
        # On remelange : sinon tous les negatifs seraient devant tous les
        # positifs, ce qui perturberait l'entrainement par lots (batches).
        .sample(frac=1.0, random_state=config.GRAINE_ALEATOIRE)
        .reset_index(drop=True)
    )

    print(f"Echantillon stratifie : {len(echantillon):,} tweets".replace(",", " "))
    print(f"  Negatifs : {(echantillon['label'] == 0).sum():,}".replace(",", " "))
    print(f"  Positifs : {(echantillon['label'] == 1).sum():,}".replace(",", " "))

    return echantillon


# ---------------------------------------------------------------------------
# 5. DECOUPAGE ENTRAINEMENT / TEST
# ---------------------------------------------------------------------------
def separer_train_test(donnees: pd.DataFrame, colonne_texte: str = "texte_nettoye"):
    """
    Separe le jeu de donnees en un jeu d'entrainement et un jeu de test.

    Le decoupage est stratifie (meme proportion de positifs de chaque cote)
    et utilise la graine du projet, donc reproductible a l'identique.

    Le jeu de test n'est utilise QU'A LA FIN, pour mesurer la performance
    finale. Tous les reglages (choix d'hyperparametres, arret anticipe) se
    font sur une portion du jeu d'entrainement. Sans cette discipline, on
    finit par choisir le modele qui colle le mieux au jeu de test, et le
    score annonce devient optimiste.

    Parametres
    ----------
    donnees : pd.DataFrame
        Le jeu de donnees, contenant au minimum `colonne_texte` et `label`.
    colonne_texte : str
        Colonne a utiliser comme entree. "texte_nettoye" par defaut ;
        le notebook BERT utilise "texte" car BERT a son propre
        pretraitement interne.

    Retourne
    --------
    tuple
        (X_train, X_test, y_train, y_test), ou les X sont des Series de
        textes et les y des Series de labels 0/1.
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

    print(f"Entrainement : {len(X_train):,} tweets".replace(",", " "))
    print(f"Test         : {len(X_test):,} tweets".replace(",", " "))

    return X_train, X_test, y_train, y_test
