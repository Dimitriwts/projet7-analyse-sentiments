"""
Evaluation des modeles : metriques et graphiques.

Ce module est utilise par les TROIS notebooks de modelisation. C'est
volontaire : si chaque approche calculait ses scores a sa facon, on ne
pourrait plus les comparer. En passant toutes par les memes fonctions, on
garantit que le tableau comparatif final a un sens.

-----------------------------------------------------------------------------
CONVENTION DE CODAGE DES CLASSES

    label 0  =  tweet NEGATIF
    label 1  =  tweet POSITIF

Les modeles renvoient une probabilite comprise entre 0 et 1, qui est toujours
la probabilite d'appartenir a la classe 1 (positif). Un tweet est classe
negatif lorsque cette probabilite passe sous le seuil de decision.
-----------------------------------------------------------------------------

QUELLE METRIQUE REGARDER ?

Le jeu de donnees etant parfaitement equilibre, l'exactitude (accuracy) est
deja une metrique honnete - contrairement au cas d'un jeu desequilibre.

Mais le besoin d'Air Paradis, lui, n'est PAS symetrique. L'entreprise veut
reperer les bad buzz : rater un tweet negatif coute beaucoup plus cher que
signaler a tort un tweet positif. Dans le premier cas une crise demarre sans
qu'on la voie venir ; dans le second, un charge de communication perd trente
secondes a lire un tweet anodin.

C'est donc le RAPPEL SUR LA CLASSE NEGATIVE qui traduit le mieux le besoin
metier : parmi tous les tweets reellement negatifs, quelle proportion le
modele a-t-il su detecter ? Toutes les fonctions ci-dessous mettent cette
metrique en avant.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Etiquettes lisibles, utilisees sur tous les graphiques du projet.
NOMS_CLASSES = ["Negatif", "Positif"]

# Seuil de decision par defaut. Au-dessus, le tweet est classe positif.
SEUIL_DECISION = 0.5


# ---------------------------------------------------------------------------
# 1. CALCUL DES METRIQUES
# ---------------------------------------------------------------------------
def calculer_metriques(y_vrai, y_probabilites, seuil: float = SEUIL_DECISION) -> dict:
    """
    Calcule les six metriques de reference du projet.

    Parametres
    ----------
    y_vrai : array-like
        Les vrais labels (0 ou 1).
    y_probabilites : array-like
        Probabilite predite d'appartenir a la classe 1 (positif), entre 0 et 1.
    seuil : float
        Seuil de decision. Au-dessus, le tweet est classe positif.

    Retourne
    --------
    dict
        Les metriques, pretes a etre envoyees telles quelles a
        `mlflow.log_metrics()`. Les cles sont sans accent ni espace, car
        MLflow n'accepte pas tous les caracteres dans un nom de metrique.
    """
    y_probabilites = np.asarray(y_probabilites).ravel()
    y_predit = (y_probabilites >= seuil).astype(int)

    return {
        # --- Vue d'ensemble -----------------------------------------------
        # Part de predictions correctes, toutes classes confondues.
        "exactitude": accuracy_score(y_vrai, y_predit),
        # Aire sous la courbe ROC. Interet particulier : elle ne depend pas du
        # seuil choisi, et mesure donc la qualite intrinseque du classement
        # produit par le modele. 0,5 = hasard, 1,0 = parfait.
        "auc_roc": roc_auc_score(y_vrai, y_probabilites),
        # Moyenne des F1 des deux classes. Resume equilibre en un seul chiffre.
        "f1_macro": f1_score(y_vrai, y_predit, average="macro"),

        # --- Classe NEGATIVE : la classe qui interesse le metier ----------
        # Parmi les tweets SIGNALES comme negatifs, combien le sont vraiment ?
        # Une precision faible sature l'equipe communication de fausses alertes.
        "precision_negatif": precision_score(y_vrai, y_predit, pos_label=0),
        # Parmi les tweets REELLEMENT negatifs, combien ont ete detectes ?
        # C'est LA metrique metier : un rappel faible = des bad buzz manques.
        "rappel_negatif": recall_score(y_vrai, y_predit, pos_label=0),
        # Compromis entre les deux precedentes.
        "f1_negatif": f1_score(y_vrai, y_predit, pos_label=0),
    }


def afficher_metriques(metriques: dict, titre: str = "Resultats") -> None:
    """
    Affiche les metriques dans un tableau lisible en console.

    Rend les sorties des notebooks comparables d'une approche a l'autre : meme
    ordre, meme mise en forme, meme nombre de decimales.
    """
    print("=" * 52)
    print(titre.upper())
    print("=" * 52)
    print(f"  Exactitude (accuracy)     : {metriques['exactitude']:.4f}")
    print(f"  AUC ROC                   : {metriques['auc_roc']:.4f}")
    print(f"  F1 macro                  : {metriques['f1_macro']:.4f}")
    print("-" * 52)
    print("  Classe NEGATIVE (bad buzz a detecter)")
    print(f"    Precision               : {metriques['precision_negatif']:.4f}")
    print(f"    Rappel                  : {metriques['rappel_negatif']:.4f}   <-- metrique metier")
    print(f"    F1                      : {metriques['f1_negatif']:.4f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# 2. MATRICE DE CONFUSION
# ---------------------------------------------------------------------------
def tracer_matrice_confusion(y_vrai, y_probabilites, titre: str,
                             seuil: float = SEUIL_DECISION):
    """
    Trace la matrice de confusion et renvoie la figure matplotlib.

    La figure est RENVOYEE plutot qu'affichee directement : cela permet au
    notebook de l'enregistrer comme artefact MLflow avant de l'afficher.

    Lecture de la matrice :
        ligne   = la verite
        colonne = la prediction du modele

    La case qui compte pour Air Paradis est en haut a droite : les tweets
    reellement negatifs que le modele a classes positifs. Ce sont les bad buzz
    manques.
    """
    y_probabilites = np.asarray(y_probabilites).ravel()
    y_predit = (y_probabilites >= seuil).astype(int)

    matrice = confusion_matrix(y_vrai, y_predit)

    figure, axe = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrice,
        annot=True,          # ecrire les effectifs dans les cases
        fmt=",d",            # separateur de milliers
        cmap="Blues",
        cbar=False,
        xticklabels=NOMS_CLASSES,
        yticklabels=NOMS_CLASSES,
        ax=axe,
    )
    axe.set_xlabel("Prediction du modele")
    axe.set_ylabel("Verite")
    axe.set_title(titre)

    # On annote explicitement la case des bad buzz manques : c'est elle qu'il
    # faut regarder en priorite, et elle passe facilement inapercue.
    negatifs_manques = matrice[0, 1]
    total_negatifs = matrice[0].sum()
    axe.text(
        0.5, -0.18,
        f"Bad buzz manques : {negatifs_manques:,} sur {total_negatifs:,} "
        f"({negatifs_manques / total_negatifs:.1%})".replace(",", " "),
        transform=axe.transAxes,
        ha="center",
        fontsize=10,
        color="#b22222",
    )

    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 3. COURBE ROC
# ---------------------------------------------------------------------------
def tracer_courbe_roc(y_vrai, y_probabilites, titre: str):
    """
    Trace la courbe ROC et renvoie la figure matplotlib.

    La courbe ROC montre le compromis entre les tweets positifs correctement
    identifies et les tweets negatifs classes positifs a tort, pour TOUS les
    seuils de decision possibles.

    Son interet ici : elle evalue le modele independamment du seuil. Deux
    modeles peuvent avoir la meme exactitude a 0,5 et des courbes tres
    differentes ; celui dont la courbe est la plus haute reste meilleur quel
    que soit le reglage retenu ensuite.

    La diagonale represente un modele qui repondrait au hasard.
    """
    y_probabilites = np.asarray(y_probabilites).ravel()
    taux_faux_positifs, taux_vrais_positifs, _ = roc_curve(y_vrai, y_probabilites)
    aire = roc_auc_score(y_vrai, y_probabilites)

    figure, axe = plt.subplots(figsize=(6, 5))
    axe.plot(taux_faux_positifs, taux_vrais_positifs,
             linewidth=2, label=f"Modele (AUC = {aire:.4f})")
    axe.plot([0, 1], [0, 1], "k--", linewidth=1, label="Hasard (AUC = 0,5000)")

    axe.set_xlabel("Taux de faux positifs")
    axe.set_ylabel("Taux de vrais positifs")
    axe.set_title(titre)
    axe.legend(loc="lower right")
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 1.02)

    figure.tight_layout()
    return figure
