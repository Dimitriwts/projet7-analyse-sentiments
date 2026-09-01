"""
Les scores et les graphiques pour évaluer les modèles.

Les trois notebooks de modélisation passent tous par ce fichier. C'est voulu :
si chaque approche calculait ses scores à sa façon, je ne pourrais plus les
comparer. En les faisant passer par les mêmes fonctions, le tableau comparatif
final a un sens.

Comment je code les classes :
    label 0 = tweet négatif
    label 1 = tweet positif

Les modèles renvoient une probabilité entre 0 et 1, qui est toujours la
probabilité d'être positif. Un tweet est classé négatif quand cette
probabilité passe sous le seuil de décision.

Quelle mesure regarder ? Comme le jeu de données est parfaitement équilibré,
l'exactitude est déjà une mesure honnête, contrairement au cas d'un jeu
déséquilibré.

Mais le besoin d'Air Paradis n'est pas symétrique. L'entreprise veut repérer
les bad buzz : rater un tweet négatif coûte beaucoup plus cher que signaler à
tort un tweet positif. Dans le premier cas une crise démarre sans qu'on la
voie venir, dans le second un chargé de communication perd trente secondes à
lire un tweet anodin.

C'est donc le rappel sur la classe négative qui traduit le mieux le besoin
métier : parmi tous les tweets vraiment négatifs, combien le modèle en a-t-il
attrapés ? Toutes les fonctions ci-dessous mettent cette mesure en avant.
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

# Les étiquettes lisibles, utilisées sur tous les graphiques du projet.
NOMS_CLASSES = ["Negatif", "Positif"]

# Le seuil de décision par défaut. Au-dessus, le tweet est classé positif.
SEUIL_DECISION = 0.5


# ---------------------------------------------------------------------------
# 1. CALCULER LES SCORES
# ---------------------------------------------------------------------------
def calculer_metriques(y_vrai, y_probabilites, seuil: float = SEUIL_DECISION) -> dict:
    """
    Calcule les six mesures de référence du projet.

    Voici ce que chacune veut dire, en français simple.

    L'exactitude (accuracy) : la part de réponses justes, toutes classes
    confondues. C'est la mesure la plus intuitive, mais elle ne dit rien sur
    la répartition des erreurs.

    L'AUC ROC : AUC veut dire "Area Under the Curve", l'aire sous la courbe, et
    ROC vient de "Receiver Operating Characteristic", un nom hérité des radars
    de la Seconde Guerre mondiale. Concrètement, cette mesure répond à la
    question : si je prends au hasard un tweet positif et un tweet négatif,
    quelle est la probabilité que le modèle donne un score plus élevé au
    positif ? 0,5 c'est le hasard, 1,0 c'est parfait. Son gros intérêt ici :
    elle ne dépend pas du seuil qu'on choisit, elle mesure la qualité du
    classement produit par le modèle. Or Air Paradis voudra sans doute baisser
    le seuil pour attraper plus de bad buzz, donc autant choisir un modèle qui
    reste le meilleur quel que soit ce réglage.

    La précision sur la classe négative : parmi les tweets que le modèle
    signale comme négatifs, combien le sont vraiment ? Si elle est faible,
    l'équipe communication crouler sous les fausses alertes.

    Le rappel sur la classe négative : parmi les tweets vraiment négatifs,
    combien le modèle en a-t-il attrapés ? C'est la mesure métier : un rappel
    faible veut dire des bad buzz manqués.

    Le F1 : la moyenne harmonique de la précision et du rappel. Elle résume les
    deux en un seul chiffre, et elle ne monte que si les deux montent.

    Le F1 macro : la moyenne des F1 des deux classes, un résumé équilibré.

    Paramètres
    ----------
    y_vrai : array-like
        Les vrais labels (0 ou 1).
    y_probabilites : array-like
        La probabilité prédite d'être positif, entre 0 et 1.
    seuil : float
        Le seuil de décision.

    Retourne
    --------
    dict
        Les mesures, prêtes à être envoyées telles quelles à
        mlflow.log_metrics(). Les clés sont sans accent ni espace, parce que
        MLflow n'accepte pas tous les caractères dans un nom de mesure.
    """
    y_probabilites = np.asarray(y_probabilites).ravel()
    y_predit = (y_probabilites >= seuil).astype(int)

    return {
        # Vue d'ensemble
        "exactitude": accuracy_score(y_vrai, y_predit),
        "auc_roc": roc_auc_score(y_vrai, y_probabilites),
        "f1_macro": f1_score(y_vrai, y_predit, average="macro"),
        # La classe négative, celle qui intéresse Air Paradis.
        # pos_label=0 dit à scikit-learn de considérer le label 0 comme la
        # classe d'intérêt, alors qu'il prend le label 1 par défaut.
        "precision_negatif": precision_score(y_vrai, y_predit, pos_label=0),
        "rappel_negatif": recall_score(y_vrai, y_predit, pos_label=0),
        "f1_negatif": f1_score(y_vrai, y_predit, pos_label=0),
    }


def afficher_metriques(metriques: dict, titre: str = "Resultats") -> None:
    """
    Affiche les scores dans un tableau lisible.

    Toujours le même ordre et la même mise en forme, pour que les sorties des
    trois notebooks se comparent d'un coup d'oeil.
    """
    print("=" * 56)
    print(titre.upper())
    print("=" * 56)
    print(f"  Exactitude                : {metriques['exactitude']:.4f}")
    print(f"  AUC ROC                   : {metriques['auc_roc']:.4f}")
    print(f"  F1 macro                  : {metriques['f1_macro']:.4f}")
    print("-" * 56)
    print("  Classe NEGATIVE (les bad buzz a detecter)")
    print(f"    Precision               : {metriques['precision_negatif']:.4f}")
    print(f"    Rappel                  : {metriques['rappel_negatif']:.4f}   <-- mesure metier")
    print(f"    F1                      : {metriques['f1_negatif']:.4f}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# 2. LA MATRICE DE CONFUSION
# ---------------------------------------------------------------------------
def tracer_matrice_confusion(y_vrai, y_probabilites, titre: str,
                             seuil: float = SEUIL_DECISION):
    """
    Trace la matrice de confusion et renvoie le graphique.

    Une matrice de confusion est un tableau à quatre cases qui croise la
    vérité et la prédiction. Elle montre non seulement combien d'erreurs le
    modèle fait, mais surtout lesquelles.

    Comment la lire :
        chaque ligne   = ce que le tweet est vraiment
        chaque colonne = ce que le modèle a répondu

    La case qui compte pour Air Paradis est en haut à droite : les tweets
    vraiment négatifs que le modèle a classés positifs. Ce sont les bad buzz
    qu'on n'aurait pas vus venir.

    Je renvoie le graphique au lieu de l'afficher directement, pour que le
    notebook puisse d'abord l'enregistrer dans MLflow puis l'afficher.
    """
    y_probabilites = np.asarray(y_probabilites).ravel()
    y_predit = (y_probabilites >= seuil).astype(int)

    matrice = confusion_matrix(y_vrai, y_predit)

    figure, axe = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrice,
        annot=True,      # écrire les nombres dans les cases
        fmt=",d",        # avec un séparateur de milliers
        cmap="Blues",
        cbar=False,
        xticklabels=NOMS_CLASSES,
        yticklabels=NOMS_CLASSES,
        ax=axe,
    )
    axe.set_xlabel("Ce que le modele a repondu")
    axe.set_ylabel("Ce que le tweet est vraiment")
    axe.set_title(titre)

    # J'annote la case des bad buzz manqués, parce que c'est celle qu'il faut
    # regarder en priorité et qu'elle passe facilement inaperçue.
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
# 3. LA COURBE ROC
# ---------------------------------------------------------------------------
def tracer_courbe_roc(y_vrai, y_probabilites, titre: str):
    """
    Trace la courbe ROC et renvoie le graphique.

    La courbe ROC montre le compromis entre les tweets positifs correctement
    repérés et les tweets négatifs classés positifs à tort, et ce pour tous les
    seuils de décision possibles, pas seulement 0,5.

    Son intérêt : deux modèles peuvent avoir la même exactitude au seuil 0,5 et
    des courbes très différentes. Celui dont la courbe monte le plus haut reste
    meilleur quel que soit le réglage qu'on choisira ensuite.

    La diagonale en pointillés représente un modèle qui répondrait au hasard.
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
