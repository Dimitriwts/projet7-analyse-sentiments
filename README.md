# Air Paradis : prédire le sentiment d'un tweet

Prototype d'un produit d'intelligence artificielle qui devine si un tweet
exprime un sentiment négatif, pour anticiper les bad buzz sur les réseaux
sociaux.

Réalisé par MIC (Marketing Intelligence Consulting) pour la compagnie aérienne
Air Paradis.

---

## 1. L'objectif du projet

Air Paradis veut être prévenue le plus tôt possible quand un message négatif
circule à son sujet sur les réseaux sociaux. Ce qu'on me demande est un
prototype qui répond à une question simple :

> À partir du texte d'un tweet, est-ce qu'il exprime un sentiment négatif ou
> non ?

Air Paradis n'a aucune donnée client exploitable sur ce sujet. Je pars donc
d'un jeu de données public : Sentiment140, 1,6 million de tweets en anglais
déjà étiquetés positifs ou négatifs.

Le projet a deux objectifs en parallèle.

Côté modélisation, je compare trois approches de complexité croissante et je
garde la plus pertinente.

Côté MLOps, je montre une démarche industrialisée complète. MLOps est la
contraction de Machine Learning et Operations. C'est l'ensemble des pratiques
qui font qu'un modèle ne reste pas coincé dans un notebook, mais qu'il finit
en production, surveillé, et qu'on puisse le remplacer sans tout casser. En
pratique ça veut dire : versionner le code, garder la trace de chaque
expérimentation, tester automatiquement, déployer automatiquement, et
surveiller ce qui se passe une fois en ligne.

### Les trois approches comparées

| Numéro | Approche | Le principe | Le notebook |
|---|---|---|---|
| 1 | Modèle sur mesure simple | TF-IDF et régression logistique. Le tweet devient un sac de mots pondérés, sans notion d'ordre. Sert de point de comparaison de base. | `02_modele_classique.ipynb` |
| 2 | Modèle sur mesure avancé | Word embeddings et réseau de neurones récurrent. Le modèle tient compte de l'ordre des mots. Je compare deux jeux d'embeddings différents. | `03_modele_avance.ipynb` |
| 3 | Modèle avancé BERT | Un modèle de langue déjà entraîné sur des milliards de mots, qui comprend le contexte de chaque mot dans la phrase. | `04_modele_bert.ipynb` |

C'est le modèle sur mesure avancé (approche 2) qui est exposé par l'API
déployée sur le cloud, comme le demande le cahier des charges.

---

## 2. Le découpage des dossiers

```
PROJET7/
│
├── README.md                Ce fichier : l'objectif du projet et le découpage
├── requirements.txt         La liste des packages Python nécessaires
├── pytest.ini               Les réglages des tests unitaires
├── .gitignore               Les fichiers que Git doit ignorer
│
├── data/                    LES DONNÉES (jamais versionnées, trop lourdes)
│   ├── raw/                 déposer ici le fichier CSV de Sentiment140
│   └── processed/           les tweets nettoyés, produits par le notebook 01
│
├── src/                     LE CODE PARTAGÉ ENTRE TOUS LES NOTEBOOKS
│   ├── config.py            toutes les constantes au même endroit
│   ├── preprocessing.py     le nettoyage des tweets, partagé avec l'API
│   ├── data_loader.py       charger, échantillonner et découper les données
│   └── evaluation.py        calculer les scores et tracer les graphiques
│
├── notebooks/               LA MODÉLISATION, avec suivi MLflow
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_modele_classique.ipynb
│   ├── 03_modele_avance.ipynb
│   ├── 04_modele_bert.ipynb
│   └── 05_comparaison_et_export.ipynb
│
├── api/                     L'API DE PRÉDICTION, déployée sur Azure
│   ├── main.py              l'application FastAPI
│   ├── preprocessing.py     copie automatique de src/preprocessing.py
│   ├── requirements.txt     les dépendances légères de l'API seulement
│   └── artefacts/           le modèle allégé et le vocabulaire
│
├── tests/                   LES TESTS UNITAIRES AUTOMATISÉS
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── interface/               L'INTERFACE DE TEST, exécutée en local
│   └── app_streamlit.py     saisir un tweet, voir la prédiction, la valider
│
├── .github/workflows/       LE DÉPLOIEMENT CONTINU
│   └── azure-deploy.yml     tests puis mise en ligne automatique
│
└── docs/                    LES LIVRABLES DE COMMUNICATION
    ├── article_blog.md
    └── presentation/
```

### Pourquoi src/ et api/ sont deux mondes séparés

C'est la décision qui structure tout le projet, donc je l'explique.

Le dossier `src/` contient le code de recherche. Il tourne sur mon poste et il
peut dépendre de TensorFlow, de pandas, de tout ce dont j'ai besoin pour
entraîner.

Le dossier `api/` contient le code de production. Il tourne sur un serveur
Azure gratuit limité à 1 Go de mémoire. Il ne doit embarquer que le strict
minimum, donc pas de TensorFlow complet et pas de pandas.

Un seul fichier traverse la frontière : `preprocessing.py`. Le tweet doit être
nettoyé exactement de la même façon à l'entraînement et en production, sinon
les prédictions se dégradent en silence. C'est pour ça qu'il est écrit sans
aucune dépendance extérieure, que le pipeline de déploiement le recopie
automatiquement dans `api/`, et qu'un test vérifie que les deux copies sont
identiques.

### Est-ce que les notebooks tournent sans le dossier src/ ?

Non, et c'est voulu. Sans `src/`, le code de nettoyage serait recopié dans
chacun des cinq notebooks, plus une sixième fois dans l'API, et ces copies
finiraient par diverger. C'est exactement le problème que ce projet doit
apprendre à éviter.

---

## 3. Installation

### Ce qu'il faut avoir

- Python 3.13
- Git

### Mise en place

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

La première commande crée un environnement virtuel, c'est-à-dire un dossier
`.venv/` qui contient sa propre installation de Python et ses propres
packages, séparés de ceux du reste de la machine. Ça évite qu'un package
installé pour un autre projet vienne casser celui-ci, et ça garantit que le
`requirements.txt` livré ne contient que ce dont ce projet a besoin.

### Récupérer les données

Le fichier fait 230 Mo une fois décompressé, il n'est donc pas dans le dépôt
Git.

1. Télécharger le jeu de données Sentiment140.
2. Le décompresser.
3. Déposer le fichier CSV dans `data/raw/`.

Le fichier attendu s'appelle `training.1600000.processed.noemoticon.csv`. Un
autre nom marche aussi, le chargeur prend le premier `.csv` du dossier.

---

## 4. Faire tourner le projet

### Étape 1 : lancer MLflow

MLflow enregistre chaque expérimentation (les réglages, les scores, la durée)
et centralise le stockage des modèles. Il doit tourner avant les notebooks,
dans un terminal à part qu'on laisse ouvert.

```bash
.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

L'interface web est ensuite sur http://127.0.0.1:5000

Le stockage est une base SQLite et pas de simples fichiers, parce que le Model
Registry (le catalogue des modèles, qui leur donne un numéro de version) a
besoin d'une vraie base de données pour fonctionner.

### Étape 2 : exécuter les notebooks dans l'ordre

```bash
.venv\Scripts\jupyter.exe notebook
```

| Ordre | Notebook | Ce qu'il produit |
|---|---|---|
| 1 | `01_exploration_donnees.ipynb` | L'analyse du jeu de données et le fichier de travail `data/processed/tweets_prepares.parquet` |
| 2 | `02_modele_classique.ipynb` | Le modèle de référence et ses runs MLflow |
| 3 | `03_modele_avance.ipynb` | Les modèles à embeddings et leurs runs MLflow |
| 4 | `04_modele_bert.ipynb` | Le modèle BERT et ses runs MLflow |
| 5 | `05_comparaison_et_export.ipynb` | La comparaison des trois approches et l'export des artefacts vers `api/artefacts/` |

Le notebook 01 doit être lancé en premier. Les trois notebooks de modélisation
partent du fichier qu'il produit, c'est ce qui garantit qu'ils travaillent sur
exactement les mêmes tweets.

### Étape 3 : lancer l'API en local

```bash
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

La documentation interactive est sur http://127.0.0.1:8000/docs

L'API n'a besoin que de 5 packages, listés dans `api/requirements.txt`, contre
22 pour la modélisation. Pas de TensorFlow, pas de pandas, pas de
scikit-learn : le modèle est servi au format TensorFlow Lite, dont le moteur
d'exécution pèse quelques mégaoctets au lieu de 600.

> **Le serveur de production tourne en Python 3.13**, la même version qu'en
> développement. Le moteur `ai-edge-litert` fournit des versions Linux pour
> Python 3.11, 3.12 et 3.13, ce qui a été vérifié sur PyPI. Plus les deux
> environnements se ressemblent, moins il y a de surprises au déploiement.

### Étape 4 : lancer l'interface de test

```bash
.venv\Scripts\streamlit.exe run interface/app_streamlit.py
```

### Étape 5 : lancer les tests

```bash
.venv\Scripts\python.exe -m pytest
```

---

## 5. Le déploiement sur Azure

L'API tourne sur un Azure App Service en plan **F1 (gratuit)**, sous Linux, en
Python 3.13, dans la région France Central.

### La configuration de l'application

Ces quatre réglages sont faits dans le portail Azure. Ils ne sont pas dans le
code, donc je les note ici : sans eux, le déploiement échoue ou l'application
démarre sans jamais répondre.

| Où | Réglage | Valeur | Pourquoi |
|---|---|---|---|
| Configuration → Paramètres de la pile | Commande de démarrage | `python -m uvicorn main:app --host 0.0.0.0 --port 8000` | Sans elle, Azure cherche un `app.py`, ne le trouve pas et sert sa page par défaut. On écrit `main:app` et non `api.main:app` car seul le contenu du dossier `api/` est déployé |
| Configuration → Paramètres généraux | Authentification de base SCM | Activé | Le déploiement par profil de publication en a besoin. Microsoft la désactive par défaut sur les nouvelles applications |
| Variables d'environnement | `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` | Sans lui, Azure copie les fichiers mais n'exécute jamais `pip install`. L'API démarrerait sans ses dépendances |
| Secrets GitHub | `AZURE_WEBAPP_PUBLISH_PROFILE` | contenu du profil de publication | Les identifiants de déploiement. Ils ne doivent jamais apparaître dans le code |

### Les limites du plan gratuit

Le plan F1 offre 1 Go de mémoire et 60 minutes de processeur par jour, ce qui
suffit largement pour un prototype. Il a en revanche deux contraintes qu'il
faut connaître.

L'option « Toujours allumé » n'est pas disponible : l'application s'endort
après vingt minutes sans requête et met une trentaine de secondes à se
réveiller. C'est pour ça que le pipeline interroge la route de santé dix fois
de suite, espacées de quinze secondes, plutôt qu'une seule fois.

Et la mémoire disponible est la raison d'être de tout le travail d'allègement
du modèle décrit dans le notebook 05 : TensorFlow ne tiendrait pas dans 1 Go.

### Reproduire le déploiement

Une fois ces réglages en place, il n'y a plus rien à faire manuellement. Tout
envoi de code sur la branche principale déclenche le pipeline, qui lance les
tests puis déploie si et seulement s'ils passent.

---

## 6. La démarche MLOps mise en oeuvre

| La brique | L'outil | À quoi elle sert |
|---|---|---|
| Versionner le code | Git et GitHub | Garder l'historique complet et savoir qui a changé quoi |
| Suivre les expérimentations | MLflow Tracking | Enregistrer les réglages, les scores et la durée de chaque entraînement |
| Centraliser les modèles | MLflow Model Registry | Un modèle versionné, avec celui qui part en production clairement désigné |
| Tester automatiquement | pytest | Les tests tournent avant chaque déploiement |
| Déployer automatiquement | GitHub Actions | Tests puis mise en ligne sur Azure, sans intervention manuelle |
| Surveiller en production | Azure Application Insights | Remonter les tweets mal prédits et déclencher une alerte |

---

## 7. Les résultats

Section complétée à la fin de la modélisation.

---

## 8. Les packages utilisés

La liste complète avec les versions exactes est dans
[`requirements.txt`](requirements.txt) pour l'environnement de modélisation, et
dans [`api/requirements.txt`](api/requirements.txt) pour l'API en production.
