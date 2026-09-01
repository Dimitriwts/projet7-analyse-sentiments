# Air Paradis — Prédiction de sentiment sur les tweets

Prototype d'un produit d'intelligence artificielle capable de **prédire le
sentiment associé à un tweet**, afin d'anticiper les bad buzz sur les réseaux
sociaux.

Réalisé par **MIC (Marketing Intelligence Consulting)** pour la compagnie
aérienne **Air Paradis**.

---

## 1. Objectif du projet

Air Paradis souhaite être alertée le plus tôt possible lorsqu'un message
négatif circule à son sujet sur les réseaux sociaux. Le produit demandé est un
**prototype fonctionnel** répondant à un besoin simple à énoncer :

> À partir du texte d'un tweet, déterminer s'il exprime un sentiment
> **négatif** ou **non négatif**.

Air Paradis ne dispose d'aucune donnée client exploitable pour ce sujet. Le
prototype s'appuie donc sur un jeu de données open source : **Sentiment140**,
1,6 million de tweets en anglais, étiquetés positif ou négatif.

Le projet poursuit deux objectifs de front :

1. **Objectif modélisation** — comparer trois approches de complexité
   croissante et retenir la plus pertinente.
2. **Objectif MLOps** — démontrer une démarche industrialisée complète :
   suivi des expérimentations, tests automatisés, déploiement continu et
   suivi de la performance du modèle en production.

### Les trois approches comparées

| # | Approche | Principe | Notebook |
|---|----------|----------|----------|
| 1 | **Modèle sur mesure simple** | TF-IDF + régression logistique. Le texte est réduit à un sac de mots pondérés, sans notion d'ordre. Sert de référence de base. | `02_modele_classique.ipynb` |
| 2 | **Modèle sur mesure avancé** | Word embeddings + réseau de neurones récurrent. Le modèle exploite l'ordre des mots. Deux jeux d'embeddings différents sont comparés. | `03_modele_avance.ipynb` |
| 3 | **Modèle avancé BERT** | Modèle de langue pré-entraîné (DistilBERT), qui comprend le contexte de chaque mot dans la phrase. | `04_modele_bert.ipynb` |

C'est le **modèle sur mesure avancé** (approche 2) qui est exposé via l'API
déployée sur le cloud, conformément au cahier des charges.

---

## 2. Découpage des dossiers

```
PROJET7/
│
├── README.md                Ce fichier — objectif du projet et découpage
├── requirements.txt         Packages nécessaires aux notebooks
├── pytest.ini               Configuration des tests unitaires
├── .gitignore               Fichiers exclus du versioning
│
├── data/                    DONNÉES (jamais versionnées — trop volumineuses)
│   ├── raw/                 → déposer ici le CSV Sentiment140
│   └── processed/           → tweets nettoyés, générés par le notebook 01
│
├── src/                     CODE SOURCE PARTAGÉ
│   ├── config.py            Chemins et constantes — source unique de vérité
│   ├── preprocessing.py     Nettoyage des tweets (partagé entraînement / API)
│   └── data_loader.py       Chargement, préparation et découpage des données
│
├── notebooks/               MODÉLISATION (avec suivi MLflow)
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_modele_classique.ipynb
│   ├── 03_modele_avance.ipynb
│   ├── 04_modele_bert.ipynb
│   └── 05_comparaison_et_export.ipynb
│
├── api/                     API DE PRÉDICTION (déployée sur Azure)
│   ├── main.py              Application FastAPI
│   ├── preprocessing.py     Copie automatique de src/preprocessing.py
│   ├── requirements.txt     Dépendances légères de l'API uniquement
│   └── artefacts/           Modèle allégé + vocabulaire, chargés au démarrage
│
├── tests/                   TESTS UNITAIRES AUTOMATISÉS
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── interface/               INTERFACE DE TEST LOCALE
│   └── app_streamlit.py     Saisie d'un tweet, appel de l'API, validation
│
├── .github/workflows/       DÉPLOIEMENT CONTINU
│   └── azure-deploy.yml     Tests puis déploiement automatique sur Azure
│
└── docs/                    LIVRABLES DE COMMUNICATION
    ├── article_blog.md      Article de blog
    └── presentation/        Support de présentation
```

### Deux dossiers, deux mondes

La séparation entre `src/` et `api/` mérite une explication, car elle
structure tout le projet.

`src/` contient le code de **recherche** : il peut dépendre de TensorFlow, de
pandas, de tout ce qui est nécessaire pour entraîner. Il tourne sur un poste
de développement.

`api/` contient le code de **production** : il tourne sur un serveur Azure
gratuit limité à **1 Go de mémoire**. Il ne doit embarquer que le strict
minimum — pas de TensorFlow complet, pas de pandas.

Un seul fichier traverse la frontière : `preprocessing.py`. Le tweet doit
être nettoyé **exactement de la même façon** à l'entraînement et en
production, sinon les prédictions se dégradent silencieusement. Ce fichier
est donc écrit sans aucune dépendance externe, et le pipeline de déploiement
le recopie automatiquement dans `api/`. Un test unitaire vérifie que les deux
copies sont identiques.

---

## 3. Installation

### Prérequis

- Python 3.13
- Git

### Mise en place

```bash
# 1. Créer l'environnement isolé
python -m venv .venv

# 2. L'activer  (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Récupérer le jeu de données

Le fichier fait 230 Mo décompressé : il n'est pas versionné dans le dépôt.

1. Télécharger le jeu de données **Sentiment140**.
2. Le décompresser.
3. Déposer le fichier CSV dans **`data/raw/`**.

Le fichier attendu s'appelle `training.1600000.processed.noemoticon.csv`.
Un autre nom fonctionne également : le chargeur prend le premier `.csv`
trouvé dans le dossier.

---

## 4. Exécution du projet

### Étape 1 — Lancer le serveur MLflow

MLflow enregistre chaque expérimentation (paramètres, métriques, modèle) et
centralise le stockage des modèles. Il doit tourner **avant** les notebooks,
dans un terminal dédié qu'on laisse ouvert.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Interface web : <http://127.0.0.1:5000>

> Le support de stockage est une base SQLite et non de simples fichiers, car
> le **Model Registry** — la centralisation des modèles — exige une base de
> données.

### Étape 2 — Exécuter les notebooks dans l'ordre

```bash
jupyter notebook
```

| Ordre | Notebook | Ce qu'il produit |
|-------|----------|------------------|
| 1 | `01_exploration_donnees.ipynb` | Analyse du jeu de données, nettoyage, fichier `data/processed/tweets_prepares.parquet` |
| 2 | `02_modele_classique.ipynb` | Modèle de référence + runs MLflow |
| 3 | `03_modele_avance.ipynb` | Modèles à embeddings + runs MLflow |
| 4 | `04_modele_bert.ipynb` | Modèle BERT + runs MLflow |
| 5 | `05_comparaison_et_export.ipynb` | Comparaison des trois approches, export des artefacts vers `api/artefacts/` |

Le notebook 01 doit impérativement être exécuté en premier : les trois
notebooks de modélisation partent du fichier qu'il produit, ce qui garantit
qu'ils travaillent tous sur exactement les mêmes tweets.

### Étape 3 — Lancer l'API en local

```bash
uvicorn api.main:app --reload
```

Documentation interactive : <http://127.0.0.1:8000/docs>

### Étape 4 — Lancer l'interface de test

```bash
streamlit run interface/app_streamlit.py
```

### Étape 5 — Lancer les tests unitaires

```bash
pytest
```

---

## 5. Démarche MLOps mise en œuvre

| Brique | Outil | Rôle |
|--------|-------|------|
| Versioning du code | Git + GitHub | Historique complet, traçabilité des modifications |
| Suivi des expérimentations | MLflow Tracking | Paramètres, métriques et durée de chaque entraînement |
| Stockage centralisé des modèles | MLflow Model Registry | Un modèle versionné et promu en production |
| Tests automatisés | pytest | Exécutés avant chaque déploiement |
| Déploiement continu | GitHub Actions | Tests puis mise en ligne automatique sur Azure |
| Suivi en production | Azure Application Insights | Traces des prédictions mal jugées + alerte automatique |

---

## 6. Résultats

*(Section complétée à l'issue de la modélisation.)*

---

## 7. Packages utilisés

La liste complète et les versions exactes figurent dans
[`requirements.txt`](requirements.txt) pour l'environnement de modélisation,
et dans [`api/requirements.txt`](api/requirements.txt) pour l'API en
production.
