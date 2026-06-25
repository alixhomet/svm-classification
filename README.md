# Classification supervisée avec des Support Vector Machines (SVM)

Étude comparative de classifieurs **SVM** sur quatre jeux de données de l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/), du plus simple au plus complexe. L'objectif est d'analyser concrètement l'impact du **choix du noyau**, de la **normalisation**, de la **sélection de variables** et de l'**optimisation des hyperparamètres** sur les performances d'un modèle SVM.

## Objectif

Un SVM cherche l'hyperplan qui sépare deux classes en maximisant la marge entre elles. C'est un modèle performant mais sensible à l'échelle des variables (il repose sur des distances) et au choix du noyau. Ce projet met ces effets en évidence en progressant volontairement des données les plus simples (numériques, propres) aux plus difficiles (catégorielles, volumineuses).

## Jeux de données

| Dataset | Tâche | Particularité |
|---|---|---|
| **Breast Cancer** | Bénin / Malin (binaire) | Variables numériques — cas de base |
| **Spambase** | Spam / Non spam (binaire) | Nombreuses variables — focus sélection de features |
| **Car Evaluation** | Acceptabilité (4 classes) | Variables catégorielles + cible multi-classes |
| **Mushroom** | Comestible / Toxique (binaire) | Entièrement catégoriel et volumineux |

## Démarche

Pour chaque dataset, le notebook suit une trame cohérente :

1. **Exploration et préparation** — détection des valeurs manquantes, typage, encodage des variables catégorielles, mise au format `X` / `y`.
2. **Modélisation** — comparaison de noyaux (linéaire, RBF, polynomial), effet de la normalisation, évaluation par validation croisée 3-fold.
3. **Sélection de variables** — `SelectKBest` (ANOVA F) pour réduire la dimension sans perdre en performance.
4. **Optimisation** — `GridSearchCV` sur le noyau et le paramètre de régularisation `C`.
5. **Évaluation finale** — matrice de confusion et rapport de classification sur un jeu de test stratifié.

À partir du second dataset, les étapes récurrentes (validation croisée, évaluation, grid search) sont **factorisées dans des fonctions réutilisables** (`do_cv`, `do_cm_cr`, `do_grid_search`).

## Principaux enseignements

- **Normalisation** : sur un modèle basé sur des distances comme le SVM, mettre les variables à la même échelle change sensiblement les scores.
- **Noyau** : le RBF capture des frontières non linéaires que le noyau linéaire ne peut pas modéliser ; le bon choix dépend du dataset.
- **Sélection de variables** : sur des datasets riches (Spambase, Mushroom), un sous-ensemble de variables suffit souvent à conserver l'essentiel de la performance, pour un modèle plus simple et plus rapide.
- **Contexte métier** : sur le dataset médical (Breast Cancer), on distingue faux positifs et faux négatifs — un faux négatif (tumeur maligne non détectée) étant bien plus critique.
- **Validation croisée** : les fichiers étant triés, l'utilisation de folds **mélangés** (`shuffle=True`) est indispensable — sans elle, les scores sont fortement sous-estimés (ex. Mushroom passe de 0.89 à 1.0).

> Les scores détaillés (accuracy, précision, rappel, f1) sont produits directement dans le notebook à l'exécution.

## Technologies

- Python 3
- pandas, NumPy
- scikit-learn (SVC, GridSearchCV, SelectKBest, métriques)
- Jupyter Notebook

## Lancer le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/alixhomet/svm-classification.git
cd svm-classification

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Placer les fichiers CSV (breast_cancer, spambase, car, mushroom)
#    à la racine du projet, puis lancer Jupyter
jupyter notebook SVM_classification.ipynb
```

Les jeux de données sont disponibles publiquement sur l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/).

## Structure du dépôt

```text
.
├── SVM_classification.ipynb   # Notebook principal
├── requirements.txt
├── README.md
└── data/                      # Fichiers CSV (à ajouter)
    ├── breast_cancer.csv
    ├── spambase.csv
    ├── car.csv
    └── mushroom.csv
```

## Auteur

**Alix Monoury-Homet** — Élève ingénieure en Data Science & IA (ECE Paris)
