#### Elian Boaglio
#### Sophie Large
#### 5 SDBD - B1

# Apprentissage supervisé

---

## 1. Compréhension du jeu de données

### 1.1 Nettoyage et transformation des données

Pour préparer les données, nous avons effectué plusieurs opérations :  
- Nous avons conservé les valeurs numériques pour les colonnes `AGEP` (Age) et `WKHP` (Nombre d'heures travaillées par semaine).  
- Le niveau d'études a été regroupé en six catégories :  
  - jusqu'au collège  
  - jusqu'au lycée  
  - de bac à bac+2  
  - bac+3  
  - bac+5  
  - doctorat  
- Toutes les autres catégories ont été binarisées à l'aide du one-hot encoding.  

### 1.2 Séparation des données en ensembles d'entraînement et de test

Une fois les données nettoyées et transformées, nous avons procédé à leur séparation :
- Le jeu de données a été divisé en 80% pour l'entraînement et 20% pour le test.


---

## 2. Recherche de bons modèles

### 2.1 Qualité d’apprentissage avec les paramètres par défaut _(Expérimentation 1)_


| Jeu de données | Train      | Test      |
|----------------|-----------|-----------|
| Lignes         | 133052    | 33263     |
| Colonnes       | 792       | 792       |

Tableau 1 – Expérimentation (1) : Taille du jeu de données en entraînement et en test


| Métrique                 | Random Forest                 | AdaBoost                      | XGBoost                       |
|---------------------------|-------------------------------|-------------------------------|-------------------------------|
| Accuracy (CV 5 folds)     | 0.8119682424434174           | 0.7802137510146409           | 0.8312389141087695           |
| Temps de calcul (sec.)    | 668.8110718727112            | 316.02423310279846           | 71.75840497016907            |
| Matrice de confusion      | [[78251, 239], [258, 54304]] | [[64910, 13580], [15663, 38899]] | [[67350, 11140], [11314, 43248]] |

Tableau 2 – Expérimentation (1) : Résultats obtenus en entraînement


| Métrique                 | Random Forest                 | AdaBoost                      | XGBoost                       |
|---------------------------|-------------------------------|-------------------------------|-------------------------------|
| Accuracy (CV 5 folds)     | 0.8087063704416318           | 0.7720590445840724           | 0.817003878182966            |
| Matrice de confusion      | [[78251, 239], [258, 54304]] | [[16141, 3481], [4101, 9540]] | [[16621, 3001], [3086, 10555]] |

Tableau 3 – Expérimentation (1) : Résultats obtenus en test

**Remarques :**  
- XGBoost semble obtenir la meilleure précision sur les deux ensembles.  
- Random Forest est proche mais nécessite beaucoup plus de temps de calcul.  
- AdaBoost est plus rapide que Random Forest mais moins précis.

### 2.2 Optimisation des hyperparamètres des modèles _(Expérimentation 2)_


### 2.3 Analyse comparative des modèles _(Expérimentation 3)_


### 2.4 Inférence sur un autre jeu de données _(Expérimentation 4)_


### 2.5 Impact de la taille du jeu de données _(Expérimentation 5)_

---

## 3. Explicabilité des prédictions

### 3.1 Importance globale des attributs


### 3.2 Explications locales


### 3.3 Explication contrefactuelle

