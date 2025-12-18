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
- Le jeu de données a été divisé en 75% pour l'entraînement et 25% pour le test.


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

Même jeu de données que pour la partie précédente
### Random Forest - Optimisation des Hyperparamètres

**Résultats d'Optimisation**

**base param**

 max_features='sqrt',
 max_depth = 10 ,
 min_samples_split=6,
 min_samples_leaf= 2 ,
 n_estimators=100,
 random_state=42

On prend ces paramètres de base et on les fait varier 1 à 1 séparemment pour observer l'évolution de l'accuracy. 

**n_estimators** (Nombre d'arbres)
| n_estimators |  CV Accuracy|
|--------------|-------------|
| 50           | 0.8109      |
| 75           | 0.8127      |
| 100          | 0.8129      |
| 150          | 0.8134      |
| 200          | 0.8138      |

**Meilleure valeur:** `n_estimators = 200` ( CV Accuracy = 0.8138)

---

**max_depth** (Profondeur maximale)
| max_depth |  CV Accuracy |
|-----------|-------------|
| None      | 0.8138      |
| 5         | 0.6923      |
| 10        | 0.7802      |
| 15        | 0.7956      |
| 20        | 0.8029      |

**Meilleure valeur:** `max_depth = None` (CV Accuracy = 0.8138)

---

**min_samples_split** (Échantillons minimum pour diviser)
| min_samples_split | CV Accuracy |
|-------------------|-------------|
| 2                 | 0.8138      |
| 4                 | 0.8168      |
| 6                 | 0.8183      |
| 8                 | 0.8187      |
| 10                | 0.8193      |

**Meilleure valeur:** `min_samples_split = 10` ( CV Accuracy = 0.8193)

---

**min_samples_leaf** (Échantillons minimum par feuille)
| min_samples_leaf |  CV Accuracy |
|------------------|-------------|
| 1                | 0.8193      |
| 2                | 0.8174      |
| 3                | 0.8148      |
| 4                | 0.8141      |
| 5                | 0.8121      |

**Meilleure valeur:** `min_samples_leaf = 1` ( CV Accuracy= 0.8193)

**max_features** (Nombre maximum de caractéristiques)
| max_features |  CV Accuracy|
|--------------|-------------|
| sqrt         | 0.8193      |
| log2         | 0.8224      |

**Meilleure valeur:** `max_features = 'log2'` ( CV Accuracy = 0.8224)

**Paramètres Finaux Optimaux**
| Paramètre          | Meilleure Valeur |
|--------------------|------------------|
| n_estimators       | 200              |
| max_depth          | None             |
| min_samples_split  | 10               |
| min_samples_leaf   | 1                |
| max_features       | log2             |

#### **Résultats en Entraînement**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.9372 |
| **Temps de calcul** | 33.9610 sec |

**Matrice de confusion :**
<pre>
[[74934 3556]
[ 4799 49763]]
</pre>
####  **Résultats en Test**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.8191 |
| **Temps de calcul** | 4.3350 sec |

**Matrice de confusion :**

<pre>
[[16954  2668]
 [ 3350 10291]]

</pre>




---

#### Résumé
Le modèle Random Forest a été optimisé grâce à une recherche systématique sur cinq hyperparamètres clés. La meilleure configuration a obtenu une précision de validation croisée de **82.24%**. On note particulièrement que :
- `max_features = 'log2'` a apporté une amélioration supplémentaire significative
- `min_samples_split = 10` a fourni une amélioration notable de la précision
- `max_depth = None` (profondeur illimitée) a donné les meilleurs résultats
- L'augmentation de `n_estimators` jusqu'à 200 a produit des améliorations graduelles

**Performance finale :**
- **Score entraînement** : 93.72%
- **Score test** : 81.91%
- La différence entre entraînement et test suggère un léger surapprentissage

---

### AdaBoost - Optimisation des Hyperparamètres

**Résultats d'Optimisation**

**n_estimators** (Nombre d'estimateurs)
| n_estimators | CV Accuracy |
|--------------|-------------|
| 50           | 0.7764      |
| 75           | 0.7802      |
| 100          | 0.7838      |
| 150          | 0.7859      |
| 200          | 0.7875      |

**Meilleure valeur:** `n_estimators = 200` (CV Accuracy = 0.7875)

---

**learning_rate** (Taux d'apprentissage)
| learning_rate | CV Accuracy |
|---------------|-------------|
| 0.01          | 0.6784      |
| 0.05          | 0.7452      |
| 0.1           | 0.7667      |
| 0.5           | 0.7831      |
| 1.0           | 0.7875      |

**Meilleure valeur:** `learning_rate = 1.0` (CV Accuracy = 0.7875)

**Paramètres Finaux Optimaux**
| Paramètre      | Meilleure Valeur |
|----------------|------------------|
| n_estimators   | 200              |
| learning_rate  | 1.0              |

**Meilleurs paramètres finaux :**
{
'n_estimators': 200,
'learning_rate': 1.0
}


#### **Résultats en Entraînement**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.7904 |
| **Temps de calcul** | 18.5088 sec |

**Matrice de confusion :**
<seq>
[[65888 12652]
[15237 39325]]
</seq>

#### **Résultats en Test**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.7820 |
| **Temps de calcul** | 4.4093 sec |

**Matrice de confusion :**
<seq>
[[16349 3273]
[3978 9663]]
</seq>

Le modèle AdaBoost a été optimisé sur deux hyperparamètres principaux. La meilleure configuration a obtenu une précision de validation croisée de **78.75%**. Les constatations sont :
- `learning_rate = 1.0` a donné les meilleurs résultats
- `n_estimators = 200` a fourni le nombre optimal d'estimateurs
- L'algorithme montre des performances plus modestes mais stables

**Performance finale :**
- **Score entraînement** : 79.04%
- **Score test** : 78.20%
- Très faible écart entre entraînement et test, indiquant une bonne généralisation

### XGBoost - Optimisation des Hyperparamètres
Même principe pour XGBoost ou on a fait varier 1 à 1 chaque paramètre individuellement, on obtient :

**Paramètres Finaux Optimaux**
| Paramètre          | Meilleure Valeur |
|--------------------|------------------|
| n_estimators       | 300              |
| learning_rate      | 0.2              |
| max_depth          | 7                |
| subsample          | 0.8              |
| colsample_bytree   | 0.8              |

#### **Résultats en Entraînement**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.8455 |
| **Temps de calcul** | 0.3632 sec |

**Matrice de confusion :**
<seq>
[[68188 10302]
[10259 44303]]
</seq>

#### **Résultats en Test**

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 0.8221 |
| **Temps de calcul** | 0.1570 sec |

**Matrice de confusion :**
<seq>
[[16685 2937]
[2981 10660]]
</seq>

Le modèle XGBoost a été optimisé sur cinq hyperparamètres principaux. La meilleure configuration a obtenu une précision de validation croisée de **82.63%**. Les observations principales sont :
- `colsample_bytree = 0.8` et `subsample = 0.8` ont fourni les meilleurs résultats d'échantillonnage
- `max_depth = 7` a offert une profondeur optimale pour la complexité du modèle
- `learning_rate = 0.2` s'est révélé efficace pour la vitesse de convergence
- `n_estimators = 300` a donné le meilleur nombre d'arbres

**Performance finale :**
- **Score entraînement** : 84.55%
- **Score test** : 82.21%
- Faible écart entre entraînement et test (2.34%), indiquant une bonne capacité de généralisation
- Temps de calcul très rapide à la fois pour l'entraînement (0.36s) et la prédiction (0.16s)



### 2.3 Analyse comparative des modèles _(Expérimentation 3)_

* Résultats des meilleurs modèles obtenus dans Expe 2

|  Evaluation en train | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            | 0.9372        | 0.7904   | 0.8455  |
|  Temps calcul        | 33.9610 sec   | 18.5088 sec | 0.3632 sec |
|  Matrice confusion   | [[74934 3556] [4799 49763]] | [[65888 12652] [15237 39325]] | [[68188 10302] [10259 44303]] |



|   Evaluation en test | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            | 0.8191        | 0.7820   | 0.8221  |
|  Temps calcul        | 4.3350 sec    | 4.4093 sec | 0.1570 sec |
|  Matrice confusion   | [[16954 2668] [3350 10291]] | [[16349 3273] [3978 9663]] | [[16685 2937] [2981 10660]] |


#### Analyse Comparative des Modèles

 1. **Performance Prédictive**
- **Meilleur score test** : XGBoost (82.21%) → Random Forest (81.91%) → Adaboost (78.20%)
- **Écart train/test** : Random Forest (11.81% d'écart) montre un surapprentissage significatif
- **Stabilité** : Adaboost a le plus faible écart (0.84%) mais score absolu plus bas

 2. **Temps de Calcul**
- **Plus rapide à l'entraînement** : XGBoost (0.36s) → Adaboost (18.51s) → Random Forest (33.96s)
- **Plus rapide en test** : XGBoost (0.16s) → Random Forest (4.34s) → Adaboost (4.41s)
- **XGBoost** est 94 fois plus rapide que Random Forest à l'entraînement

 3. **Analyse des Matrices de Confusion**
- **Random Forest** : Bon nombres mais significativement plus de FN (3350) que de FP (2668)
- **XGBoost** : Bon équilibre avec 2937 FP et 2981 FN (similaires)
- **Adaboost** : Plus d'erreurs (3273 FP, 3978 FN) → performance globale plus faible

 4. **Compromis Performance/Complexité**
- **XGBoost** : Meilleur compromis (performance élevée + rapidité)
- **Random Forest** : Performance d'entraînement excellente mais généralisation moins bonne
- **Adaboost** : Plus stable mais performance limitée


**XGBoost** est le modèle recommandé pour ce problème :
- Meilleure précision en test (82.21%)
- Temps de calcul nettement inférieur
- Bon équilibre entre performance et généralisation
- Faible écart entre entraînement et test



### 2.4 Inférence sur un autre jeu de données _(Expérimentation 4)_

Pour cette expérience, les modèles ont été entraînés sur les données de Californie et testés sur les jeux de données du Colorado et du Nevada.  
L'objectif est de mesurer la capacité de généralisation des modèles.

| Métrique    | RandomForest | AdaBoost   | XGBoost    |
|-------------|--------------|------------|------------|
| Accuracy_NV | 0.755216     | 0.717663   | 0.759852   |
| F1_NV       | 0.679612     | 0.639858   | 0.687198   |
| AUC_NV      | 0.860404     | 0.825411   | 0.865947   |
| Accuracy_CO | 0.778346     | 0.750240   | 0.778186   |
| F1_CO       | 0.760111     | 0.722400   | 0.760888   |
| AUC_CO      | 0.871310     | 0.835703   | 0.876914   |

Tableau X – Expérimentation (4) : Inférence inter-États (Nevada et Colorado)

**Métriques calculées :** 
- _Accuracy_ : C'est la proportion de bonnes prédictions par rapport à toutes les prédictions.
- _F1_ : C'est une mesure qui combine la précision et le rappel pour la classe positive.  
  (Ce score montre si le modèle fait peu d'erreurs quand il prédit un revenu élevé et s'il réussit à détecter la plupart des personnes ayant un revenu élevé.)
- _AUC (Area Under Curve)_ : C'est un nombre entre 0 et 1 qui montre si le modèle différencie bien les classes. (Plus le chiffre est proche de 1, mieux le modèle distingue les revenus faibles et élevés.)

**Remarques :**
- Les performances sont légèrement plus faibles que celles obtenues sur l’ensemble de Californie (train/test).
- XGBoost et RandomForest restent les modèles les plus performants, avec des Accuracy et F1 supérieures à AdaBoost sur les deux états.  
- La capacité de généralisation est meilleure pour XGBoost (AUC_CO la plus élevée : 0.876914).

### 2.5 Impact de la taille du jeu de données _(Expérimentation 5)_

Nous avons entraîné les modèles RandomForest, AdaBoost et XGBoost avec différentes proportions du jeu de données de Californie (de 10% à 100%).  
Nous avons mesuré l'évolution du F1-score en fonction du nombre d'échantillons.

<img width="571" height="433" alt="Capture d’écran 2025-12-15 à 18 33 36" src="https://github.com/user-attachments/assets/25b24212-83b9-49b2-b6b0-b29903c0ed4c" />

Tableau X – Expérimentation (5) : Impact de la taille du jeu d'entraînement

**Remarques :**
- Pour RandomForest et XGBoost, le F1-score augmente avec la taille du jeu de données. Plus on a de données, plus le modèle est performant.  
- AdaBoost montre une légère amélioration mais reste plus faible que les autres modèles. Il apprend donc moins bien avec un jeu de données plus grand.
- Le score F1 se stabilise à partir de 80-100% des données. Donc au-delà d'un certain point, ajouter plus de données n'améliore pas beaucoup la performance.  
- XGBoost atteint le score le plus élevé (0.7869) avec le jeu entier.

---

## 3. Explicabilité des prédictions

### 3.1 Classement des attributs

L’objectif de cette partie est d’identifier quels attributs ont le plus d’influence sur les prédictions du modèle.
Nous utilisons le modèle XGBoost, qui c'est révélé comme le plus performant dans la partie 2.

Pour cela, nous utilisons une méthode basée sur la permutation des valeurs : 
- On commence par mesurer la performance du modèle sur le jeu de test original.
- Pour chaque attribut :
  - On mélange aléatoirement les valeurs de cet attribut.
  - Les autres attributs restent inchangés.
  - On refait une prédiction avec le modèle sur ce jeu de test modifié.
  - On mesure la nouvelle performance.
- On calcule la différence de performance entre le jeu de test normal et le jeu perturbé.

Une forte baisse de performance indique que l’attribut est important.
À l’inverse, une faible baisse signifie que l’attribut a peu d’impact sur la prédiction.

<img width="878" height="487" alt="Capture d’écran 2025-12-17 à 10 57 29" src="https://github.com/user-attachments/assets/b8236ecf-6a3e-4e7c-a575-4f00ef8f3311" />

Tableau X – Expérimentation (6) : Top 15 - l'importance des attributs

**Remarques :** 
- L’importance a été calculée pour l’ensemble des attributs du jeu de données. Mais seuls les 15 attributs avec les plus d'impacts sont affichés, pour des raisons de lisibilité.
- Nous avons donc pu voir que les attributs qui sont le plus importants sont :
  - `WKHP` (le nombre d’heures travaillées par semaine)
  - `AGEP` (l'âge )
  - `4_Bachelor` (le niveau d’études : bac+3)
  - `5_Master_To_Professionnal` (le niveau d’études : bac+5)
 -  Les résultats montrent aussi que certains attributs n'ont quasiment pas d'impact.

### 3.2 Explications locales

### LIME 

#### Exemple d'utilisation

On va ici présenter un cas d'utilsation de LIME et voir comment analyser les données ainsi récoltées.

On a pour une première personne choisie au hasard dans le dataset ces résultats sur les 10 features les plus importantes :

Exemple 0 (index 8457) - Features les plus influentes :
POBP_460.0 <= 0.00: -0.309

OCCP_2755.0 <= 0.00: -0.293

POBP_515.0 <= 0.00: -0.292

POBP_459.0 <= 0.00: -0.281

POBP_72.0 <= 0.00: -0.272

OCCP_8930.0 <= 0.00: -0.271

OCCP_1106.0 <= 0.00: -0.270

OCCP_3321.0 <= 0.00: -0.223

OCCP_3725.0 <= 0.00: 0.104

POBP_369.0 <= 0.00: 0.103

qui se traduisent en image par :


IMG


Exemple 1 LIME : Les 10 features les plus importantes pour un cas particulier(négative en rouge et positive en vert)

La première ligne POBP_460.0 <= 0.00: -0.309 se traduit par, le fait de ne PAS être née en ZAMBIE (voir le document American Community Survey 
and Puerto Rico Community Survey 2018) est négatif(0.309) donc cela diminue la probabilité d'avoir > 50k$" pour le modèle.


 La seconde ligne OCCP_2755.0 <= 0.00: -0.293 traduit que le fait que le travail de la personne de soit pas " Disc jockeys, except radio  " est également très pénalisant.

 Enfin la ligne OCCP_3725.0 <= 0.00: 0.104 explique qu'il est positif que la personne n'ait pas comme travail "First-line supervisors of security workers"


#### Avantages et Limites

 **Intérêts :**

**Interprétabilité locale** - Permet de comprendre pourquoi une prédiction spécifique a été faite

**Accessible** - Visualisation simple (vert/rouge) compréhensible par tous

**Détection de biais** - Permet de voir si le modèle utilise des features pertinentes


 **Limites :**

**Instable** - Les résultats peuvent changer entre différentes exécutions

**Local seulement** - N'explique pas le comportement global du modèle

**Coûteux en calcul** - Lent sur de grands jeux de données

**Interprétabilité** - Peux être très dur à comprendre et flou. Il est par exemple difficile de comprendre pourquoi ne pas venir spécifiquement de Zambie a un tel impact négatif sur la prédiction.

### SHAP
On va maintenant voir 2 exemples de SHAP pour mieux comprendre son fonctionnement.

IMG

Exemple 1 SHAP : L'impact des 10 caractéristiques les plus importantes, d'une personne choisis au hasard sur ses chances de gagner >50k(résultat négatif)

Toutes les personnes dans le dataset partent avec une valeur initiale de -03966 ( on estime qu'il est plus probable qu'une personne "moyenne" ne gagne pas plus de 50 k)
Ensuite, SHAP évalue chaque caractéristique puis dit si elle est positive(sous entendu augmente la probabilté pour le modèle que la personne gagne + de 50 k) ou l'inverse.


Dans cet exemple on voit que son age et son sexe sont légèrement favorables, mais que son travail "Janitors and building cleaners"(-0.99) ainsi que son lieu de naissance "Mexico" (-0.56) joue fortement contre lui. Au final SHAP lui attribue une note de -2.36 ce qui équivaut à une probabilté de 11.7% de gagner + de 50K.





IMG2

Exemple 2 SHAP : L'impact des 10 caractéristiques  les plus importantes, d'une personne choisis au hasard sur ses chances de gagner >50k(résultat positif)

Dans cette exemple, on se rend compte que les 7 premières features(Age, sexe, ethnie,...) n'ont que très peu d'impact.
Ce qui va avoir un impact indéniable sera son nombre d'heures de travail(+1.22), son master (+1.17) et enfin son travail en lui même "Financial managers".

Tout cela pris en compte, le modèle estime qu'il y a 97% de chances que cette personne gagne plus de 50K.

Au final SHAP permet de comprendre de façon précise les caractéristiques qui impactent chaque individu. On comprend plus facilement la logique du modèle et ainsi ce qui compte vraiment pour gagner + de 50k.

#### Comparaison Lime et SHAP 

LIME permet de comprendre une décision précise du modèle en simplifiant son fonctionnement autour d'un cas particulier.Cela permete notamment de comprendre pourquoi une personne a été classée d'une certaine manière.

SHAP va mesurer la contribution exacte de chaque caractéristique. Il ne se contente pas d'analyser un cas isolé : il peut aussi donner une vision d'ensemble de l'importance des variables dans le modèle complet. Il permet de comprendre l'impact concret de chaque caractéristique de façon visuelle qui plus est.


#### Analyse summary-plot de SHAP

Le Summary plot de SHAP permet de voir quelles caractéristiques sont les plus importantes dans le modèle et dans quel sens elles sur influent la prédiction.

IMG


Il est complexe d'analyser dans son entièreté ce graphique mais voici 3 choses qui nous semblent importantes.

- On voit que pour les WKHP (working hours), il est  grandement valorisé d'en faire beaucoup, et que l'âge élevé d'une personne a tendance à accroître ses chances.
- Que visiblement le sexe masculin est toujours positif là où le sexe féminin lui est toujours négatif.

- Enfin qu'avoir un diplôme (Licence ou Master+) est tout le temps valorisé.

### 3.3 Explication contrefactuelle

L'objectif est d'arriver à faire basculer la prédiction du modèle en changeant simplement un attribut. Pour cela il nous a paru pertinent de prendre un faux positif.

L'individu 28818 est prédit comme  "gagnant + de 50k" alors que ce n'est pas le cas. Le fait qu'il travail 60 heures par semaine semble jouer un rôle important donc on à décider d'observer les prédictions du modèles en faisant juste varier son nombre d'heures travaillées entre 30 et 55.

IMG

On observe qu'en effet son nombre d'heures au travail a un très fort impact sur la prédiction et donc qu'il faudrait être en dessous de 40 heures pour que le modèle change d'avis.


## 4. Evaluation sur un nouvel échantillon

Nous avons testé notre meilleur modèle (XGBoost) sur un nouvel échantillon du jeu de données de Californie, mais qui n’avait pas été utilisé lors de l’entraînement.

### Préparation des données
Comme pour les autres jeux de données, nous avons effectué un prétraitement :

1. On garde les variables numériques pertinentes (AGEP et WKPH).
2. On fait des catégorie pour les niveaux d'étude.
3. On applique one-hot encoding pour le reste des attributs.

Après cette préparation, certaines colonnes du nouvel échantillon étaient absentes car certaines catégories n’étaient pas présentes dans l’échantillon testé. Nous avions rencontré ce même problème lors de l'utilisation du jeud e données du Nevada.

### Ajustement pour correspondre au modèle
Pour appliquer notre modèle, nous avons ensuite :

1. Ajouté les colonnes manquantes (initialisées à 0) correspondant aux catégories absentes.
2. Supprimé les colonnes supplémentaires présentes uniquement dans l’échantillon.
3. Réordonné les colonnes pour qu’elles correspondent exactement à celles attendues par le modèle.

### Résultat 

<img width="520" height="402" alt="Capture d’écran 2025-12-18 à 17 39 46" src="https://github.com/user-attachments/assets/26672715-58f6-46c6-a0f2-045e33c2da15" />

Tableau X – Evaluation d'un échantillon de Californie : Performance du modèle


Nous remarquons que le modèle ne se trompe pas beaucoup. Cela est cohérent car les données sont du même environnement que celles utilisées pour l'entrainement.
