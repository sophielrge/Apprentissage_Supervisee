import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# ① Charger les données  (à adapter selon ton dataset)
# ---------------------------------------------------------

# Exemple à adapter :
# df = pd.read_csv("ton_fichier.csv")
# X = df.drop("label", axis=1)
# y = df["label"]

# ----- À REMPLACER PAR TES DONNÉES -----
X = ...
y = ...
# ----------------------------------------

# ---------------------------------------------------------
# ② Découpage train / test
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Taille Train :", X_train.shape)
print("Taille Test  :", X_test.shape)

# ---------------------------------------------------------
# Fonction utilitaire : exécuter un gridsearch proprement
# ---------------------------------------------------------
def run_gridsearch(model, param_grid, name, folds=5):
    print(f"\n===== {name} =====")
    gs = GridSearchCV(model, param_grid, cv=folds, n_jobs=-1)

    start = time.time()
    gs.fit(X_train, y_train)
    end = time.time()

    train_acc = gs.score(X_train, y_train)
    test_acc = gs.score(X_test, y_test)
    cpu_time = end - start

    print("Meilleurs hyperparams :", gs.best_params_)
    print("Train Acc :", train_acc)
    print("Test Acc  :", test_acc)
    print("CPU Time  :", round(cpu_time, 3), "s")

    return {
        "Modèle": name,
        "Best Params": gs.best_params_,
        "Train Acc": train_acc,
        "Test Acc": test_acc,
        "CPU Time": cpu_time
    }

# ---------------------------------------------------------
# ③ Définition des hyperparamètres pour chaque modèle
# ---------------------------------------------------------

params_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

params_svm = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "poly"],
    "gamma": ["scale", "auto"],
    "degree": [2, 3]
}

params_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
}

# ---------------------------------------------------------
# ④ Lancer les 3 GridSearchCV
# ---------------------------------------------------------

results = []

results.append(
    run_gridsearch(KNeighborsClassifier(), params_knn, "KNN")
)

results.append(
    run_gridsearch(SVC(), params_svm, "SVM")
)

results.append(
    run_gridsearch(RandomForestClassifier(), params_rf, "Random Forest")
)

# ---------------------------------------------------------
# ⑤ Résultats sous forme de tableau
# ---------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n===== Résultats Finaux =====")
print(df_results)
