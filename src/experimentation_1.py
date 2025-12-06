import pandas as pd
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# =========================
# 1. Chargement des données
# =========================

df_train = pd.read_csv("../resources/2-Dataset/dataset_train80_0_1_final.csv")
df_test = pd.read_csv("../resources/2-Dataset/dataset_test20_0_1_final.csv")

print("Taille train :", df_train.shape)
print("Taille test  :", df_test.shape)

# =========================
# 2. Séparation X / y
# =========================

X_train = df_train.drop(columns=["INCOME_ABOVE_50K"])
y_train = df_train["INCOME_ABOVE_50K"]

X_test = df_test.drop(columns=["INCOME_ABOVE_50K"])
y_test = df_test["INCOME_ABOVE_50K"]

# =========================
# 3. Modèles par défaut
# =========================

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# =========================
# 4. EXPÉRIMENTATION 1
# =========================

resultats_train = {}
resultats_test = {}

for nom, model in models.items():
    print("\n==============================")
    print("MODÈLE :", nom)
    print("==============================")

    # ---- VALIDATION CROISÉE (TRAIN) ----
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    end = time.time()

    print("Accuracy (CV 5 folds) :", scores.mean())
    print("Temps de calcul (sec) :", end - start)

    # ---- ENTRAÎNEMENT FINAL ----
    model.fit(X_train, y_train)

    # ---- PRÉDICTIONS TRAIN ----
    y_train_pred = model.predict(X_train)

    print("\n--- RÉSULTATS TRAIN ---")
    print("Accuracy :", accuracy_score(y_train, y_train_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_train, y_train_pred))
    print("Classification report :\n", classification_report(y_train, y_train_pred))

    # ---- PRÉDICTIONS TEST ----
    y_test_pred = model.predict(X_test)

    print("\n--- RÉSULTATS TEST ---")
    print("Accuracy :", accuracy_score(y_test, y_test_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_test_pred))
    print("Classification report :\n", classification_report(y_test, y_test_pred))
