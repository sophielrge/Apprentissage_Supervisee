
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

labels = pd.read_csv("../resources/2-Dataset/dataset_schl_recategorise.csv")
features = pd.read_csv("../resources/2-Dataset/alt_acsincome_ca_features_85.csv")

df = pd.concat([labels, features], axis=1)

df["PINCP"] = df["PINCP"].map({True: 1, False: 0})

print(df.head())
print("\nTaille totale du dataset :", df.shape)

X = df.drop(columns=["PINCP"])
y = df["PINCP"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTaille du train :", X_train.shape[0])
print("Taille du test :", X_test.shape[0])

models = {
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}

train_results = []

for name, model in models.items():
    print("\n==============================")
    print(f" ➤ Entraînement : {name}")
    print("==============================")
    
    # Temps d'entraînement
    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start

    # CV accuracy
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=5)
    cm_cv = confusion_matrix(y_train, y_pred_cv).T
    
    train_results.append([
        name,
        f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        round(train_time, 4),
        cm_cv
    ])
    
    print("Accuracy (CV=5) :", cv_scores.mean())
    print("STD :", cv_scores.std())
    print("Temps (sec) :", train_time)
    print("Matrice de confusion CV (TRANSPOSEE) :\n", cm_cv)

df_train = pd.DataFrame(train_results,
                        columns=["Méthode", "Accuracy CV", "Temps (sec.)", "Matrice de Confusion"])
df_train

test_results = []

for name, model in models.items():
    print("\n==============================")
    print(f" ➤ Test : {name}")
    print("==============================")
    
    # Prédiction
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    # Matrice de confusion TRANSPOSEE
    cm = confusion_matrix(y_test, y_pred).T
    
    test_results.append([
        name,
        acc,
        cm
    ])
    
    print("Accuracy :", acc)
    print("Classification report :\n", classification_report(y_test, y_pred))
    print("Matrice de confusion (TRANSPOSEE) :\n", cm)

# Tableau de test
df_test = pd.DataFrame(test_results, columns=["Méthode", "Accuracy", "Matrice de Confusion"])
df_test