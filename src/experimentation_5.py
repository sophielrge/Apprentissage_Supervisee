import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from best_models import get_models

from xgboost import XGBClassifier

data = pd.read_csv("../resources/2-Dataset/dataset_train80_0_1_final.csv")

target_col = "INCOME_ABOVE_50K"
X = data.drop(columns=[target_col])
y = data[target_col]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# A modifier avec les valeuts d'Elian
models = get_models()

train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, None]

results = []

for model_name, model in models.items():
    for size in train_sizes:
        if size is None:
            X_sub = X_train_full
            y_sub = y_train_full
            size_percent = 100
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_full,
                y_train_full,
                train_size=size,
                random_state=42,
                stratify=y_train_full
            )
            size_percent = int(size * 100)

        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": model_name,
            "Train_size_%": size_percent,
            "Nb_samples": len(X_sub),
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8, 6))
for model_name in results_df["Model"].unique():
    subset = results_df[results_df["Model"] == model_name]
    plt.plot(subset["Nb_samples"], subset["F1-score"], marker='o', label=model_name)
plt.xlabel("Nombre d'échantillons d'entraînement")
plt.ylabel("F1-score")
plt.title("Impact de la taille du jeu d'entraînement")
plt.legend()
plt.grid(True)
plt.show()
