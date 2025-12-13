import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from best_models import get_models

test_CA = pd.read_csv("resources/2-Dataset/dataset_test20_0_1_final.csv")
target_col = "INCOME_ABOVE_50K"

X_test = test_CA.drop(columns=[target_col])
y_test = test_CA[target_col]

model = get_models()["RandomForest"]  # Mettre meilleur modèle

# Calcul de l'importance par permutation
perm_importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring='f1',
    n_repeats=10,
    random_state=42
)

# Récupération et tri des importances
importances = perm_importance.importances_mean
features = X_test.columns
sorted_idx = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(features)), importances[sorted_idx], align='center')
plt.xticks(range(len(features)), features[sorted_idx], rotation=90)
plt.ylabel("Importance (Permutation)")
plt.title("Permutation Feature Importance - Californie")
plt.tight_layout()
plt.show()

for idx in sorted_idx:
    print(f"{features[idx]} : {importances[idx]:.4f}")
