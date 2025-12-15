import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import load

from src.best_models import get_models

train_CA = pd.read_csv("../resources/2-Dataset/dataset_train80_0_1_final.csv", low_memory=False)
test_NV = pd.read_csv("../resources/Complementary_data/dataset_test20_0_1_final_ne.csv", low_memory=False)
test_CO = pd.read_csv("../resources/Complementary_data/dataset_test20_0_1_final_co.csv", low_memory=False)

target_col = "INCOME_ABOVE_50K"

# Colonnes du train (sans la cible)
train_cols = [col for col in train_CA.columns if col != target_col]


def preprocess(X, expected_cols):
    # Remplir les valeurs manquantes
    X = X.fillna(0)

    # Aligner les colonnes avec celles du modèle
    missing_cols = [col for col in expected_cols if col not in X.columns]
    if missing_cols:
        X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing_cols)], axis=1)

    extra_cols = [col for col in X.columns if col not in expected_cols]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    # Réordonner
    X = X[expected_cols]

    # Convertir tout en float pour éviter les erreurs
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'bool':
            X[col] = X[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0)
    return X.astype(float)


X_test_NV = test_NV.drop(columns=[target_col])
y_test_NV = test_NV[target_col]
X_test_CO = test_CO.drop(columns=[target_col])
y_test_CO = test_CO[target_col]


models = get_models()

# Préparer les données de test alignées
X_test_NV_aligned = preprocess(X_test_NV.copy(), train_cols)
X_test_CO_aligned = preprocess(X_test_CO.copy(), train_cols)


results = []

for name, model in models.items():
    # Prédictions Nevada
    y_pred_NV = model.predict(X_test_NV_aligned)
    y_proba_NV = model.predict_proba(X_test_NV_aligned)[:, 1]

    # Prédictions Colorado
    y_pred_CO = model.predict(X_test_CO_aligned)
    y_proba_CO = model.predict_proba(X_test_CO_aligned)[:, 1]

    # Stocker les résultats
    results.append({
        "Model": name,
        "Accuracy_NV": accuracy_score(y_test_NV, y_pred_NV),
        "F1_NV": f1_score(y_test_NV, y_pred_NV),
        "AUC_NV": roc_auc_score(y_test_NV, y_proba_NV),
        "Accuracy_CO": accuracy_score(y_test_CO, y_pred_CO),
        "F1_CO": f1_score(y_test_CO, y_pred_CO),
        "AUC_CO": roc_auc_score(y_test_CO, y_proba_CO),
    })

results_df = pd.DataFrame(results)
print("\n=== Résultats Expe 4 – Inférence inter-États ===")
print(results_df)
