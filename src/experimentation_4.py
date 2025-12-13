import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


train_CA = pd.read_csv("resources/2-Dataset/dataset_train80_0_1_final.csv")
test_NV = pd.read_csv("resources/Complementary_data/dataset_test20_0_1_final_ne.csv")
test_CO = pd.read_csv("resources/Complementary_data/dataset_test20_0_1_final_co.csv")

target_col = "INCOME_ABOVE_50K"


X_train = train_CA.drop(columns=[target_col])
y_train = train_CA[target_col]

X_test_NV = test_NV.drop(columns=[target_col])
y_test_NV = test_NV[target_col]

X_test_CO = test_CO.drop(columns=[target_col])
y_test_CO = test_CO[target_col]


models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    ),

    # A changer
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    # A changer
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
}



results = []

for name, model in models.items():

    # Entraînement sur la Californie
    model.fit(X_train, y_train)

    # ---------- Nevada ----------
    y_pred_NV = model.predict(X_test_NV)
    y_proba_NV = model.predict_proba(X_test_NV)[:, 1]

    # ---------- Colorado ----------
    y_pred_CO = model.predict(X_test_CO)
    y_proba_CO = model.predict_proba(X_test_CO)[:, 1]

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

