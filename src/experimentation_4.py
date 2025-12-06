
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Remplacer les hyperparamètres par tes valeurs optimales
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
}

results = []

for name, model in models.items():
    # Entraînement sur Californie
    model.fit(X_train, y_train)

    # Prédiction sur Nevada
    y_pred_NV = model.predict(X_test_NV)
    mse_NV = mean_squared_error(y_test_NV, y_pred_NV)
    r2_NV = r2_score(y_test_NV, y_pred_NV)

    # Prédiction sur Colorado
    y_pred_CO = model.predict(X_test_CO)
    mse_CO = mean_squared_error(y_test_CO, y_pred_CO)
    r2_CO = r2_score(y_test_CO, y_pred_CO)

    results.append({
        "Model": name,
        "MSE_NV": mse_NV,
        "R2_NV": r2_NV,
        "MSE_CO": mse_CO,
        "R2_CO": r2_CO
    })

results_df = pd.DataFrame(results)
print(results_df)
