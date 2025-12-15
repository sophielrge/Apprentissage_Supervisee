import pandas as pd
import joblib
from xgboost import XGBClassifier

best_params = {
    "n_estimators": 300,
    "learning_rate": 0.2,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}



df_train = pd.read_csv("../../resources/2-Dataset/dataset_train80_0_1_final.csv")
X_train = df_train.drop("INCOME_ABOVE_50K", axis=1)
y_train = df_train["INCOME_ABOVE_50K"]


best_model = XGBClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)

print("\n==============================")
print("🏆 BEST XGBoost MODEL")
print("==============================")
print(best_model)
print("Meilleurs paramètres finaux :", best_params)

# Sauvegarde
joblib.dump(best_model, "../../resources/models/xg_boost.pkl")
print("\n💾 Modèle sauvegardé : xg_boost.pkl")