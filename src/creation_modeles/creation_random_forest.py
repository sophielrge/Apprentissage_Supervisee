import pandas as pd
from sklearn.ensemble import RandomForestClassifier

best_params = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": "log2"
}



df_train = pd.read_csv("../../resources/2-Dataset/dataset_train80_0_1_final.csv")
X_train = df_train.drop("INCOME_ABOVE_50K", axis=1)
y_train = df_train["INCOME_ABOVE_50K"]


best_rf_model = RandomForestClassifier(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

print("\n==============================")
print("🏆 BEST RANDOM FOREST MODEL")
print("==============================")
print(best_rf_model)
print("Meilleurs paramètres finaux :", best_params)

import joblib
joblib.dump(best_rf_model, "../../resources/models/random_forest.pkl")
print("\n💾 Modèle sauvegardé : best_random_forest_sequential.pkl")
