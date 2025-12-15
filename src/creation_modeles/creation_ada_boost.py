import pandas as pd
import joblib
from sklearn.ensemble import AdaBoostClassifier

best_params = {
    "n_estimators": 100,
    "learning_rate": 1.0
}



df_train = pd.read_csv("../../resources/2-Dataset/dataset_train80_0_1_final.csv")
X_train = df_train.drop("INCOME_ABOVE_50K", axis=1)
y_train = df_train["INCOME_ABOVE_50K"]


best_ada_model = AdaBoostClassifier(random_state=42, **best_params)
best_ada_model.fit(X_train, y_train)

print("\n==============================")
print("🏆 BEST ADABOOST MODEL")
print("==============================")
print(best_ada_model)
print("Meilleurs paramètres finaux :", best_params)

# Sauvegarde
joblib.dump(best_ada_model, "../../resources/models/ada_boost.pkl")
print("\n💾 Modèle sauvegardé : ada_boost.pkl")