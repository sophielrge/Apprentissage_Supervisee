import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "RandomForest": joblib.load(
            "../resources/models/random_forest.pkl", ),
        "AdaBoost": joblib.load("../resources/models/ada_boost.pkl", ),
        "XGBoost": joblib.load("../resources/models/xg_boost.pkl", ),
    }
    return models
