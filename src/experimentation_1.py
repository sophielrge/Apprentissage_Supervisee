{
 "cells": [
  {
   "cell_type": "code",
   "id": "88ad3d7c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-11-28T15:45:33.628068Z",
     "start_time": "2025-11-28T15:45:20.796335Z"
    }
   },
   "source": [
    "import pandas as pd\n",
    "from time import time\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, cross_val_score, cross_val_predict\n",
    "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "# -----------------------------\n",
    "# 1) Chargement du dataset\n",
    "# -----------------------------\n",
    "df_train = pd.read_csv('cdv_train.csv')\n",
    "df_test = pd.read_csv('cdv_test.csv')\n",
    "\n",
    "X_train = df_train.drop(columns=['PINCP'])\n",
    "y_train = df_train['PINCP']\n",
    "X_test = df_test.drop(columns=['PINCP'])\n",
    "y_test = df_test['PINCP']\n",
    "\n",
    "print(\"Taille train:\", X_train.shape[0], \"| Taille test:\", X_test.shape[0])\n",
    "\n",
    "# -----------------------------\n",
    "# 2) Modèles par défaut\n",
    "# -----------------------------\n",
    "models = {\n",
    "    'RandomForest': RandomForestClassifier(),\n",
    "    'AdaBoost': AdaBoostClassifier(),\n",
    "    'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)\n",
    "}\n",
    "\n",
    "train_results = []\n",
    "test_results = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f'\\n===== Exp1 : {name} =====')\n",
    "    start = time()\n",
    "    model.fit(X_train, y_train)\n",
    "    train_time = time() - start\n",
    "    \n",
    "    # Cross-validation\n",
    "    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')\n",
    "    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=5)\n",
    "    cm_cv = confusion_matrix(y_train, y_pred_cv).T  # matrice transposée\n",
    "    \n",
    "    train_results.append([name, f'{cv_scores.mean():.4f} ± {cv_scores.std():.4f}', round(train_time,4), cm_cv])\n",
    "    \n",
    "    # Test\n",
    "    y_pred_test = model.predict(X_test)\n",
    "    acc_test = accuracy_score(y_test, y_pred_test)\n",
    "    cm_test = confusion_matrix(y_test, y_pred_test).T\n",
    "    test_results.append([name, acc_test, cm_test])\n",
    "    \n",
    "    print('Train CV Accuracy:', cv_scores.mean())\n",
    "    print('Test Accuracy:', acc_test)\n",
    "\n",
    "df_train_exp1 = pd.DataFrame(train_results, columns=['Méthode','Accuracy CV','Temps (sec.)','Matrice Confusion'])\n",
    "df_test_exp1 = pd.DataFrame(test_results, columns=['Méthode','Accuracy','Matrice Confusion'])\n",
    "\n",
    "print('\\n==== Tableau Exp1 (Train) ====')\n",
    "print(df_train_exp1)\n",
    "print('\\n==== Tableau Exp1 (Test) ====')\n",
    "print(df_test_exp1)\n"
   ],
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'sklearn'",
     "output_type": "error",
     "traceback": [
      "\u001B[31m---------------------------------------------------------------------------\u001B[39m",
      "\u001B[31mModuleNotFoundError\u001B[39m                       Traceback (most recent call last)",
      "\u001B[36mCell\u001B[39m\u001B[36m \u001B[39m\u001B[32mIn[2]\u001B[39m\u001B[32m, line 3\u001B[39m\n\u001B[32m      1\u001B[39m \u001B[38;5;28;01mimport\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01mpandas\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[38;5;28;01mas\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01mpd\u001B[39;00m\n\u001B[32m      2\u001B[39m \u001B[38;5;28;01mfrom\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01mtime\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[38;5;28;01mimport\u001B[39;00m time\n\u001B[32m----> \u001B[39m\u001B[32m3\u001B[39m \u001B[38;5;28;01mfrom\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01msklearn\u001B[39;00m\u001B[34;01m.\u001B[39;00m\u001B[34;01mmetrics\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[38;5;28;01mimport\u001B[39;00m accuracy_score, confusion_matrix, cross_val_score, cross_val_predict\n\u001B[32m      4\u001B[39m \u001B[38;5;28;01mfrom\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01msklearn\u001B[39;00m\u001B[34;01m.\u001B[39;00m\u001B[34;01mensemble\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[38;5;28;01mimport\u001B[39;00m RandomForestClassifier, AdaBoostClassifier\n\u001B[32m      5\u001B[39m \u001B[38;5;28;01mfrom\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[34;01mxgboost\u001B[39;00m\u001B[38;5;250m \u001B[39m\u001B[38;5;28;01mimport\u001B[39;00m XGBClassifier\n",
      "\u001B[31mModuleNotFoundError\u001B[39m: No module named 'sklearn'"
     ]
    }
   ],
   "execution_count": 2
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
