import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=== SPLIT DU DATASET ORIGINAL 80/20 ===")
print(os.getcwd())
# 1. CHARGEMENT DES DONNÉES ORIGINALES
print("\n1. 📥 CHARGEMENT DES DONNÉES ORIGINALES...")

labels = pd.read_csv("resources/2-Dataset/alt_acsincome_ca_labels_85.csv")
features = pd.read_csv("resources/2-Dataset/alt_acsincome_ca_features_85.csv")

print(f"   • Features : {features.shape}")
print(f"   • Labels : {labels.shape}")

# 2. FUSIONNER FEATURES ET LABELS
print("\n2. 🔄 FUSION DES DONNÉES...")

dataset = features.copy()
dataset['INCOME_ABOVE_50K'] = labels.iloc[:, 0]

print(f"   • Dataset complet : {dataset.shape}")
print(f"   • Distribution de la target :")
print(dataset['INCOME_ABOVE_50K'].value_counts(normalize=True).round(3))

# 3. SPLIT 80/20
print("\n3. ✂️  SPLIT TRAIN/TEST 80/20...")

# Séparation features/target
X = dataset.drop('INCOME_ABOVE_50K', axis=1)
y = dataset['INCOME_ABOVE_50K']

# Split avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print(f"   • X_train : {X_train.shape} ({(X_train.shape[0]/len(X))*100:.1f}%)")
print(f"   • X_test  : {X_test.shape} ({(X_test.shape[0]/len(X))*100:.1f}%)")

# 4. CRÉATION DES DATASETS FINAUX
print("\n4. 🏗️  CRÉATION DES FICHIERS FINAUX...")

# Dataset d'entraînement complet
train_dataset = X_train.copy()
train_dataset['INCOME_ABOVE_50K'] = y_train.values

# Dataset de test complet  
test_dataset = X_test.copy()
test_dataset['INCOME_ABOVE_50K'] = y_test.values

print(f"   • Train dataset : {train_dataset.shape}")
print(f"   • Test dataset  : {test_dataset.shape}")

# 5. SAUVEGARDE
print("\n5. 💾 SAUVEGARDE...")

# Sauvegarder les datasets
train_dataset.to_csv("resources/2-Dataset/train_original_80.csv", index=False)
test_dataset.to_csv("resources/2-Dataset/test_original_20.csv", index=False)

# Sauvegarder aussi les features et labels séparés (optionnel)
#X_train.to_csv("../resources/2-Dataset/X_train_original.csv", index=False)
#X_test.to_csv("../resources/2-Dataset/X_test_original.csv", index=False)
#y_train.to_csv("../resources/2-Dataset/y_train_original.csv", index=False)
#y_test.to_csv("../resources/2-Dataset/y_test_original.csv", index=False)

print(f"   ✅ Fichiers sauvegardés :")
print(f"     - train_original_80.csv")
print(f"     - test_original_20.csv")
print(f"     - X_train_original.csv")
print(f"     - X_test_original.csv") 
print(f"     - y_train_original.csv")
print(f"     - y_test_original.csv")

# 6. VÉRIFICATION
print("\n6. ✅ VÉRIFICATION...")

print(f"   • Distribution dans le train : {y_train.mean():.3f}")
print(f"   • Distribution dans le test  : {y_test.mean():.3f}")
print(f"   • Écart : {abs(y_train.mean() - y_test.mean()):.4f}")

# Vérifier la taille des fichiers
import os
train_size = os.path.getsize("resources/2-Dataset/train_original_80.csv") / (1024*1024)
test_size = os.path.getsize("resources/2-Dataset/test_original_20.csv") / (1024*1024)

print(f"   • Taille train : {train_size:.1f} MB")
print(f"   • Taille test  : {test_size:.1f} MB")

print("\n🎉 SPLIT TERMINÉ AVEC SUCCÈSazea !")