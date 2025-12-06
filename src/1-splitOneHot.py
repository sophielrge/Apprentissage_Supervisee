import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=== DÉBUT DU PROGRAMME ===")
print("Répertoire courant :", os.getcwd())

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n1. 📥 CHARGEMENT DES DONNÉES ORIGINALES...")

labels = pd.read_csv("resources/Complementary_data/acsincome_ne_label.csv")
features = pd.read_csv("resources/Complementary_data/acsincome_ne_allfeatures.csv")

print(f"   • Features : {features.shape}")
print(f"   • Labels   : {labels.shape}")

# Fusion pour split
dataset = features.copy()
dataset["INCOME_ABOVE_50K"] = labels.iloc[:, 0]

print(f"   • Dataset complet : {dataset.shape}")

# =============================================================================
# 2. SPLIT 80/20
# =============================================================================
print("\n2. ✂️ SPLIT TRAIN/TEST 80/20...")

X = dataset.drop("INCOME_ABOVE_50K", axis=1)
y = dataset["INCOME_ABOVE_50K"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

print(f"   • X_train : {X_train.shape}")
print(f"   • X_test  : {X_test.shape}")

# Reconstruction datasets
train_dataset = X_train.copy()
train_dataset["INCOME_ABOVE_50K"] = y_train.values

test_dataset = X_test.copy()
test_dataset["INCOME_ABOVE_50K"] = y_test.values

# =============================================================================
# 3. BINARISATION / ONE-HOT ENCODING (APPLIQUÉE SUR TRAIN SEULEMENT)
# =============================================================================
print("\n3. 🔄 BINARISATION AVEC ONE-HOT ENCODING...")

colonnes_a_garder = ['AGEP', 'WKHP']
colonnes_a_binariser = [col for col in X_train.columns if col not in colonnes_a_garder]

print(f"   • Colonnes numériques conservées : {colonnes_a_garder}")
print(f"   • Colonnes binarisées : {len(colonnes_a_binariser)}")

# Binarisation TRAIN
train_encoded = pd.get_dummies(
    train_dataset,
    columns=colonnes_a_binariser,
    drop_first=True
)

# Binarisation TEST (en réutilisant EXACTEMENT les colonnes du train)
test_encoded = pd.get_dummies(
    test_dataset,
    columns=colonnes_a_binariser,
    drop_first=True
)

# Réaligner les colonnes pour éviter les erreurs
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

print(f"   • Train avant : {train_dataset.shape}")
print(f"   • Train après : {train_encoded.shape}")
print(f"   • Test après  : {test_encoded.shape}")

# =============================================================================
# 4. SAUVEGARDE
# =============================================================================
print("\n4. 💾 SAUVEGARDE DES FICHIERS...")

train_encoded.to_csv("resources/Complementary_data/train_binarise_80_ne.csv", index=False)
test_encoded.to_csv("resources/Complementary_data/test_binarise_20_ne.csv", index=False)

print("   ✅ train_binarise_80.csv")
print("   ✅ test_binarise_20.csv")

# =============================================================================
# 5. VÉRIFICATIONS
# =============================================================================
print("\n5. 🔍 VÉRIFICATIONS...")

print(f"   • Distribution train : {y_train.mean():.3f}")
print(f"   • Distribution test  : {y_test.mean():.3f}")

print("\nExemple colonnes encodées :")
print(list(train_encoded.columns[:10]))

print("\n🎉 PROGRAMME TERMINÉ AVEC SUCCÈS !")