import pandas as pd
import os

print("=== CATÉGORISATION DE SCHL AVEC .0 ===")
print("=== SPLIT DU DATASET ORIGINAL 80/20 ===")

print(os.listdir("resources/2-Dataset"))

# 1. CHARGEMENT DU FICHIER BINARISÉ
print("\n1. 📥 CHARGEMENT DU FICHIER BINARISÉ...")
df = pd.read_csv("resources/2-Dataset/train_binarise_80.csv")
print(f"   • Dimensions : {df.shape}")

# 2. IDENTIFICATION DES COLONNES SCHL
print("\n2. 🔍 IDENTIFICATION DES COLONNES SCHL...")
schl_columns = [col for col in df.columns if 'SCHL' in col]
print("   • Toutes les colonnes SCHL trouvées :")
for col in schl_columns:
    print(f"     - {col}")

# 3. MAPPING DES CATÉGORIES AVEC .0
print("\n3. 🎓 CRÉATION DES CATÉGORIES...")

categories = {
    '1_NoSchool_to_9th': [
        'SCHL_1.0', 'SCHL_2.0', 'SCHL_3.0', 'SCHL_4.0',
        'SCHL_5.0', 'SCHL_6.0', 'SCHL_7.0', 'SCHL_8.0',
        'SCHL_9.0', 'SCHL_10.0', 'SCHL_11.0', 'SCHL_12.0'
    ],
    '2_10th_to_12th': ['SCHL_13.0', 'SCHL_14.0', 'SCHL_15.0'],
    '3_HighSchool_to_Associates': [
        'SCHL_16.0', 'SCHL_17.0', 'SCHL_18.0', 'SCHL_19.0', 'SCHL_20.0'
    ],
    '4_Bachelors': ['SCHL_21.0'],
    '5_Masters_to_Professional': ['SCHL_22.0', 'SCHL_23.0'],
    '6_Doctorate': ['SCHL_24.0']
}

# 4. CRÉATION DES NOUVELLES CATÉGORIES
print("\n4. 🔄 CRÉATION DES NOUVELLES COLONNES...")

for categorie, colonnes in categories.items():

    # Colonnes existantes dans le dataset
    colonnes_existantes = [col for col in colonnes if col in df.columns]

    if colonnes_existantes:
        df[categorie] = df[colonnes_existantes].max(axis=1)
        print(f"   ✅ {categorie} créée à partir de {len(colonnes_existantes)} colonnes")
    else:
        df[categorie] = 0
        print(f"   ❌ Aucune colonne trouvée pour {categorie}")

# 5. SUPPRESSION DES ANCIENNES COLONNES SCHL
print("\n5. 🗑️ SUPPRESSION DES ANCIENNES COLONNES SCHL...")

colonnes_a_supprimer = [
    col for col in df.columns
    if 'SCHL_' in col and col not in categories.keys()
]

df_final = df.drop(columns=colonnes_a_supprimer)

print(f"   • {len(colonnes_a_supprimer)} anciennes colonnes supprimées")
print(f"   • Nouvelles dimensions : {df_final.shape}")

# 6. VÉRIFICATION DES CATÉGORIES
print("\n6. ✅ VÉRIFICATION DES CATÉGORIES...")
print("   • Distribution des nouvelles catégories :")

for categorie in categories.keys():
    if categorie in df_final.columns:
        count = df_final[categorie].sum()
        print(f"     - {categorie} : {count} individus")

# Vérification des chevauchements
print("\n   • Vérification des chevauchements :")

for i in range(len(df_final)):
    row = df_final.iloc[i]
    categories_actives = [
        cat for cat in categories.keys()
        if cat in df_final.columns and row[cat] == 1
    ]
    if len(categories_actives) > 1:
        print(f"     ⚠️ Individu {i} dans plusieurs catégories : {categories_actives}")

# 7. SAUVEGARDE
print("\n7. 💾 SAUVEGARDE...")

chemin_sauvegarde = "resources/2-Dataset/dataset_train80_schl_recategorise_train.csv"
df_final.to_csv(chemin_sauvegarde, index=False)

print(f"   ✅ Fichier sauvegardé : {chemin_sauvegarde}")

# 8. RAPPORT FINAL
print("\n" + "="*60)
print("📊 RAPPORT FINAL")
print("="*60)

print(f"• Fichier d'entrée  : {df.shape}")
print(f"• Fichier de sortie : {df_final.shape}")
print(f"• Colonnes supprimées : {len(colonnes_a_supprimer)}")
print("• Nouvelles catégories créées :")

nouvelles_categories = [
    cat for cat in categories.keys()
    if cat in df_final.columns
]

for cat in nouvelles_categories:
    count = df_final[cat].sum()
    percentage = (count / len(df_final)) * 100
    print(f"  - {cat} : {count} individus ({percentage:.1f}%)")

print("="*60)

# Aperçu des données
print("\n🔍 APERÇU DES NOUVELLES CATÉGORIES :")
print(df_final[nouvelles_categories].head())
