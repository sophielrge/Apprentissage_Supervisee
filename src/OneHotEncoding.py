import pandas as pd

# =============================================================================
# PROGRAMME SIMPLE DE BINARISATION AVEC ONE-HOT ENCODING
# =============================================================================

print("=== DÉBUT DU PROGRAMME ===")

# 1. CHARGEMENT DES DONNÉES
print("\n1. 📥 CHARGEMENT DES DONNÉES...")
labels = pd.read_csv("resources/2-Dataset/alt_acsincome_ca_labels_85.csv")
features = pd.read_csv("resources/2-Dataset/alt_acsincome_ca_features_85.csv")

print(f"   • Features : {features.shape}")
print(f"   • Labels : {labels.shape}")

# 2. IDENTIFICATION DES COLONNES
print("\n2. 🔍 IDENTIFICATION DES COLONNES...")

# Colonnes à NE PAS binariser (garder en numérique)
colonnes_a_garder = ['AGEP', 'WKHP']  # Âge et heures travaillées

# Colonnes à binariser (tout le reste)
colonnes_a_binariser = [col for col in features.columns if col not in colonnes_a_garder]

print(f"   • Colonnes GARDÉES (numériques) : {colonnes_a_garder}")
print(f"   • Colonnes BINARISÉES (catégorielles) : {colonnes_a_binariser}")

# 3. ONE-HOT ENCODING
print("\n3. 🔄 APPLICATION ONE-HOT ENCODING...")

# Créer une copie et appliquer One-Hot Encoding
features_binarisees = features.copy()
features_binarisees = pd.get_dummies(features_binarisees, 
                                   columns=colonnes_a_binariser, 
                                   drop_first=True)

print(f"   • Dimensions avant : {features.shape}")
print(f"   • Dimensions après : {features_binarisees.shape}")
print(f"   • Nombre de colonnes ajoutées : {features_binarisees.shape[1] - features.shape[1]}")

# 4. AJOUT DE LA TARGET (VARIABLE À PRÉDIRE)
print("\n4. 🎯 AJOUT DE LA TARGET...")

# La target est la première colonne du fichier labels
target_column = labels.columns[0]
features_binarisees['INCOME_ABOVE_50K'] = labels[target_column]

print(f"   • Target ajoutée : '{target_column}'")
print(f"   • Dimensions finales : {features_binarisees.shape}")

# 5. SAUVEGARDE
print("\n5. 💾 SAUVEGARDE DU NOUVEAU CSV...")

chemin_sauvegarde = "resources/2-Dataset/dataset_final_binarise.csv"
features_binarisees.to_csv(chemin_sauvegarde, index=False)

print(f"   ✅ Fichier sauvegardé : {chemin_sauvegarde}")

# 6. VÉRIFICATION FINALE
print("\n6. ✅ VÉRIFICATION FINALE...")

# Aperçu des types de données
print(f"   • Types de données finaux :")
print(f"     - AGEP : {features_binarisees['AGEP'].dtype} (conservé numérique)")
print(f"     - WKHP : {features_binarisees['WKHP'].dtype} (conservé numérique)")

# Distribution de la target
distribution = features_binarisees['INCOME_ABOVE_50K'].value_counts()
print(f"   • Distribution de la target :")
print(f"     - 0 (revenu ≤50K) : {distribution[0]} échantillons")
print(f"     - 1 (revenu >50K) : {distribution[1]} échantillons")

# Aperçu des nouvelles colonnes
nouvelles_colonnes = [col for col in features_binarisees.columns 
                      if col not in ['AGEP', 'WKHP', 'INCOME_ABOVE_50K']]
print(f"   • Exemples de nouvelles colonnes créées :")
for col in nouvelles_colonnes[:5]:  # Affiche les 5 premières
    print(f"     - {col}")

print(f"\n   • Nombre total de colonnes : {len(features_binarisees.columns)}")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "="*60)
print("🎉 PROGRAMME TERMINÉ AVEC SUCCÈS !")
print("="*60)
print(f"📊 RÉSULTAT :")
print(f"   • Fichier d'entrée  : {features.shape} → {labels.shape}")
print(f"   • Fichier de sortie : {features_binarisees.shape}")
print(f"   • Colonnes numériques conservées : 2 (AGEP, WKHP)")
print(f"   • Colonnes binaires créées : {len(nouvelles_colonnes)}")
print(f"   • Fichier sauvegardé : dataset_final_binarise.csv")
print("="*60)

# Aperçu du résultat (optionnel)
print("\n🔍 APERÇU DES PREMIÈRES LIGNES :")
print(features_binarisees.head(3))