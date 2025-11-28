import pandas as pd

# Charger le CSV
df = pd.read_csv("resources/2-Dataset/dataset_train80_schl_recategorise_train.csv")

# Remplacer True/False par 1/0
df = df.replace({True: 1, False: 0})

# Sauvegarder le nouveau CSV
df.to_csv("resources/2-Dataset/dataset_train80_0_1_final.csv", index=False)

print("✅ Terminé ! True/False transformés en 1/0")