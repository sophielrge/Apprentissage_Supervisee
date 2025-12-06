import pandas as pd

# Charger le CSV
df = pd.read_csv("resources/Complementary_data/dataset_test20_schl_recategorise_test_co.csv")

# Remplacer True/False par 1/0
df = df.replace({True: 1, False: 0})

# Sauvegarder le nouveau CSV
df.to_csv("resources/Complementary_data/dataset_test20_0_1_final_co.csv", index=False)

print("✅ Terminé ! True/False transformés en 1/0")