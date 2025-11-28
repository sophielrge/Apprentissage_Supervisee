import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

labels = pd.read_csv("alt_acsincome_ca_labels_85.csv")
features = pd.read_csv("alt_acsincome_ca_features_85.csv")

df = pd.concat([labels, features], axis=1)
df["PINCP"] = df["PINCP"].map({True: 1, False: 0})

prob_by_sex = df.groupby("SEX")["PINCP"].mean()
sex = np.arange(1, 3)

prob_by_age = df.groupby("AGEP")["PINCP"].mean()
ages = np.arange(0, 100)

plt.figure(figsize=(12, 6))
sns.lineplot(x=ages, y=prob_by_age.reindex(ages, fill_value=0))
plt.title("Probability of 50k + Income by Age")
plt.xlabel("Age")
plt.ylabel("Probability of Income")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=sex, y=prob_by_sex.reindex(sex, fill_value=0))
plt.title("Probability of 50k + Income by Sex")
plt.xticks(ticks=[0, 1], labels=["Male", "Female"])
plt.xlabel("Sex")
plt.ylabel("Probability of Income")
plt.grid()
plt.show()
