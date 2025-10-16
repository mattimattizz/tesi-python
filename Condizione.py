import pandas as pd
import os

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
print(f"Carico il file da: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"⚠️  Il file non è stato trovato a: {csv_path}")

df = pd.read_csv(csv_path)
print("✓ CSV caricato, dimensioni:", df.shape)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cols_needed = ["product_condition", "price_usd"]
missing_cols = set(cols_needed) - set(df.columns)
if missing_cols:
    raise KeyError(f"⚠️  Colonne mancanti nel CSV: {missing_cols}")
df = df[cols_needed]

df = df[df["price_usd"] > 0]


condition_order = [
    "Never worn, with tag",
    "Never worn",
    "Very good condition",
    "Good condition",
    "Fair condition",
]
df = df[df["product_condition"].isin(condition_order)].copy()
df["product_condition"] = pd.Categorical(
    df["product_condition"], categories=condition_order, ordered=True
)

print("✓ Dopo il filtro:", df.shape)


df["log_price"] = np.log10(df["price_usd"])

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="product_condition",
    y="log_price",
    order=condition_order,
    showfliers=False,
    width=0.6,
)
sns.stripplot(
    data=df.sample(min(5000, len(df)), random_state=0),  # sotto-campione per chiarezza
    x="product_condition",
    y="log_price",
    order=condition_order,
    color="black",
    size=2,
    jitter=0.25,
    alpha=0.15,
)

plt.xlabel("Condizione del capo")
plt.ylabel("log\u2081\u2080(Prezzo in USD)")
plt.title("Distribuzione del log\u2081\u2080(prezzo) per condizione")
plt.tight_layout()

plt.show()

