import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
print(f"Carico il file da: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"⚠️  Il file non è stato trovato a: {csv_path}")


df = pd.read_csv(csv_path)
print("✓ CSV caricato, dimensioni:", df.shape)


COL_PRICE   = "price_usd"
COL_COUNTRY = "seller_country"


df_country = (
    df[[COL_COUNTRY, COL_PRICE]]
    .dropna(subset=[COL_COUNTRY, COL_PRICE])
    .query(f"{COL_PRICE} > 0")
    .copy()
)


stats_cty = (
    df_country.groupby(COL_COUNTRY)[COL_PRICE]
    .agg(
        n      ="size",
        mean   ="mean",
        median ="median",
        std    ="std"
    )
    .query("n >= 200")
    .sort_values("mean", ascending=False)
    .round(2)
)

stats_cty.to_csv("country_price_stats_mean.csv")
print("✓ Tabella Paesi salvata in country_price_stats_mean.csv")


topN = 10
top_mean = stats_cty.head(topN)

plt.figure(figsize=(11, 6))
sns.barplot(
    y=top_mean.index,
    x=top_mean["mean"],
    color="steelblue"
)
plt.xlabel("Prezzo medio [$]")
plt.ylabel("Paese venditore")
plt.title(f"Top {topN} Paesi per prezzo medio")
for idx, val in enumerate(top_mean["mean"]):
    plt.text(val + 5, idx, f"{val}", va="center")
plt.tight_layout()
plt.show()


top5 = top_mean.index.tolist()[:5]
plt.figure(figsize=(11, 5))
sns.boxplot(
    data=df_country[df_country[COL_COUNTRY].isin(top5)],
    x=COL_COUNTRY,
    y=np.log10(df_country[COL_PRICE]),
    order=top5,
    showfliers=False,
    palette="viridis"
)
plt.ylabel("log₁₀(Prezzo in USD)")
plt.xlabel("")
plt.title("Distribuzione prezzi (scala log) – Top 5 Paesi (media più alta)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=stats_cty,
    x="n",
    y="mean",
    alpha=.7
)
sns.scatterplot(
    data=top_mean,
    x="n",
    y="mean",
    color="crimson",
    s=80,
    label="Top-10 media"
)


for i, row in top_mean.iterrows():
    plt.text(
        row["n"], row["mean"] + 100,
        i,                            
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", color="black"
    )

plt.xscale("log")
plt.xlabel("Numero annunci (log-scale)")
plt.ylabel("Prezzo medio [$]")
plt.title("Prezzo medio per Paese vs dimensione campione")
plt.legend()
plt.tight_layout()
plt.show()
