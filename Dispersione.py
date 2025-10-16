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


COL_PRICE = "price_usd"
COL_CAT   = "product_category"


df_cat = (
    df[[COL_CAT, COL_PRICE]]
    .dropna(subset=[COL_CAT, COL_PRICE])
    .query(f"{COL_PRICE} > 0")
    .copy()
)


def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

def cv(x):
    return x.std() / x.mean() if x.mean() > 0 else np.nan


stats = (
    df_cat.groupby(COL_CAT)[COL_PRICE]
    .agg(
        n     ="size",
        mean  ="mean",
        std   ="std",
        iqr   =iqr,
        cv    =cv
    )
    .query("n >= 100")           # soglia minima di osservazioni
    .sort_values("iqr", ascending=False)
    .round(2)
)


stats.to_csv("category_dispersion.csv")
print("✓ Tabella completa salvata in category_dispersion.csv")


topN = 10
top_iqr = stats.head(topN)

plt.figure(figsize=(10, 6))
sns.barplot(
    y=top_iqr.index,
    x=top_iqr["iqr"],
    color="steelblue"
)
plt.xlabel("Intervallo Interquartile (IQR) del prezzo [$]")
plt.ylabel("Categoria prodotto")
plt.title(f"Top {topN} categorie per dispersione di prezzo (IQR)")
# etichette a destra delle barre
for idx, val in enumerate(top_iqr["iqr"]):
    plt.text(val + 5, idx, f"{val}", va="center")
plt.tight_layout()
plt.show()


top5 = top_iqr.index.tolist()[:5]
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_cat[df_cat[COL_CAT].isin(top5)],
    x=COL_CAT,
    y=np.log10(df_cat[COL_PRICE]),      
    order=top5,
    showfliers=False
)
plt.ylabel("log₁₀(Prezzo in USD)")
plt.xlabel("")
plt.title("Distribuzione prezzi (scala log) – Top 5 categorie più disperse")
plt.tight_layout()
plt.show()
