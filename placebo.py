"""
placebo_APAC.py – Placebo test per il cluster APAC (Japan, China, Hong Kong)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")

df = pd.read_csv(csv_path, low_memory=False)
keep_cols = [
    "price_usd",
    "product_condition",
    "product_category",
    "seller_country",
    "product_like_count",
    "seller_community_rank",
]
df = (
    df[keep_cols]
    .dropna(subset=keep_cols)
    .query("price_usd > 0")
    .copy()
)

df = df.sample(frac=0.10, random_state=42)

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)

min_n = 50
country_counts = df["seller_country"].value_counts()
rare = country_counts[country_counts < min_n].index
df["seller_country"] = df["seller_country"].replace(rare, "Other")

formula = """
log_price ~ log_likes + rating_star
           + C(product_condition)
           + C(product_category)
           + C(seller_country)
"""
model = smf.ols(formula, data=df).fit()

keys = {
    "Japan": "C(seller_country)[T.Japan]",
    "China": "C(seller_country)[T.China]",
    "Hong Kong": "C(seller_country)[T.Hong Kong]",
}
real_betas = {k: model.params.get(v, np.nan) for k, v in keys.items()}
for k, b in real_betas.items():
    print(f"β reale ({k}) = {b:.4f}")

N_REPS = 50
rng = np.random.default_rng(42)
placebo_betas = {k: [] for k in keys}

for i in range(N_REPS):
    df_rand = df.copy()
    df_rand["seller_country"] = rng.permutation(df_rand["seller_country"].values)
    try:
        m_rand = smf.ols(formula, data=df_rand).fit()
        for k, v in keys.items():
            if v in m_rand.params:
                placebo_betas[k].append(m_rand.params[v])
    except Exception as e:
        print(f"Errore replica {i}: {e}")

p99 = {k: np.percentile(v, 99) for k, v in placebo_betas.items() if len(v) > 0}
flat_placebo = np.concatenate([v for v in placebo_betas.values() if len(v) > 0])

plt.figure(figsize=(8, 4))
sns.histplot(flat_placebo, bins=25, color="lightgray", edgecolor="black")
plt.xlabel("β stimato (Placebo)")
plt.ylabel("Frequenza")
plt.title("F4.15 – Distribuzione placebo β_Paese (APAC)")

plt.axvline(real_betas["Japan"], color="red", linestyle="--", linewidth=2,
            label=f"Japan = {real_betas['Japan']:.3f}")
plt.axvline(real_betas["China"], color="green", linestyle="--", linewidth=2,
            label=f"China = {real_betas['China']:.3f}")
plt.axvline(real_betas["Hong Kong"], color="blue", linestyle="--", linewidth=2,
            label=f"Hong Kong = {real_betas['Hong Kong']:.3f}")

p99_global = np.percentile(flat_placebo, 99)
plt.axvline(p99_global, color="black", linestyle=":", linewidth=1.5,
            label=f"99° percentile placebo = {p99_global:.3f}")

plt.legend()
plt.tight_layout()
plt.savefig("F4_15_placebo_APAC.png", dpi=300, bbox_inches="tight")
plt.show()
