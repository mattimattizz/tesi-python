"""
diagnostics.py – Diagnostica e qualità dell’adattamento OLS
Tesi: Metodi Statistici per il Marketing (Mattia)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")

df = pd.read_csv(csv_path)

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

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)

min_count = 100
country_counts = df["seller_country"].value_counts()
rare_countries = country_counts[country_counts < min_count].index
df["seller_country"] = df["seller_country"].replace(rare_countries, "Other")

formula = """
log_price ~ log_likes + rating_star
           + C(product_condition)
           + C(product_category)
           + C(seller_country)
"""
model = smf.ols(formula, data=df).fit(cov_type="HC3")
print(model.summary())

df["fitted"] = model.fittedvalues
df["resid"] = model.resid

plt.figure(figsize=(7, 5))
sns.scatterplot(x="fitted", y="resid", data=df, alpha=0.3)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Valori stimati (log-prezzo)")
plt.ylabel("Residui (log-prezzo)")
plt.title("F4.13 – Residui vs valori stimati")
plt.tight_layout()
plt.savefig("F4_13_residui_vs_fit.png", dpi=300, bbox_inches="tight")
plt.show()

sm.qqplot(df["resid"], line="45", fit=True)
plt.title("F4.14 – QQ-plot dei residui (log-prezzi)")
plt.tight_layout()
plt.savefig("F4_14_QQ_residui.png", dpi=300, bbox_inches="tight")
plt.show()

influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

df["leverage"] = leverage
df["cooks_d"] = cooks_d

lev_thr = 2 * (model.df_model + 1) / len(df)
infl_thr = 4 / len(df)

high_lev = df[df["leverage"] > lev_thr]
high_infl = df[df["cooks_d"] > infl_thr]
print(f"Osservazioni con leverage > {lev_thr:.4f}: {len(high_lev)}")
print(f"Osservazioni con Cook’s D > {infl_thr:.4f}: {len(high_infl)}")

X = pd.get_dummies(
    df[["log_likes", "rating_star", "product_condition", "product_category", "seller_country"]],
    drop_first=True,
    dtype=float
)

vif_df = pd.DataFrame({
    "variabile": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values("VIF", ascending=False)

vif_df.to_csv("T4_5_VIF.csv", index=False)
print("\nTabella T4.5 – VIF (prime variabili):")
print(vif_df.head(15))
