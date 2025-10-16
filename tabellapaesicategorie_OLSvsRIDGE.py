"""
grafico_OLS_vs_Ridge_dataset.py”
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")

df = pd.read_csv(csv_path)

df = df[[
    "price_usd",
    "product_condition",
    "product_category",
    "seller_country",
    "product_like_count",
    "seller_community_rank"
]].dropna()

df = df.query("price_usd > 0").copy()

df["log_price"]   = np.log10(df["price_usd"])
df["log_likes"]   = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)

min_count = 200
rare_countries = df["seller_country"].value_counts()[lambda x: x < min_count].index
df["seller_country"] = df["seller_country"].replace(rare_countries, "Other")

formula = """
log_price ~ C(product_condition)
         + C(product_category)
         + C(seller_country)
         + log_likes
         + rating_star
"""

model_ols = smf.ols(formula, data=df).fit(cov_type="HC3")
coef_ols = model_ols.params.rename("beta_ols")

coef_ols.index = (coef_ols.index
    .str.replace("C(product_condition)[T.", "product_condition_", regex=False)
    .str.replace("C(product_category)[T.", "product_category_", regex=False)
    .str.replace("C(seller_country)[T.", "seller_country_", regex=False)
    .str.replace("]", "", regex=False)
)

y = df["log_price"]
X = pd.get_dummies(
    df[["product_condition", "product_category", "seller_country", "log_likes", "rating_star"]],
    drop_first=True,
    dtype=float
)

alphas = np.logspace(-3, 3, 13)
ridge_pipe = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
)
ridge_pipe.fit(X, y)
coef_ridge = pd.Series(ridge_pipe.named_steps["ridgecv"].coef_, index=X.columns, name="beta_ridge")

df_coef = pd.concat([coef_ols, coef_ridge], axis=1, join="inner").dropna()

df_coef["panel"] = "Other"
df_coef.loc[df_coef.index.str.contains("seller_country", na=False), "panel"] = "Paesi"
df_coef.loc[df_coef.index.str.contains("product_category", na=False), "panel"] = "Macro-categorie"

df_coef["variabile"] = (
    df_coef.index.to_series()
      .str.replace("seller_country_", "Paese: ", regex=False)
      .str.replace("product_category_", "Categoria: ", regex=False)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=False)

for ax, panel in zip(axes, ["Paesi", "Macro-categorie"]):
    sub = df_coef[df_coef["panel"] == panel].sort_values("beta_ols", ascending=True)

    if sub.empty:
        ax.set_title(f"{panel} (vuoto)")
        continue

    y_pos = range(len(sub))
    ax.barh(y_pos, sub["beta_ols"], height=0.35, label="OLS", color="steelblue")
    ax.barh([p + 0.35 for p in y_pos], sub["beta_ridge"], height=0.35, label="Ridge (α=10)", color="orange")

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks([p + 0.2 for p in y_pos])
    ax.set_yticklabels(sub["variabile"])
    ax.set_title(f"{panel}")
    ax.legend()

fig.suptitle("β OLS vs β Ridge (α=10)", fontsize=14)
plt.tight_layout()
plt.savefig("OLS_vs_Ridge.png", dpi=300)
plt.show()
