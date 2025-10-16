
"""
price_model.py  –  stima del prezzo di listino (scala log) nel marketplace
Mattia – Tesi Metodi Statistici per il Marketing
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")


df = pd.read_csv(csv_path)
print("✓ CSV caricato, dimensioni:", df.shape)


base_vars = [
    "price_usd",
    "product_condition",
    "product_category",
    "seller_country",
    "product_like_count",
    "seller_community_rank",
]


if "seller_num_items_sold" in df.columns:
    exp_var = "seller_num_items_sold"
elif "seller_num_items_listed" in df.columns:
    exp_var = "seller_num_items_listed"
else:
    exp_var = None

vars_keep = base_vars + ([exp_var] if exp_var else [])
df = (
    df[vars_keep]
    .dropna(subset=base_vars)
    .query("price_usd > 0")
    .copy()
)


min_count = 100
country_counts = df["seller_country"].value_counts()
rare = country_counts[country_counts < min_count].index
df["seller_country"] = df["seller_country"].replace(rare, "Other")


df["log_price"]   = np.log10(df["price_usd"])
df["log_likes"]   = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)
if exp_var:
    df["log_experience"] = np.log10(1 + df[exp_var])
else:
    df["log_experience"] = 0


train, test = train_test_split(
    df,
    test_size=0.20,
    stratify=df["product_category"],
    random_state=42
)


formula = """
log_price ~ C(product_condition)
         + C(product_category)
         + C(seller_country)
         + log_likes
         + rating_star
         + log_experience
         + rating_star:log_experience
"""


model = smf.ols(formula, data=train).fit(cov_type="HC3")
print(model.summary())


y_pred    = model.predict(test)
mape_val  = mean_absolute_percentage_error(test["log_price"], y_pred)
mse_val   = mean_squared_error(test["log_price"], y_pred)
rmse_val  = np.sqrt(mse_val)

print("\n=== Performance test set ===")
print(f"MAPE : {mape_val:.3%}")
print(f"RMSE : {rmse_val:.3f} (log₁₀ scale)")


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


y = train["log_price"]
X = pd.get_dummies(
    train[
        [
            "product_condition",
            "product_category",
            "seller_country",
            "log_likes",
            "rating_star",
            "log_experience",
        ]
    ],
    drop_first=True,
    dtype=float,
)


alphas = np.logspace(-3, 3, 13)
ridge_pipe = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
)
ridge_pipe.fit(X, y)

best_alpha = ridge_pipe.named_steps["ridgecv"].alpha_
print(f"\n=== RidgeCV completato – α ottimale: {best_alpha:.4f}")


X_test = pd.get_dummies(
    test[
        [
            "product_condition",
            "product_category",
            "seller_country",
            "log_likes",
            "rating_star",
            "log_experience",
        ]
    ],
    drop_first=True,
    dtype=float,
)


X_test = X_test.reindex(columns=X.columns, fill_value=0)

y_pred_ridge = ridge_pipe.predict(X_test)
mape_ridge   = mean_absolute_percentage_error(test["log_price"], y_pred_ridge)
rmse_ridge   = np.sqrt(mean_squared_error(test["log_price"], y_pred_ridge))

print("=== Ridge – performance test ===")
print(f"MAPE : {mape_ridge:.3%}")
print(f"RMSE : {rmse_ridge:.3f} (log₁₀ scale)")


coefs = pd.Series(
    ridge_pipe.named_steps["ridgecv"].coef_,
    index=X.columns
).round(4)
coefs.to_csv("coefficients_price_model_ridge.csv")
print("Coefficienti Ridge salvati in coefficients_price_model_ridge.csv")



model.params.round(4).to_csv("coefficients_price_model.csv")
print("Coefficienti salvati in coefficients_price_model.csv")


coef = model.params
conf = model.conf_int(alpha=0.05)
conf.columns = ["lower", "upper"]
coef_df = pd.concat([coef, conf], axis=1)


vars_to_plot = [
    "log_likes",
    "rating_star",
    "log_experience",
    "rating_star:log_experience"
]
plot_df = coef_df.loc[vars_to_plot]


plt.figure(figsize=(8, 5))
plt.errorbar(
    x=plot_df.index,
    y=plot_df[0],
    yerr=[plot_df[0] - plot_df["lower"], plot_df["upper"] - plot_df[0]],
    fmt='o',
    capsize=5
)
plt.axhline(0, color='grey', linewidth=0.8)
plt.ylabel("Coefficiente stimato")
plt.title("Coefficiente e 95% CI – variabili chiave")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


coef = model.params
conf = model.conf_int(alpha=0.05)
conf.columns = ["lower", "upper"]


coef_like = coef["log_likes"]
conf_like = conf.loc["log_likes"]

plt.figure(figsize=(4, 5))
plt.errorbar(
    x=coef_like,
    y=["log_likes"],
    xerr=[[coef_like - conf_like["lower"]], [conf_like["upper"] - coef_like]],
    fmt='o',
    capsize=5,
    color="crimson"
)
plt.axvline(0, color="grey", linestyle="--", linewidth=1)
plt.title("Coefficiente OLS per log_likes con IC 95%")
plt.xlabel("Stima β (log₁₀ prezzo)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("ols_loglikes_coef.png", dpi=300)
plt.show()



coef_full = model.params
conf_full = model.conf_int(alpha=0.05)
conf_full.columns = ["lower", "upper"]
coef_full_df = pd.concat([coef_full, conf_full], axis=1).sort_values(0)

plt.figure(figsize=(8, len(coef_full_df) / 2))
plt.errorbar(
    x=coef_full_df[0],
    y=coef_full_df.index,
    xerr=[coef_full_df[0] - coef_full_df["lower"], coef_full_df["upper"] - coef_full_df[0]],
    fmt='o', capsize=3, color="darkblue"
)
plt.axvline(0, color="grey", linestyle="--")
plt.title("Coefficienti OLS con intervalli di confidenza al 95%")
plt.xlabel("Stima β")
plt.tight_layout()
plt.savefig("ols_coefficients_forestplot.png", dpi=300)
plt.show()



country_counts = df["seller_country"].value_counts()
valid_countries = country_counts[country_counts >= 200].index.tolist()


coef_country = model.params.filter(like="C(seller_country)")
conf_country = model.conf_int(alpha=0.05).loc[coef_country.index]
conf_country.columns = ["lower", "upper"]

coef_country_df = pd.concat([coef_country, conf_country], axis=1)
coef_country_df = coef_country_df.reset_index().rename(columns={"index": "variable", 0: "coef"})


coef_country_df["country"] = (
    coef_country_df["variable"]
    .str.replace("C(seller_country)[T.", "", regex=False)
    .str.replace("]", "", regex=False)
)


coef_country_df = coef_country_df[coef_country_df["country"].isin(valid_countries)]


coef_country_df = coef_country_df.sort_values("coef", ascending=True)


apac = ["Japan", "China", "Hong Kong"]
coef_country_df["is_apac"] = coef_country_df["country"].isin(apac)


plt.figure(figsize=(8, len(coef_country_df) / 2))
plt.errorbar(
    x=coef_country_df["coef"],
    y=coef_country_df["country"],
    xerr=[coef_country_df["coef"] - coef_country_df["lower"], coef_country_df["upper"] - coef_country_df["coef"]],
    fmt='o',
    capsize=3,
    color="grey",
    alpha=0.7
)


apac_df = coef_country_df[coef_country_df["is_apac"]]
plt.errorbar(
    x=apac_df["coef"],
    y=apac_df["country"],
    xerr=[apac_df["coef"] - apac_df["lower"], apac_df["upper"] - apac_df["coef"]],
    fmt='o',
    capsize=3,
    color="crimson",
    label="APAC"
)

plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Coefficienti OLS (log prezzo) – Dummies Paese con n ≥ 200")
plt.xlabel("Stima β (log₁₀ prezzo)")
plt.ylabel("Paese venditore")
plt.legend()
plt.tight_layout()
plt.savefig("ols_coefficients_country.png", dpi=300)
plt.show()

