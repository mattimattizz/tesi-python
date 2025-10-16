"""
robustness_sensitivity.py – Robustezza e sensitivity analysis
Output:
  - T4.6: confronto coefficienti β_Paese per specifica
  - F4.15: distribuzione placebo β_Paese (istogramma, highlight 99°)
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

q1, q99 = df["log_price"].quantile([0.01, 0.99])
df["log_price_w"] = df["log_price"].clip(lower=q1, upper=q99)

min_n = 100
country_counts = df["seller_country"].value_counts()
rare = country_counts[country_counts < min_n].index
df["seller_country"] = df["seller_country"].replace(rare, "Other")

models = {}
specs = {
    "Baseline": "log_price ~ log_likes + rating_star + C(product_condition) + C(product_category) + C(seller_country)",
    "NoLikesRating": "log_price ~ C(product_condition) + C(product_category) + C(seller_country)",
    "Winsorized": "log_price_w ~ log_likes + rating_star + C(product_condition) + C(product_category) + C(seller_country)",
    "HighN": "log_price ~ log_likes + rating_star + C(product_condition) + C(product_category) + C(seller_country)",
}

high_n = 300
country_counts_hn = df["seller_country"].value_counts()
rare_hn = country_counts_hn[country_counts_hn < high_n].index
df_hn = df.copy()
df_hn["seller_country"] = df_hn["seller_country"].replace(rare_hn, "Other")

models["Baseline"] = smf.ols(specs["Baseline"], data=df).fit(cov_type="HC3")
models["NoLikesRating"] = smf.ols(specs["NoLikesRating"], data=df).fit(cov_type="HC3")
models["Winsorized"] = smf.ols(specs["Winsorized"], data=df).fit(cov_type="HC3")
models["HighN"] = smf.ols(specs["HighN"], data=df_hn).fit(cov_type="HC3")


countries = ["Japan", "China", "Hong Kong", "Russia"]
table_data = []

for spec_name, model in models.items():
    for c in countries:
        key = f"C(seller_country)[T.{c}]"
        if key in model.params:
            beta = model.params[key]
            ci = model.conf_int().loc[key]
            table_data.append({
                "Specifica": spec_name,
                "Paese": c,
                "β": beta,
                "CI_lower": ci[0],
                "CI_upper": ci[1],
                "Sign.": "✓" if model.pvalues[key] < 0.05 else ""
            })

tab = pd.DataFrame(table_data)
tab["β(%)"] = (10**tab["β"] - 1) * 100
tab["CI_lower(%)"] = (10**tab["CI_lower"] - 1) * 100
tab["CI_upper(%)"] = (10**tab["CI_upper"] - 1) * 100
tab = tab.round(2)

print("\n=== Tabella T4.6 – Coefficienti per specifica ===")
print(tab)

tab.to_csv("T4_6_confronto_specifiche.csv", index=False)

N_REPS = 300
placebo_betas = []

for i in range(N_REPS):
    df_rand = df.copy()
    df_rand["seller_country"] = np.random.permutation(df_rand["seller_country"])
    m_rand = smf.ols(specs["Baseline"], data=df_rand).fit(cov_type="HC3")
    key = "C(seller_country)[T.Japan]"
    if key in m_rand.params:
        placebo_betas.append(m_rand.params[key])

placebo_betas = np.array(placebo_betas)
real_beta = models["Baseline"].params.get("C(seller_country)[T.Japan]", np.nan)

plt.figure(figsize=(7, 4))
sns.histplot(placebo_betas, bins=30, color="lightgray", edgecolor="black")
plt.axvline(real_beta, color="red", linestyle="--", label=f"β Japan reale = {real_beta:.3f}")
p99 = np.percentile(placebo_betas, 99)
plt.axvline(p99, color="black", linestyle=":", label="99° percentile placebo")
plt.xlabel("β stimato (Placebo)")
plt.ylabel("Frequenza")
plt.title("F4.15 – Distribuzione placebo β_Paese (Japan)")
plt.legend()
plt.tight_layout()
plt.savefig("F4_15_placebo.png", dpi=300, bbox_inches="tight")
plt.show()
