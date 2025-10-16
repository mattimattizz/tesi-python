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

keep_cols = ["price_usd", "product_condition", "product_category",
             "seller_country", "product_like_count"]
df = (
    df[keep_cols]
    .dropna(subset=keep_cols)
    .query("price_usd > 0")
    .copy()
)

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])

apac_countries = ["China", "Hong Kong", "Japan", "Singapore", "South Korea", "Taiwan"]
df["APAC"] = df["seller_country"].isin(apac_countries).astype(int)

formula_base = "log_price ~ log_likes + C(product_condition) + C(product_category)"
base_model = smf.ols(formula_base, data=df).fit(cov_type="HC3")

df["yhat_base"] = base_model.predict(df)

n_segments = 4
df["segment"] = pd.qcut(df["yhat_base"], q=n_segments, labels=False) + 1

results = []

formula_apac = formula_base + " + APAC"
model_global = smf.ols(formula_apac, data=df).fit(cov_type="HC3")
beta_apac = model_global.params["APAC"]
ci_low, ci_up = model_global.conf_int().loc["APAC"]

results.append({
    "segment": "Global",
    "coef": beta_apac,
    "ci_low": ci_low,
    "ci_up": ci_up
})

for q in sorted(df["segment"].unique()):
    seg_data = df[df["segment"] == q].copy()
    model_seg = smf.ols(formula_apac, data=seg_data).fit(cov_type="HC3")
    b = model_seg.params.get("APAC", np.nan)
    ci = model_seg.conf_int().loc["APAC"].tolist() if "APAC" in model_seg.params else [np.nan, np.nan]
    results.append({
        "segment": f"Q{q}",
        "coef": b,
        "ci_low": ci[0],
        "ci_up": ci[1]
    })

res_df = pd.DataFrame(results)

margins = []
for q in res_df["segment"]:
    if q == "Global":
        mean_price = df["price_usd"].mean()
    else:
        seg_id = int(q[1])
        mean_price = df.loc[df["segment"] == seg_id, "price_usd"].mean()
    coef = res_df.loc[res_df["segment"] == q, "coef"].values[0]
    if pd.notnull(coef):
        margin = (10**coef - 1) * mean_price
    else:
        margin = np.nan
    margins.append(margin)

res_df["margin_euro"] = margins

plt.figure(figsize=(8, 5))
sns.pointplot(
    data=res_df,
    x="segment", y="coef",
    join=False, capsize=0.2, color="crimson"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylabel("β_APAC (log10-prezzo)")
plt.xlabel("Segmento prognostico (quantili di ŷ)")
plt.title("Premio APAC per segmento di prezzo atteso")
plt.tight_layout()
plt.savefig("premio_apac_per_segmento.png", dpi=300)
plt.show()

res_df.to_csv("tabella_apac_stratification.csv", index=False)
print(res_df)