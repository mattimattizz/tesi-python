"""
interactions_apac_condition_bars.py
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

keep_cols = ["price_usd", "product_condition", "seller_country", "product_like_count"]
df = (
    df[keep_cols]
    .dropna(subset=keep_cols)
    .query("price_usd > 0")
    .copy()
)

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])

apac_countries = ["China", "Hong Kong", "Japan", "Singapore", "South Korea", "Taiwan"]
df = df[df["seller_country"].isin(apac_countries)].copy()

present_countries = sorted(df["seller_country"].unique())
if not present_countries:
    raise ValueError("Nessun Paese APAC presente dopo il filtro. Controlla i nomi nel CSV.")

print("Paesi APAC presenti nel dataset:", present_countries)


formula_int = """
log_price ~ log_likes + C(seller_country) * C(product_condition)
"""
model = smf.ols(formula_int, data=df).fit(cov_type="HC3")
print(model.summary())

base_country = sorted(df["seller_country"].unique())[0]
base_condition = sorted(df["product_condition"].unique())[0]

cond_map_disp = {
    "Good": "Good Condition",
    "Very good": "Very Good Condition",
    "Never worn": "Never Worn",
    "Never worn, with tag": "Never Worn With Tag",
}
display_order = ["Good Condition", "Very Good Condition", "Never Worn", "Never Worn With Tag"]


raw_conditions_present = [c for c in ["Good", "Very good", "Never worn", "Never worn, with tag"]
                          if c in df["product_condition"].unique()]
if not raw_conditions_present:
    raise ValueError("Nessuna delle condizioni attese è presente nel dataset.")

params = model.params

def total_beta(country, cond_raw):

    beta = 0.0

    if country != base_country:
        key_country = f"C(seller_country)[T.{country}]"
        beta += params.get(key_country, 0.0)

    if cond_raw != base_condition:
        key_cond = f"C(product_condition)[T.{cond_raw}]"
        beta += params.get(key_cond, 0.0)

    if (country != base_country) and (cond_raw != base_condition):
        key_inter = f"C(seller_country)[T.{country}]:C(product_category)[T.{cond_raw}]"
        key_inter = f"C(seller_country)[T.{country}]:C(product_condition)[T.{cond_raw}]"
        beta += params.get(key_inter, 0.0)

    return beta

rows = []
for country in present_countries:
    for cond_raw in raw_conditions_present:
        beta = total_beta(country, cond_raw)
        delta_pct = (10**beta - 1) * 100
        cond_disp = cond_map_disp.get(cond_raw, cond_raw)  # fallback al raw se non mappato
        rows.append({
            "Paese": country,
            "Condizione_disp": cond_disp,
            "beta": beta,
            "Delta_%": delta_pct
        })

res = pd.DataFrame(rows)

res = res[res["Condizione_disp"].isin(display_order)].copy()

res["Condizione_disp"] = pd.Categorical(res["Condizione_disp"], categories=display_order, ordered=True)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=res,
    x="Condizione_disp", y="Delta_%", hue="Paese",
    order=display_order, palette="Set2", dodge=True
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("APAC × Condizioni d'uso – Effetti stimati (Δ%)")
plt.ylabel("Δ% sul prezzo atteso")
plt.xlabel("Condizione d'uso")
plt.legend(title="Paese APAC", ncol=2)
plt.tight_layout()
plt.savefig("grafico_apac_condizioni_ord.png", dpi=300)
plt.show()
