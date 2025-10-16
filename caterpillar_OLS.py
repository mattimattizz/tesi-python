"""
caterpillar_ols_country_matched.py
Caterpillar OLS (dummies Paese, n ≥ 200) 
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")


df = pd.read_csv(csv_path, low_memory=False)
df = (
    df[[
        "price_usd",
        "product_condition",
        "product_category",
        "seller_country",
        "product_like_count",
        "seller_community_rank",
    ]]
    .dropna()
    .query("price_usd > 0")
    .copy()
)
df["log_price"]   = np.log10(df["price_usd"])
df["log_likes"]   = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)


formula = """
log_price ~ C(product_condition)
         + C(product_category)
         + C(seller_country)
         + log_likes
         + rating_star
"""
model = smf.ols(formula, data=df).fit(cov_type="HC3")


counts = df["seller_country"].value_counts()
valid  = counts[counts >= 200].index.tolist()

coef = model.params.filter(like="C(seller_country)")
ci   = model.conf_int(0.05).loc[coef.index]
ci.columns = ["lower", "upper"]

tab = (
    pd.concat([coef.rename("coef"), ci], axis=1)
      .reset_index().rename(columns={"index": "var"})
)

tab["country"] = (
    tab["var"].str.replace("C(seller_country)[T.", "", regex=False)
              .str.replace("]", "", regex=False)
)
tab = tab[tab["country"].isin(valid)].sort_values("coef").reset_index(drop=True)


apac_list = ["Japan", "China", "Hong Kong"]
tab["is_apac"] = tab["country"].isin(apac_list)


xmin, xmax = tab["lower"].min(), tab["upper"].max()
span = max(1e-6, xmax - xmin)
xpad = 0.06 * span
xlim = (xmin - xpad, xmax + 2.0 * xpad)



rows   = len(tab)
height = max(6.2, 0.34 * rows + 2.0)     #
width  = 11.0

plt.rcParams.update({
    "figure.dpi": 180,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 4,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(width, height))


ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlim(*xlim)


ypos = np.arange(rows)


ax.errorbar(
    tab["coef"], ypos,
    xerr=[tab["coef"] - tab["lower"], tab["upper"] - tab["coef"]],
    fmt="o", ms=3, capsize=2, elinewidth=0.8,
    color="grey", alpha=0.75, zorder=2
)


mask = tab["is_apac"].values
if mask.any():
    ax.errorbar(
        tab.loc[mask, "coef"], ypos[mask],
        xerr=[tab.loc[mask, "coef"] - tab.loc[mask, "lower"],
              tab.loc[mask, "upper"] - tab.loc[mask, "coef"]],
        fmt="o", ms=3.4, capsize=2, elinewidth=0.9,
        color="crimson", label="APAC", zorder=3
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.02), frameon=False)


ax.set_yticks(ypos)
ax.set_yticklabels(tab["country"])
ax.tick_params(axis="y", pad=2)


ax.set_title("Dummies Paese con n ≥ 200 – Coefficienti OLS (log₁₀ prezzo)", pad=6)
ax.set_xlabel("Stima β (log₁₀ prezzo)")
ax.set_ylabel("Paese venditore")


left = min(0.50, max(0.30, 0.0115 * max(len(s) for s in tab["country"])))
fig.subplots_adjust(left=left, right=0.98, top=0.90, bottom=0.16)

out_path = "caterpillar_ols_matched.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.30)
plt.show()
print(f"Grafico salvato in: {out_path}")
