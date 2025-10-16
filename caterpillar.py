'''caterpillar'''
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")
df = pd.read_csv(csv_path)

df = df[["price_usd", "seller_country", "product_like_count", "seller_community_rank"]].dropna()
df = df.query("price_usd > 0").copy()
df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].astype(int)

counts = df["seller_country"].value_counts()
valid_countries = counts[counts >= 200].index
df = df[df["seller_country"].isin(valid_countries)].copy()

md = sm.MixedLM.from_formula(
    "log_price ~ log_likes + rating_star",
    groups="seller_country",
    data=df
)
mfit = md.fit(reml=True, method="powell", maxiter=200)
print(mfit.summary())

re = mfit.random_effects
var_random = mfit.cov_re.iloc[0, 0]
sd_random = np.sqrt(var_random)

blup = pd.DataFrame({
    "country": list(re.keys()),
    "blup": [v.values[0] for v in re.values()],
})
blup["lower"] = blup["blup"] - 1.96 * sd_random
blup["upper"] = blup["blup"] + 1.96 * sd_random
blup = blup.merge(counts.rename("n_j"), left_on="country", right_index=True)

blup = blup.sort_values("blup")

apac = ["Japan", "China", "Hong Kong"]
blup["is_apac"] = blup["country"].isin(apac)

fig, ax = plt.subplots(figsize=(8.5, len(blup) / 2), constrained_layout=False)

ax.errorbar(
    x=blup["blup"], y=blup["country"],
    xerr=[blup["blup"] - blup["lower"], blup["upper"] - blup["blup"]],
    fmt="o", capsize=3, color="grey", alpha=0.7
)

apac_df = blup[blup["is_apac"]]
ax.errorbar(
    x=apac_df["blup"], y=apac_df["country"],
    xerr=[apac_df["blup"] - apac_df["lower"], apac_df["upper"] - apac_df["blup"]],
    fmt="o", capsize=3, color="crimson", label="APAC"
)

ax.axvline(0, color="black", linestyle="--", linewidth=1)

for _, row in blup.iterrows():
    ax.text(row["upper"] + 0.01, row["country"], f"n={row['n_j']}", va="center", fontsize=8)

ax.set_title("Intercette casuali per Paese (BLUP) con IC 95% – log(prezzo)")
ax.set_xlabel("Intercetta casuale stimata (log₁₀ prezzo)")
ax.set_ylabel("Paese venditore")
ax.legend()

fig.subplots_adjust(left=0.33, bottom=0.18, right=0.98, top=0.95)

ax.margins(x=0.05)

plt.savefig("caterpillar_blup_country.png", dpi=300, bbox_inches="tight")
plt.show()