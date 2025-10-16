import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns          # pip install seaborn
import statsmodels.formula.api as smf    # pip install statsmodels


csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
print(f"Carico il file da: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"⚠️  Il file non è stato trovato a: {csv_path}")


df = pd.read_csv(csv_path)
print("✓ CSV caricato, dimensioni:", df.shape)


cols_needed = ["product_like_count", "sold"]
missing_cols = set(cols_needed) - set(df.columns)
if missing_cols:
    raise KeyError(f"⚠️  Colonne mancanti nel CSV: {missing_cols}")

df_likes = (
    df[cols_needed]
    .dropna(subset=cols_needed)
    .copy()
)


df_likes["sold"] = df_likes["sold"].astype(bool).astype(int)


df_likes["log_likes"] = np.log1p(df_likes["product_like_count"])


mask_zero = df_likes["product_like_count"] == 0
df_likes.loc[mask_zero, "likes_group"] = "0 likes"


non_zero = df_likes.loc[~mask_zero, "product_like_count"]
df_likes.loc[~mask_zero, "likes_group"] = pd.qcut(
    non_zero,
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
)


group_order = ["0 likes", "Q1", "Q2", "Q3", "Q4"]


quintile_rate = (
    df_likes.groupby("likes_group", observed=True)["sold"]
    .mean()
    .mul(100)
    .round(1)
    .reindex(group_order)
)


logit_res = smf.logit('sold ~ log_likes', data=df_likes).fit(disp=False)
print(logit_res.summary())


fig, axes = plt.subplots(
    1, 2, figsize=(12, 6),
    gridspec_kw=dict(width_ratios=[1, 1.2])
)


df_likes["sold"] = df_likes["sold"].astype("category")

sns.boxplot(
    data=df_likes,
    x="sold",
    y="log_likes",
    order=[0, 1],
    showfliers=False,
    ax=axes[0],
    width=.6,
)

sample = df_likes.sample(min(5000, len(df_likes)), random_state=0)
sns.stripplot(
    data=sample,
    x="sold",
    y="log_likes",
    order=[0, 1],
    ax=axes[0],
    size=2,
    jitter=.25,
    alpha=.15,
    color="black",
)
axes[0].set_xticklabels(["Non venduto", "Venduto"])
axes[0].set_xlabel("")
axes[0].set_ylabel("log₁₀(1 + like)")
axes[0].set_title("Distribuzione like (log-scala)")


axes[1].bar(quintile_rate.index, quintile_rate.values)
axes[1].set_ylim(0, 100)
axes[1].set_ylabel("Percentuale venduti (%)")
axes[1].set_xlabel("Gruppo di like")
axes[1].set_title("Probabilità di vendita ↑ con i like")

for idx, val in enumerate(quintile_rate.values):
    if not np.isnan(val):
        axes[1].text(idx, val + 2, f"{val}%", ha="center", va="bottom", fontsize=9)

fig.suptitle(
    "Articoli con più like vengono venduti più spesso?",
    fontsize=14, weight="bold"
)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
