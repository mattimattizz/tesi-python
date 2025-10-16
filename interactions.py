"""
price_country_interactions.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

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
]

df = (
    df[keep_cols]
    .dropna(subset=keep_cols)
    .query("price_usd > 0")
    .copy()
)


min_n = 100
country_counts = df["seller_country"].value_counts()
rare = country_counts[country_counts < min_n].index
df["seller_country"] = df["seller_country"].replace(rare, "Other")

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])

samp = (
    df.groupby("seller_country", group_keys=False, observed=True)
      .apply(lambda x: x.sample(frac=0.25, random_state=42))
)

train, test = train_test_split(
    samp, test_size=0.20, stratify=samp["product_category"], random_state=42
)

formula_int = """
log_price ~ log_likes
          + C(product_condition)
          + C(seller_country)*C(product_category)
"""

model_int = smf.ols(formula_int, data=train).fit(cov_type="HC3")
print(model_int.summary())

import matplotlib.pyplot as plt
import seaborn as sns

mask = model_int.params.index.str.contains(
    r"C\(seller_country\)\[T\..+\]:C\(product_category\)\[T\..+\]"
)
inter_coefs = model_int.params[mask]

if inter_coefs.empty:
    print("⚠️ Nessun termine di interazione 'Paese × Categoria' trovato.")
else:
    parsed = inter_coefs.index.to_series().str.extract(
        r"C\(seller_country\)\[T\.(?P<Paese>.+?)\]:C\(product_category\)\[T\.(?P<Categoria>.+?)\]"
    )

    coef_df = pd.DataFrame({
        "Paese": parsed["Paese"].values,
        "Categoria": parsed["Categoria"].values,
        "beta": inter_coefs.values,
        "pval": model_int.pvalues[mask].values
    })

    coef_df["Delta_%"] = (10**coef_df["beta"] - 1) * 100

    all_countries = sorted(samp["seller_country"].unique())
    all_categories = sorted(samp["product_category"].unique())

    pivot = (coef_df
             .pivot(index="Paese", columns="Categoria", values="Delta_%")
             .reindex(index=all_countries, columns=all_categories)
             .fillna(0.0))

    base_country = sorted(df["seller_country"].unique())[0]
    base_category = sorted(df["product_category"].unique())[0]

    fig, ax = plt.subplots(
        figsize=(max(14, 0.8 * len(all_categories)), max(10, 0.4 * len(all_countries)))
    )

    sns.heatmap(
        pivot, annot=False, cmap="coolwarm", center=0,
        vmin=-150, vmax=150,
        cbar_kws={"label": "Δ% sul prezzo atteso (interazione)"},
        ax=ax
    )

    ax.set_title("Grafico Paese x Categoria – Effetti di interazione (Δ%)", pad=20)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Paese")

    fig.subplots_adjust(bottom=0.2)
    fig.text(
        0.5, 0.05,
        f"Didascalia: categoria di riferimento = '{base_category}', "
        f"Paese base = '{base_country}'.",
        wrap=True, ha="center", fontsize=9
    )

    fig.savefig("grafico_paese_x_categoria.png", dpi=300, bbox_inches="tight")
    plt.show()


    sns.heatmap(
        pivot, annot=False, cmap="coolwarm", center=0,
        vmin=-150, vmax=150, cbar_kws={"label": "Δ% sul prezzo atteso (interazione)"},
        ax=ax
    )

    not_sig = pvals >= 0.05
    for i in range(not_sig.shape[0]):
        for j in range(not_sig.shape[1]):
            if not_sig.iloc[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, hatch='///', edgecolor='gray', lw=0
                ))

    ax.set_title("Grafico Paese x Categoria – Effetti di interazione (Δ%)", pad=20)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Paese")

    fig.subplots_adjust(bottom=0.2)
    fig.text(
        0.5, 0.05,
        f"Didascalia: categoria di riferimento = '{base_category}', "
        f"Paese base = '{base_country}'. "
        f"Celle tratteggiate: interazioni non significative (p ≥ 0.05).",
        wrap=True, ha="center", fontsize=9
    )

    fig.savefig("grafico_paese_x_categoria.png", dpi=300, bbox_inches="tight")
    plt.show()

    not_sig = pvals >= 0.05
    for i in range(not_sig.shape[0]):
        for j in range(not_sig.shape[1]):
            if not_sig.iloc[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, hatch='///', edgecolor='gray', lw=0
                ))

    base_country = df["seller_country"].value_counts().idxmax()  # il più frequente
    base_category = df["product_category"].value_counts().idxmax()  # idem
    plt.title("Grafico Paese x Categoria – Effetti di interazione (Δ%)")
    plt.xlabel("Categoria")
    plt.ylabel("Paese")
    plt.figtext(
        0.5, -0.02,
        f"Didascalia: categoria di riferimento = '{base_category}', Paese base = '{base_country}'. "
        f"Celle tratteggiate: interazioni non significative (p ≥ 0.05).",
        wrap=True, ha="center", fontsize=9
    )

    plt.tight_layout()
    plt.savefig("grafico_paese_x_categoria.png", dpi=300, bbox_inches="tight")
    plt.show()
