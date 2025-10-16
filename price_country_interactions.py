"""
price_country_interactions.py
Stima OLS su log10(prezzo)
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
