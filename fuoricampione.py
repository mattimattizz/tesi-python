import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import statsmodels.formula.api as smf

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File non trovato: {csv_path}")

df = pd.read_csv(csv_path)

keep_cols = [
    "price_usd", "product_condition", "product_category",
    "seller_country", "product_like_count", "seller_community_rank"
]
df = (
    df[keep_cols]
    .dropna(subset=["price_usd", "product_condition", "product_category", "seller_country"])
    .query("price_usd > 0")
    .copy()
)

df["log_price"] = np.log10(df["price_usd"])
df["log_likes"] = np.log10(1 + df["product_like_count"])
df["rating_star"] = df["seller_community_rank"].fillna(df["seller_community_rank"].median())

X_full = pd.get_dummies(
    df[["log_likes", "rating_star", "product_condition",
        "product_category", "seller_country"]],
    drop_first=True, dtype=float
)
y_full = df["log_price"].values

X_full = sm.add_constant(X_full)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    ols_model = sm.OLS(y_train, X_train).fit()
    y_pred_ols = ols_model.predict(X_test)
    r2_ols = r2_score(y_test, y_pred_ols)
    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    factor_ols = 10**rmse_ols

    ridge = Ridge(alpha=10)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    factor_ridge = 10**rmse_ridge

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    try:
        model_ml = smf.mixedlm(
            "log_price ~ log_likes + rating_star + C(product_condition) + C(product_category)",
            data=train_df, groups=train_df["seller_country"]
        ).fit(reml=False, method="lbfgs", maxiter=200)
        y_pred_ml = model_ml.predict(test_df)
        r2_ml = r2_score(test_df["log_price"], y_pred_ml)
        rmse_ml = np.sqrt(mean_squared_error(test_df["log_price"], y_pred_ml))
        factor_ml = 10**rmse_ml
    except Exception:
        r2_ml = np.nan
        rmse_ml = np.nan
        factor_ml = np.nan

    metrics.extend([
        {"fold": fold, "model": "OLS", "R2": r2_ols, "RMSE_log": rmse_ols, "Factor": factor_ols},
        {"fold": fold, "model": "Ridge", "R2": r2_ridge, "RMSE_log": rmse_ridge, "Factor": factor_ridge},
        {"fold": fold, "model": "Multilevel", "R2": r2_ml, "RMSE_log": rmse_ml, "Factor": factor_ml},
    ])

res = pd.DataFrame(metrics)


summary = res.groupby("model").agg(
    R2_mean=("R2", "mean"),
    R2_sd=("R2", "std"),
    RMSE_mean=("RMSE_log", "mean"),
    Factor_mean=("Factor", "mean")
).round(3)

print("\n=== Prestazioni 5-fold CV ===")
print(summary)
summary.to_csv("cv_summary_models_fixed.csv")

plt.figure(figsize=(8, 5))
sns.boxplot(data=res, x="model", y="R2", palette="Set2")
plt.title("Validazione 5-fold – R² fuori campione")
plt.ylabel("R² (distribuzione per fold)")
plt.xlabel("Modello")
plt.tight_layout()
plt.savefig("cv_r2_boxplot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=res, x="model", y="Factor", palette="Set2")
plt.title("Errore fuori campione – fattore moltiplicativo (10^RMSE_log)")
plt.ylabel("Fattore moltiplicativo")
plt.xlabel("Modello")
plt.tight_layout()
plt.savefig("cv_factor_boxplot.png", dpi=300)
plt.show()
