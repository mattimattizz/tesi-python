import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
ROW_SUBSAMPLE_FRAC = 0.50
N_ESTIMATORS       = 120
MAX_DEPTH          = 12
MIN_SAMPLES_LEAF   = 10
MAX_SAMPLES_TREE   = 0.30
N_REPEATS_PERM     = 5

usecols = ["price_usd","product_like_count","seller_num_followers",
           "seller_products_sold","seller_badge","seller_country"]
try:
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
except ValueError:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[[c for c in usecols if c in df.columns]]


if 0 < ROW_SUBSAMPLE_FRAC < 1:
    df = df.sample(frac=ROW_SUBSAMPLE_FRAC, random_state=42)

df = df.copy()
df["price_usd"] = pd.to_numeric(df.get("price_usd"), errors="coerce")
df = df[df["price_usd"] > 0]
df["log_price"] = np.log10(df["price_usd"]).astype(np.float32)

def log10p(s): return np.log10(1 + pd.to_numeric(s, errors="coerce").fillna(0)).astype(np.float32)
df["log_likes"]     = log10p(df["product_like_count"])   if "product_like_count"   in df.columns else np.float32(0)
df["log_followers"] = log10p(df["seller_num_followers"]) if "seller_num_followers" in df.columns else np.float32(0)
df["log_sold"]      = log10p(df["seller_products_sold"]) if "seller_products_sold" in df.columns else np.float32(0)

def badge_to_int(s): return int(str(s).strip().lower() in {"true","1","yes","pro","trusted","verified","ambassador","expert","official"})
df["seller_badge_dummy"] = (
    df["seller_badge"].map(badge_to_int).fillna(0).astype(np.float32)
    if "seller_badge" in df.columns else np.float32(0)
)

APAC_SET = {"china","japan","south korea","korea, republic of","republic of korea","singapore",
            "hong kong","hong kong sar","taiwan","macau","thailand","vietnam","malaysia",
            "philippines","indonesia","australia","new zealand"}
df["APAC"] = (
    df["seller_country"].astype(str).str.strip().str.lower().isin(APAC_SET).astype(np.float32)
    if "seller_country" in df.columns else np.float32(0)
)

X = df[["APAC","log_likes","log_followers","log_sold","seller_badge_dummy"]].astype(np.float32)
y = df["log_price"].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

try:
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt", bootstrap=True, max_samples=MAX_SAMPLES_TREE,
        n_jobs=1, random_state=42, oob_score=False
    )
except TypeError:
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt", bootstrap=True, n_jobs=1, random_state=42, oob_score=False
    )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2   = float(r2_score(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
err_mult = float(10**rmse)

if "APAC" in X_test.columns:
    perm = permutation_importance(rf, X_test, y_test, n_repeats=N_REPEATS_PERM,
                                  random_state=42, scoring="r2", n_jobs=1)
    apac_idx = list(X_test.columns).index("APAC")
    apac_perm_dR2 = float(perm.importances_mean[apac_idx])
else:
    apac_perm_dR2 = None

if "APAC" in X_test.columns:
    X0, X1 = X_test.copy(), X_test.copy()
    X0["APAC"], X1["APAC"] = np.float32(0), np.float32(1)
    d = rf.predict(X1) - rf.predict(X0)
    pdp_mean = float(d.mean())
    mult_mean = float(10**pdp_mean)

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(d), (80, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    ci_mult = (float(10**lo), float(10**hi))
else:
    pdp_mean, mult_mean, ci_mult = None, None, None

summary = {
    "R2_test": round(r2, 4),
    "RMSE_log10_test": round(rmse, 4),
    "Err_moltiplicativo_tipico": round(err_mult, 3),
    "APAC_perm_dR2": None if apac_perm_dR2 is None else round(apac_perm_dR2, 6),
    "APAC_PDP_delta_log10_mean": None if pdp_mean is None else round(pdp_mean, 5),
    "APAC_PDP_moltiplicatore_mean": None if mult_mean is None else round(mult_mean, 4),
    "APAC_PDP_CI95_moltiplicatore": None if ci_mult is None else (round(ci_mult[0], 4), round(ci_mult[1], 4)),
}
print(summary)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=N_REPEATS_PERM, random_state=42, scoring="r2", n_jobs=1
)

pi_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_dR2": perm.importances_mean,
    "std_dR2": perm.importances_std
}).sort_values("mean_dR2", ascending=True)

plt.figure(figsize=(7, 4))
ax = sns.barplot(
    data=pi_df, x="mean_dR2", y="feature",
    hue="feature", dodge=False, legend=False, palette="viridis"
)

for i, (mean, std) in enumerate(zip(pi_df["mean_dR2"], pi_df["std_dR2"])):
    ax.errorbar(x=mean, y=i, xerr=std, fmt="none", ecolor="black", capsize=3, elinewidth=1)

plt.xlabel("ΔR² medio (Permutation Importance)")
plt.ylabel("")
plt.title("F4.11 – Permutation Importance (Random Forest)")
plt.tight_layout()
plt.savefig("F4_11_PI.png", dpi=300, bbox_inches="tight")
plt.show()

if "APAC" in X_test.columns:
    X0, X1 = X_test.copy(), X_test.copy()
    X0["APAC"], X1["APAC"] = np.float32(0), np.float32(1)
    pred0, pred1 = rf.predict(X0), rf.predict(X1)

    mult0 = 10**pred0
    mult1 = 10**pred1
    diff_log = pred1 - pred0

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff_log), (300, len(diff_log)))
    boot = diff_log[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    mult_mean = 10**diff_log.mean()
    lo_mult, hi_mult = 10**lo, 10**hi

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["Non-APAC", "APAC"],
        y=[mult0.mean(), mult1.mean()],
        palette=["#1f77b4", "#d62728"]
    )
    plt.ylabel("Prezzo atteso (scala originale)")
    plt.title("F4.12 – Partial Dependence Plot (APAC = 0 vs 1)")
    plt.text(0.5, (mult0.mean() + mult1.mean()) / 2,
             f"Premio medio: ×{mult_mean:.2f}\n(IC95%: {lo_mult:.2f} – {hi_mult:.2f})",
             ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("F4_12_PDP_APAC.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("⚠️ Variabile APAC non disponibile nel dataset.")
