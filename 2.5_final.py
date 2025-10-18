# -*- coding: utf-8 -*-
"""
EDA riproducibile per Vestiaire Collective (fix mix categoria×Paese + map country names)
- Caricamento CSV (snippet richiesto)
- Pulizia robusta del prezzo formattato
- Mappatura Paese "nome esteso" -> ISO-2 per macro-aree
- Statistiche pre/post-log
- Macro-aree, Spearman, concentrazione like, brand>=2000
- Mix categoria×Paese con calcolo percentuali senza reset_index conflittosi
- Esportazione risultati in eda_outputs/
Autore: <tuo nome> | Data: 2025-10-17
"""

import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------ CONFIG ------------------------------------ #
csv_path = r"C:\Users\Mattia\PycharmProjects\PythonProject\data\vestiaire.csv"
print(f"Carico il file da: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"⚠️  Il file non è stato trovato a: {csv_path}")

OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_PRICE_COLS = ["price_usd", "price_eur", "listing_price", "price", "seller_price"]
CANDIDATE_LIKES_COLS = ["likes_count", "product_like_count", "like_count"]
CANDIDATE_BRAND_COLS = ["brand_name", "brand", "item_brand", "seller_brand", "brand_id"]
CANDIDATE_CAT_COLS   = ["category", "product_category", "sub_category", "product_type"]
CANDIDATE_COND_COLS  = ["item_condition", "product_condition", "condition"]
CANDIDATE_COUNTRY_COLS = ["seller_country", "country", "seller_country_code"]

# Gruppi macro-area (ISO-2)
EUROPE = {
    "AL","AD","AT","BY","BE","BA","BG","HR","CY","CZ","DK","EE","FO","FI","FR",
    "DE","GI","GR","HU","IS","IE","IT","XK","LV","LI","LT","LU","MT","MD","MC",
    "ME","NL","MK","NO","PL","PT","RO","RU","SM","RS","SK","SI","ES","SE","CH",
    "UA","GB","VA"
}
NORTH_AMERICA = {"US","CA","MX"}
APAC = {
    "AE","AF","AM","AU","AZ","BD","BH","BN","BT","CN","GE","HK","ID","IL","IN",
    "IQ","IR","JP","JO","KG","KH","KP","KR","KW","KZ","LA","LB","LK","MM","MN",
    "MO","MV","MY","NP","NZ","OM","PH","PK","QA","SA","SG","SY","TH","TJ","TM",
    "TW","UZ","VN","YE"
}

# Mappa "nome paese" -> ISO-2 (espandibile in base ai tuoi dati)
COUNTRY_NAME_TO_ISO2 = {
    "germany": "DE", "deutschland": "DE",
    "belgium": "BE", "belgië": "BE", "belgique": "BE",
    "spain": "ES", "españa": "ES",
    "france": "FR",
    "italy": "IT", "italia": "IT",
    "united states": "US", "usa": "US", "u.s.a.": "US", "united states of america": "US",
    "united kingdom": "GB", "uk": "GB", "u.k.": "GB", "great britain": "GB",
    "portugal": "PT",
    "netherlands": "NL", "the netherlands": "NL", "holland": "NL",
    "switzerland": "CH", "schweiz": "CH", "suisse": "CH", "svizzera": "CH",
    "austria": "AT",
    "ireland": "IE",
    "denmark": "DK",
    "sweden": "SE",
    "norway": "NO",
    "finland": "FI",
    "poland": "PL",
    "czech republic": "CZ", "czechia": "CZ",
    "slovakia": "SK",
    "slovenia": "SI",
    "croatia": "HR",
    "greece": "GR",
    "romania": "RO",
    "bulgaria": "BG",
    "hungary": "HU",
    "estonia": "EE",
    "latvia": "LV",
    "lithuania": "LT",
    "luxembourg": "LU",
    "malta": "MT",
    "iceland": "IS",
    "andorra": "AD",
    "monaco": "MC",
    "san marino": "SM",
    "vatican city": "VA", "holy see": "VA",
    "canada": "CA",
    "mexico": "MX",
    "japan": "JP",
    "hong kong": "HK",
    "china": "CN",
    "taiwan": "TW",
    "south korea": "KR", "korea, republic of": "KR",
    "singapore": "SG",
    "australia": "AU",
    "new zealand": "NZ",
    "united arab emirates": "AE", "uae": "AE",
    "qatar": "QA",
    "saudi arabia": "SA",
    "turkey": "TR",
    "russia": "RU",
}

HARD_LUXURY_KEYWORDS = [
    "bag", "bors", "sac", "handbag", "tote", "backpack",
    "watch", "orolog", "montre",
    "jewel", "gioiell", "ring", "earring", "necklace", "bracelet"
]

# ------------------------------ IO ---------------------------------------- #
print("🟡 Lettura CSV in corso...")
df = pd.read_csv(csv_path)
print(f"✅ CSV caricato. Righe: {len(df):,} | Colonne: {len(df.columns)}")
print("Colonne disponibili:", list(df.columns))

def pick_col(df, candidates, label):
    found = [c for c in candidates if c in df.columns]
    if not found:
        raise KeyError(
            f"❌ Nessuna colonna trovata per {label}. "
            f"Attese una tra: {candidates}. "
            f"Colonne presenti: {list(df.columns)}"
        )
    if len(found) > 1:
        print(f"ℹ️  Colonne multiple per {label} trovate {found}. Uso '{found[0]}'.")
    return found[0]

price_col   = pick_col(df, CANDIDATE_PRICE_COLS, "prezzo")
likes_col   = pick_col(df, CANDIDATE_LIKES_COLS, "likes")
brand_col   = pick_col(df, CANDIDATE_BRAND_COLS, "brand")
cat_col     = pick_col(df, CANDIDATE_CAT_COLS, "category")
cond_col    = pick_col(df, CANDIDATE_COND_COLS, "item_condition")
country_col = pick_col(df, CANDIDATE_COUNTRY_COLS, "seller_country")

print("\n🔎 Colonne utilizzate:")
print(f" - Prezzo:          {price_col}")
print(f" - Likes:           {likes_col}")
print(f" - Brand:           {brand_col}")
print(f" - Categoria:       {cat_col}")
print(f" - Condizione:      {cond_col}")
print(f" - Paese venditore: {country_col}")

print("\n🔬 Esempi grezzi (prime 5) — prezzo e paese:")
print(df[price_col].astype(str).head().to_string(index=False))
print(df[country_col].astype(str).head().to_string(index=False))

# ------------------------------ CLEANING ---------------------------------- #
# 1) Pulizia del prezzo
_price = df[price_col].astype(str)

def to_float_price(x: str) -> float:
    m = re.search(r'[-+]?[0-9][0-9.,]*', x)
    if not m:
        return np.nan
    s = m.group(0)
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return np.nan

df["_price_clean"] = _price.map(to_float_price)
use_price = "_price_clean"

# 2) Likes numerici
df[likes_col] = pd.to_numeric(df[likes_col], errors="coerce").fillna(0).astype(int)

# 3) Paese: normalizza a ISO-2 dove possibile (non scartiamo se non è ISO-2)
def normalize_country(val: str) -> str:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    key = s.lower()
    return COUNTRY_NAME_TO_ISO2.get(key, s)  # se non mappato, restituisce il nome originale

df["_seller_country_norm"] = df[country_col].map(normalize_country)

# 4) Filtro base: prezzo > 0 e paese non nullo
df = df[(df[use_price] > 0) & (df["_seller_country_norm"].notna())].copy()
print(f"\n✅ Dopo cleaning base: Righe valide: {len(df):,}")

# ------------------------------ FEATURE ----------------------------------- #
df["log_price"] = np.log10(df[use_price].astype(float))
df["log1p_likes"] = np.log1p(df[likes_col].astype(float))

def map_area(cc):
    if pd.isna(cc):
        return "Other"
    s = str(cc).strip()
    if len(s) == 2 and s.isalpha():
        s = s.upper()
        if s in EUROPE:         return "Europe"
        if s in NORTH_AMERICA:  return "North America"
        if s in APAC:           return "Asia-Pacific"
        return "Other"
    # nomi estesi non mappati restano Other
    return "Other"

df["macro_area"] = df["_seller_country_norm"].map(map_area)

def is_hard_luxury(txt):
    if pd.isna(txt):
        return False
    t = str(txt).lower()
    return any(kw in t for kw in HARD_LUXURY_KEYWORDS)

df["is_hard_luxury"] = df[cat_col].apply(is_hard_luxury)

# ------------------------------ ANALISI ----------------------------------- #
log_lines = []
def logprint(s):
    print(s); log_lines.append(s)

logprint("\n=== DIMENSIONE DATASET ANALITICO ===")
logprint(f"Righe valide: {len(df):,}")
logprint(f"Colonne disponibili: {len(df.columns)}")
logprint(f"Colonna prezzo usata (pulita): {use_price}")

# Statistiche pre/post-log
def robust_summary(series, name):
    desc = {
        "count": float(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "skew": float(stats.skew(series, bias=False)),
        "kurtosis": float(stats.kurtosis(series, fisher=True, bias=False) + 3)
    }
    return pd.Series(desc, name=name)

summ_price = robust_summary(df[use_price].astype(float), "price_clean")
summ_log   = robust_summary(df["log_price"], "log_price")

logprint("\n=== STATISTICHE PREZZO (PULITO) ===")
logprint(summ_price.to_string())

logprint("\n=== STATISTICHE PREZZO (LOG10) ===")
logprint(summ_log.to_string())

med_log = summ_log["median"]
iqr_low = summ_log["p25"]
iqr_hi  = summ_log["p75"]
med_orig = 10**med_log
iqr_low_orig = 10**iqr_low
iqr_hi_orig  = 10**iqr_hi

logprint("\n=== INTERPRETAZIONE (LOG10) IN METRICA ORIGINALE ===")
logprint(f"Mediana log10 = {med_log:.3f}  → mediana ≈ {med_orig:,.2f}")
logprint(f"IQR log10 = [{iqr_low:.3f}, {iqr_hi:.3f}] → IQR ≈ [{iqr_low_orig:,.2f}, {iqr_hi_orig:,.2f}]")

# Macro-aree: media log-prezzo
area_stats = (
    df.groupby("macro_area", dropna=False)["log_price"]
      .agg(["count","mean","std"])
      .sort_values("mean", ascending=False)
)
logprint("\n=== LOG-PREZZO PER MACRO-AREA ===")
logprint(area_stats.to_string())

def area_diff(a, b):
    if a in area_stats.index and b in area_stats.index:
        return area_stats.loc[a, "mean"] - area_stats.loc[b, "mean"]
    return np.nan

d_apac_eu = area_diff("Asia-Pacific", "Europe")
d_na_eu   = area_diff("North America", "Europe")

logprint("\nDifferenze di media log_price (macro-aree):")
logprint(f"Asia-Pacific vs Europe: {d_apac_eu:.3f}" if not np.isnan(d_apac_eu) else "n/d")
logprint(f"North America vs Europe: {d_na_eu:.3f}" if not np.isnan(d_na_eu) else "n/d")

# Spearman
rho_all, p_all = stats.spearmanr(df["log1p_likes"], df["log_price"], nan_policy="omit")

# Subset hard-luxury (controllo dimensione)
hl = df.loc[df["is_hard_luxury"], ["log1p_likes", "log_price"]].dropna()
if len(hl) >= 10 and hl["log1p_likes"].nunique() >= 3 and hl["log_price"].nunique() >= 3:
    rho_hl, p_hl = stats.spearmanr(hl["log1p_likes"], hl["log_price"], nan_policy="omit")
else:
    rho_hl, p_hl = np.nan, np.nan

logprint("\n=== CORRELAZIONE SPEARMAN log1p(likes) ~ log_price ===")
logprint(f"Tutto il campione: rho = {rho_all:.3f}, p-value = {p_all:.3g}")
logprint(f"Hard-luxury:       rho = {rho_hl if not np.isnan(rho_hl) else 'n/d'}, "
         f"p-value = {p_hl if not np.isnan(p_hl) else 'n/d'}")

# Concentrazione like
df_sorted_likes = df.sort_values(likes_col, ascending=False)
top10_cut = max(1, int(math.ceil(0.10 * len(df_sorted_likes))))
likes_total = df_sorted_likes[likes_col].sum()
likes_top10 = df_sorted_likes.iloc[:top10_cut][likes_col].sum()
share_top10 = (likes_top10 / likes_total) if likes_total > 0 else np.nan

logprint("\n=== CONCENTRAZIONE LIKE ===")
logprint(
    f"Top 10% annunci raccoglie ≈ {share_top10*100:,.2f}% dei like totali"
    if not np.isnan(share_top10)
    else "Like totali = 0: impossibile calcolare la concentrazione."
)

# Brand >= 2000
brand_counts = df[brand_col].astype(str).value_counts(dropna=False)
brands_ge_2000 = brand_counts[brand_counts >= 2000].index.tolist()

brand_stats = (
    df[df[brand_col].isin(brands_ge_2000)]
    .groupby(brand_col)["log_price"]
    .agg(count="count", mean="mean", std="std")
    .sort_values("mean", ascending=False)
)

logprint("\n=== BRAND CON n ≥ 2000: MEDIA LOG-PREZZO (prime 20 righe) ===")
if len(brand_stats) == 0:
    logprint("Nessun brand con n ≥ 2000.")
else:
    logprint(brand_stats.head(20).to_string())

# Mix categoria × Paese (prime 10 nazioni per numerosità) — versione senza reset_index conflittosi
top_countries = df["_seller_country_norm"].value_counts().head(10).index.tolist()

mix_counts = (
    df[df["_seller_country_norm"].isin(top_countries)]
    .groupby(["_seller_country_norm", cat_col], dropna=False)
    .size()
    .rename("count")
    .to_frame()
)

totals = (
    mix_counts
    .groupby(level=0)["count"]
    .sum()
    .rename("total")
    .to_frame()
)

mix = mix_counts.join(totals, on="_seller_country_norm")
mix["pct"] = (mix["count"] / mix["total"]) * 100.0
mix = mix.reset_index(names=["seller_country_iso2", "category_name"])

print("\n=== MIX CATEGORIA × PAESE (prime 10 nazioni per numerosità) ===")
for cc in top_countries:
    sub = mix[mix["seller_country_iso2"] == cc].sort_values("pct", ascending=False).head(10)
    print(f"\n- {cc}: top categorie (% su paese)")
    print(sub[["seller_country_iso2", "category_name", "pct"]].to_string(index=False))

# ------------------------------ EXPORT ------------------------------------ #
summ_df = pd.concat([summ_price, summ_log], axis=1)
summ_df.columns = ["price_clean", "price_log10"]
summ_df.to_csv(OUTPUT_DIR / "summary_price_pre_post_log.csv", index=True)
area_stats.to_csv(OUTPUT_DIR / "macro_area_logprice_stats.csv", index=True)
brand_stats.to_csv(OUTPUT_DIR / "brand_logprice_ge2000.csv", index=True)
mix.to_csv(OUTPUT_DIR / "mix_category_by_country_top10.csv", index=False)

with open(OUTPUT_DIR / "eda_printout.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print(f"\n✅ Esportazioni completate in: {OUTPUT_DIR.resolve()}")
