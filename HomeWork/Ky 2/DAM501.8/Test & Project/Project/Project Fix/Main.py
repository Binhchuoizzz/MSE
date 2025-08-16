# -*- coding: utf-8 -*-
"""
Market Basket Analysis (Online Retail I/II) — Memory-Safe Version
- Filters to top-K frequent items OR by min item support before Apriori
- Uses max_len=2 and low_memory=True for Apriori
- Stick to assignment: odd -> Apriori; even -> FP-Max + FP-Growth (also max_len=2)

Run:
    python mba_online_retail_memorysafe.py
"""

import os, re
from typing import List, Tuple
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
FILE_PATH    = r"E:\MSE\HomeWork\Ky 2\DAM501.8\Test & Project\Project\Project Fix\online_retail_II.csv"
ITEM_MODE    = "stockcode"            # "stockcode" recommended (fewer unique items) | "description"
ID_PARITY    = "even"                  # "odd" -> Apriori | "even" -> FP-Max + FP-Growth
SUPPORT_GRID = [0.003, 0.002, 0.001, 0.0008, 0.0005]
CONF_GRID    = [0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]
MIN_LIFT     = 1.0
TARGET_RANGE = (10, 20)
OUTDIR       = "out"

# Item filtering to avoid OOM
USE_TOPK         = True
TOPK_ITEMS       = 600                 # keep only top 600 frequent items
MIN_ITEM_SUPPORT = 0.002               # used if USE_TOPK=False (e.g., keep items with >=0.2% support)

MAX_LEN = 2                            # only pairs for bundles/combos (sufficient for cross-sell)
# ====================================================

def step(msg: str): print(f"\n🔹 {msg}")

def choose_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns: return c
    raise KeyError(f"None of the columns {candidates} found. Available: {list(df.columns)}")

def normalize_text(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', '').replace("'", "")
    return s

def load_and_clean(path: str):
    step("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    invoice_col = choose_col(df, ["InvoiceNo", "Invoice"])
    desc_col    = choose_col(df, ["Description"])
    code_col    = choose_col(df, ["StockCode", "Stock Code"])
    qty_col     = choose_col(df, ["Quantity"])
    price_col   = choose_col(df, ["UnitPrice", "Price"])

    df = df[[invoice_col, desc_col, code_col, qty_col, price_col]].dropna()
    df[invoice_col] = df[invoice_col].astype(str)
    df = df[~df[invoice_col].str.startswith("C")]
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)].copy()

    if ITEM_MODE == "stockcode":
        df["ITEM"] = df[code_col].astype(str).str.strip()
    else:
        df["ITEM"] = df[desc_col].astype(str).map(normalize_text)

    df = df[[invoice_col, "ITEM"]].dropna().drop_duplicates()

    sizes = df.groupby(invoice_col)["ITEM"].nunique()
    keep_ids = sizes.index[sizes >= 2]
    df = df[df[invoice_col].isin(keep_ids)].copy()

    print(f"✅ After cleaning: {df[invoice_col].nunique():,} baskets | {len(df):,} rows | {df['ITEM'].nunique():,} unique items")
    return df, invoice_col

def build_basket(df: pd.DataFrame, invoice_col: str):
    step("Building transactions & one-hot encoding...")
    transactions = df.groupby(invoice_col)["ITEM"].apply(lambda s: sorted(set(s))).tolist()
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=te.columns_)
    print(f"✅ Encoded basket (pre-filter): {basket.shape[0]} tx × {basket.shape[1]} items")

    # ---- Item filtering to avoid OOM ----
    item_support = basket.mean(axis=0)  # fraction of transactions containing the item
    if USE_TOPK:
        top_items = item_support.sort_values(ascending=False).head(TOPK_ITEMS).index
        basket = basket[top_items]
        print(f"✅ Kept top-{TOPK_ITEMS} items by support → now {basket.shape[1]} items")
    else:
        keep = item_support[item_support >= MIN_ITEM_SUPPORT].index
        basket = basket[keep]
        print(f"✅ Kept items with support ≥ {MIN_ITEM_SUPPORT} → now {basket.shape[1]} items")

    return basket

def mine_itemsets_apriori(basket: pd.DataFrame, min_support: float):
    return apriori(basket, min_support=min_support, use_colnames=True, max_len=MAX_LEN, low_memory=True)

def mine_itemsets_fpgrowth(basket: pd.DataFrame, min_support: float):
    return fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=MAX_LEN)

def rules_from_itemsets(freq: pd.DataFrame, min_conf: float, min_lift: float):
    if freq is None or freq.empty: return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if rules.empty: return rules
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1)].copy()
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules

def auto_tune(basket: pd.DataFrame, algo: str):
    low, high = TARGET_RANGE
    best = (None, None, None, None, -1)
    for sup in SUPPORT_GRID:
        if algo == "apriori":
            freq = mine_itemsets_apriori(basket, sup)
        else:
            freq = mine_itemsets_fpgrowth(basket, sup)
        if freq is None or freq.empty: continue
        for conf in CONF_GRID:
            rules = rules_from_itemsets(freq, conf, MIN_LIFT)
            n = 0 if rules is None else len(rules)
            if n == 0: continue
            if low <= n <= high: return freq, rules, sup, conf
            if n > best[-1]: best = (freq, rules, sup, conf, n)
    return best[0], best[1], best[2], best[3]

def save_all(freq, rules, parity):
    os.makedirs(OUTDIR, exist_ok=True)
    (freq if freq is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "frequent_itemsets.csv"), index=False)
    (rules if rules is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "association_rules.csv"), index=False)
    if rules is not None and not rules.empty:
        rules.sort_values(["lift","confidence"], ascending=False).head(20).to_csv(os.path.join(OUTDIR, "top_rules.csv"), index=False)
    print(f"\n📦 Saved CSVs to ./{OUTDIR}/")

def plot_scatter(rules):
    if rules is None or rules.empty: return
    plt.figure(figsize=(7,5))
    sizes = (rules["lift"] * 40).clip(10, 400)
    plt.scatter(rules["support"], rules["confidence"], s=sizes, alpha=0.6)
    plt.title("Association Rules: support vs confidence (size ~ lift)")
    plt.xlabel("support"); plt.ylabel("confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "rules_scatter.png"), dpi=150)
    print(" - rules_scatter.png")

def main():
    df, invoice_col = load_and_clean(FILE_PATH)
    basket = build_basket(df, invoice_col)

    if basket.shape[1] < 2:
        print("❗Not enough items after filtering. Reduce filtering or increase TOPK_ITEMS.")
        return

    if ID_PARITY == "even":
        # FP-Max for reporting (optional, at a sane support)
        sup_for_max = max(0.001, 5 / basket.shape[0])
        freq_max = fpmax(basket, min_support=sup_for_max, use_colnames=True, max_len=MAX_LEN)
        os.makedirs(OUTDIR, exist_ok=True)
        (freq_max if freq_max is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "frequent_itemsets_maximal.csv"), index=False)
        print(f"✅ FP-Max saved ({0 if freq_max is None else len(freq_max)} rows).")
        algo = "fpgrowth"
    else:
        algo = "apriori"

    print(f"\n▶ Using algorithm: {algo.upper()} (MAX_LEN={MAX_LEN}, ITEM_MODE={ITEM_MODE}, TOPK_ITEMS={TOPK_ITEMS if USE_TOPK else 'N/A'})")
    freq, rules, sup, conf = auto_tune(basket, algo)

    if freq is None or rules is None or rules.empty:
        print("❗No positive rules (lift>=1) within configured ranges. Try increasing TOPK_ITEMS or lowering SUPPORT/CONF grids.")
        return

    print(f"\n⚙️ Best params -> min_support={sup}, min_confidence={conf}")
    print(f"Frequent itemsets: {len(freq)} | Positive rules: {len(rules)}")

    save_all(freq, rules, ID_PARITY)
    plot_scatter(rules)

    print("\n📊 Top 10 rules:")
    cols = ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
    print(rules.sort_values(['lift','confidence'], ascending=False)[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
