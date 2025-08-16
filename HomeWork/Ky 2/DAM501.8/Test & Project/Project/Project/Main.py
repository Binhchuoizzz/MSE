# -*- coding: utf-8 -*-
"""
Market Basket Analysis – Assignment Version
- ID lẻ  : Apriori -> Association Rules
- ID chẵn: FP-Max (maximal patterns) + FP-Growth -> Association Rules
Output: out/frequent_itemsets.csv, out/association_rules.csv,
        (ID chẵn) out/frequent_itemsets_maximal.csv
  |  Requires: pandas, mlxtend
"""

import os
import math
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

# ========= CONFIG =========
FILE_PATH       = r"E:\MSE\HomeWork\Ky 2\DAM501.8\Test & Project\Project\Project\new_retail_data.csv"
INVOICE_COL     = "Transaction_ID"     # hóa đơn
ITEM_COL        = "Product_Category"   # mặt hàng / group
ID_PARITY       = "even"                # "odd" = ID lẻ (Apriori) | "even" = ID chẵn (FP-Max + FP-Growth)
USE_MULTI_ONLY  = True                 # Bật USE_MULTI_ONLY=True để chỉ giữ hoá đơn có ≥2 mặt hàng
TARGET_RANGE    = (50, 300)            # mục tiêu số lượng luật để thuyết trình
MIN_LIFT        = 1.0                  # nới 1.0 cho dataset thưa (có thể tăng 1.1+ nếu nhiều luật)
OUTDIR          = "out"
# ==========================

def step(msg): print(f"\n🔹 {msg}")

def load_and_clean(path: str, inv: str, item: str):
    step("Step 1: Loading dataset...")
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    step("Step 2: Cleaning & standardizing...")
    df = df[[inv, item]].dropna().copy()
    # ✅ dùng .str.strip() thay vì .strip()
    df[item] = df[item].astype("string").str.strip().str.upper()
    # 1 invoice – 1 item (loại trùng)
    df = df.drop_duplicates(subset=[inv, item])
    sizes = df.groupby(inv)[item].nunique()
    multi, total = (sizes >= 2).sum(), sizes.shape[0]
    print(f"✅ After clean: {len(df):,} rows | Invoices ≥2 items: {multi}/{total}")
    return df, sizes


def build_basket(df: pd.DataFrame, inv: str, item: str, multi_only: bool):
    step("Step 3: Building transactions...")
    if multi_only:
        keep_ids = df.groupby(inv)[item].nunique()
        keep_ids = keep_ids.index[keep_ids >= 2]
        df = df[df[inv].isin(keep_ids)].copy()

    transactions = (
        df.groupby(inv)[item]
          .apply(lambda s: sorted(set(map(str, s))))
          .tolist()
    )
    print(f"✅ Built {len(transactions):,} transactions.")

    step("Step 4: Encoding transactions (one-hot)...")
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=te.columns_)
    print(f"✅ Encoded matrix: {basket.shape[0]} rows (transactions), {basket.shape[1]} items.")
    return basket

def mine_itemsets(basket: pd.DataFrame, algo: str, min_support: float):
    if basket.empty:
        return pd.DataFrame()
    if algo == "apriori":
        return apriori(basket, min_support=min_support, use_colnames=True)
    elif algo == "fpgrowth":
        return fpgrowth(basket, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("algo must be 'apriori' or 'fpgrowth'.")

def make_rules(freq: pd.DataFrame, min_conf: float, min_lift: float):
    if freq is None or freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    base = rules.copy()
    rules = base[base["lift"] >= min_lift].copy()
    if rules.empty:
        rules = base
        return rules
    # chỉ để consequent 1 item cho dễ trình bày
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1)].copy()
    # làm đẹp
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    # lọc lift
    rules = rules[rules["lift"] >= min_lift].copy()
    return rules

def dynamic_grids(n_tx: int):
    """
    Lưới support dựa trên 'số lần xuất hiện tối thiểu' để phù hợp với số giỏ hiện có.
    """
    min_counts = [200,150,120,100,80,60,50,40,30,25,20,15,10,8,6,5,4,3,2,1]
    sup_grid = sorted({max(c / n_tx, 1 / n_tx) for c in min_counts}, reverse=True)
    #bắt buộc có luật - Dữ liệu cực thưa (rất ít hoá đơn có ≥2 món, chỉ 5 category), nên mọi cặp chỉ xuất hiện lác đác → confidence của các luật < 0.1 ⇒ 0 rules là bình thường
    conf_grid = [0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.02,0.01]

    #conf_grid = [0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]
    return sup_grid, conf_grid

def auto_tune_and_mine(basket: pd.DataFrame, algo: str, min_lift: float):
    """
    Tự động tìm (support, confidence) cho số luật nằm trong TARGET_RANGE.
    Nếu không đạt, chọn cấu hình gần nhất với biên dưới để luật 'gọn'.
    """
    low, high = TARGET_RANGE
    sup_grid, conf_grid = dynamic_grids(basket.shape[0])

    best = (None, None, None, None, math.inf)  # freq, rules, sup, conf, score
    for sup in sup_grid:
        freq = mine_itemsets(basket, algo, sup)
        if freq is None or freq.empty:
            continue
        freq["length"] = freq["itemsets"].apply(len)
        for conf in conf_grid:
            rules = make_rules(freq, conf, min_lift)
            n = 0 if rules is None else len(rules)
            if n == 0:
                continue
            score = abs(n - low)
            if (low <= n <= high) or score < best[-1]:
                best = (freq, rules, sup, conf, score)
                if low <= n <= high:
                    return best[0], best[1], best[2], best[3]
    return best[0], best[1], best[2], best[3]

def save_and_report(freq: pd.DataFrame, rules: pd.DataFrame, parity: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    if freq is not None and not freq.empty:
        freq.to_csv(os.path.join(outdir, "frequent_itemsets.csv"), index=False)
    else:
        pd.DataFrame(columns=["itemsets","support","length"]).to_csv(os.path.join(outdir, "frequent_itemsets.csv"), index=False)

    if rules is not None and not rules.empty:
        rules.sort_values(["lift","confidence"], ascending=False)\
             .to_csv(os.path.join(outdir, "association_rules.csv"), index=False)
    else:
        pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift","leverage","conviction"]).to_csv(
            os.path.join(outdir, "association_rules.csv"), index=False
        )

    print(f"\n📦 Output saved in '{outdir}/'")
    print(" - frequent_itemsets.csv")
    print(" - association_rules.csv")
    if parity == "even":
        print(" - frequent_itemsets_maximal.csv (FP-Max)")

def main():
    df, sizes = load_and_clean(FILE_PATH, INVOICE_COL, ITEM_COL)

    basket = build_basket(df, INVOICE_COL, ITEM_COL, USE_MULTI_ONLY)
    n_tx, n_items = basket.shape
    if n_tx == 0 or n_items < 2:
        print("❗Không đủ dữ liệu để khai thác. Hãy đặt USE_MULTI_ONLY=False hoặc dùng dữ liệu có nhiều item hơn.")
        return

    if ID_PARITY == "odd":
        print("\n▶ Thuật toán: APRIORI (ID lẻ – theo đề)")
        freq, rules, sup, conf = auto_tune_and_mine(basket, "apriori", MIN_LIFT)
        if freq is None or rules is None:
            # fallback: support & conf thấp nhất
            sup, conf = dynamic_grids(n_tx)[0][-1], dynamic_grids(n_tx)[1][-1]
            print("⚠️ Auto-tune không đạt range mục tiêu → dùng fallback.")
            freq = mine_itemsets(basket, "apriori", sup)
            if freq is not None and not freq.empty:
                freq["length"] = freq["itemsets"].apply(len)
            rules = make_rules(freq, conf, MIN_LIFT)

        print(f"min_support={sup:.6f} (~{round(sup*n_tx)} tx) | min_confidence={conf}")
        print(f"Frequent itemsets: {0 if freq is None else len(freq):,} | Association rules: {0 if rules is None else len(rules):,}")
        save_and_report(freq, rules, "odd", OUTDIR)

        if rules is not None and not rules.empty:
            cols = ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
            print("\nTop 10 rules by lift:")
            print(rules.sort_values(["lift","confidence"], ascending=False)[cols].head(10).to_string(index=False))
        else:
            print("❗Chưa có luật. Giảm thêm ngưỡng (min_lift↑ khó, min_conf↓, hoặc đặt USE_MULTI_ONLY=False).")

    else:
        print("\n▶ Thuật toán: FP-MAX (maximal) + FP-GROWTH (ID chẵn – theo đề)")
        # 1) FP-Max (để có tập tối đại)
        sup_for_max = max(1 / n_tx, 5 / n_tx)  # đừng quá thấp để tránh quá nhiều pattern trùng
        step("Step 5a: Mining FP-Max (maximal frequent itemsets)...")
        freq_max = fpmax(basket, min_support=sup_for_max, use_colnames=True)
        if freq_max is not None and not freq_max.empty:
            freq_max["length"] = freq_max["itemsets"].apply(len)
        os.makedirs(OUTDIR, exist_ok=True)
        (freq_max if freq_max is not None else pd.DataFrame(columns=["itemsets","support","length"]))\
            .to_csv(os.path.join(OUTDIR, "frequent_itemsets_maximal.csv"), index=False)
        print(f"✅ FP-Max: {0 if freq_max is None else len(freq_max):,} maximal itemsets saved.")

        # 2) FP-Growth (để sinh luật)
        step("Step 5b: Mining FP-Growth (full frequent itemsets) for rules...")
        freq, rules, sup, conf = auto_tune_and_mine(basket, "fpgrowth", MIN_LIFT)
        if freq is None or rules is None:
            sup, conf = dynamic_grids(n_tx)[0][-1], dynamic_grids(n_tx)[1][-1]
            print("⚠️ Auto-tune không đạt range mục tiêu → dùng fallback.")
            freq = mine_itemsets(basket, "fpgrowth", sup)
            if freq is not None and not freq.empty:
                freq["length"] = freq["itemsets"].apply(len)
            rules = make_rules(freq, conf, MIN_LIFT)

        print(f"min_support={sup:.6f} (~{round(sup*n_tx)} tx) | min_confidence={conf}")
        print(f"Frequent itemsets: {0 if freq is None else len(freq):,} | Association rules: {0 if rules is None else len(rules):,}")
        save_and_report(freq, rules, "even", OUTDIR)

        if rules is not None and not rules.empty:
            cols = ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
            print("\nTop 10 rules by lift:")
            print(rules.sort_values(["lift","confidence"], ascending=False)[cols].head(10).to_string(index=False))
        else:
            print("❗Chưa có luật. Giảm thêm ngưỡng (min_conf↓) hoặc tắt USE_MULTI_ONLY để tăng mẫu.")

if __name__ == "__main__":
    main()
