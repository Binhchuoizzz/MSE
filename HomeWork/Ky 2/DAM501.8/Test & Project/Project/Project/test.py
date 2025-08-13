# file: market_basket_miner.py
# -------------------------------------------------------
# Yêu cầu: pip install pandas mlxtend
# Chạy:    python market_basket_miner.py
# Đầu vào: new_retail_data.csv (cùng thư mục)
# Đầu ra:  out/frequent_itemsets.csv
#          out/association_rules.csv
#          (thêm) out/frequent_itemsets_maximal.csv nếu ID chẵn
# -------------------------------------------------------

import os
import math
import pandas as pd
from typing import List, Tuple
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

# === 0) CẤU HÌNH THEO BÀI ===
FILE_PATH = "new_retail_data.csv"
ID_PARITY = "odd"   # "odd"  = ID lẻ -> Apriori
# "even" = ID chẵn -> FP-Growth (+ FP-Max)
TARGET_RULES_RANGE = (50, 300)  # mục tiêu số lượng luật
INITIAL_SUPPORT_GRID = [0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.003]
CONFIDENCE_GRID = [0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
MIN_LIFT_FILTER = 1.1        # lọc luật nâng lift > 1.1 để chất lượng hơn

# === 1) HÀM TIỆN ÍCH ===


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Tìm cột theo danh sách tên gợi ý (case-insensitive). Trả về '' nếu không thấy."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    # fuzzy: thử chứa chuỗi
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return ""


def map_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    invoice_cands = ["InvoiceNo", "Invoice", "InvoiceID",
                     "OrderNo", "OrderID", "BillNo", "Transaction", "Receipt"]
    item_cands = ["Description", "ItemDescription", "Item",
                  "Product", "ProductName", "SKU", "ItemName", "StockCode"]
    qty_cands = ["Quantity", "Qty", "Count", "Units", "Unit"]
    price_cands = ["UnitPrice", "Price", "Unit_Price"]
    date_cands = ["InvoiceDate", "Date", "OrderDate", "Timestamp", "Datetime"]

    invoice_col = _find_col(df, invoice_cands)
    item_col = _find_col(df, item_cands)
    qty_col = _find_col(df, qty_cands)
    price_col = _find_col(df, price_cands)
    date_col = _find_col(df, date_cands)

    if not invoice_col or not item_col:
        raise ValueError(
            f"Không tìm thấy cột bắt buộc. "
            f"Invoice candidates={invoice_cands}, Item candidates={item_cands}. "
            f"Các cột hiện có: {list(df.columns)}"
        )
    return invoice_col, item_col, qty_col, price_col, date_col


def clean_df(df: pd.DataFrame, invoice_col: str, item_col: str, qty_col: str, price_col: str) -> pd.DataFrame:
    # bỏ thiếu invoice / item
    df = df.dropna(subset=[invoice_col, item_col])

    # chuẩn hóa item
    df[item_col] = df[item_col].astype(str).str.strip().str.upper()

    # bỏ trả hàng, invoice âm... nếu có pattern 'C' đầu hóa đơn (theo OnlineRetail)
    inv_str = df[invoice_col].astype(str)
    df = df[~inv_str.str.startswith("C")]

    # lọc số lượng > 0 nếu có
    if qty_col and qty_col in df.columns:
        df = df[pd.to_numeric(df[qty_col], errors="coerce").fillna(0) > 0]

    # lọc giá > 0 nếu có
    if price_col and price_col in df.columns:
        df = df[pd.to_numeric(df[price_col], errors="coerce").fillna(0) > 0]

    # loại các item nhiễu thường gặp
    ban_list = {"POSTAGE", "DOTCOM POSTAGE", "SAMPLES", "CARRIAGE"}
    df = df[~df[item_col].isin(ban_list)]

    return df


def build_basket(df: pd.DataFrame, invoice_col: str, item_col: str) -> pd.DataFrame:
    # group theo hóa đơn → list các item (unique)
    transactions = (
        df.groupby(invoice_col)[item_col]
          .apply(lambda s: sorted(set(map(str, s))))
          .tolist()
    )
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=te.columns_)
    return basket


def mine_itemsets(basket: pd.DataFrame, algo: str, min_support: float) -> pd.DataFrame:
    if algo == "apriori":
        freq = apriori(basket, min_support=min_support, use_colnames=True)
    elif algo == "fpgrowth":
        freq = fpgrowth(basket, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("algo phải là 'apriori' hoặc 'fpgrowth'")
    freq["length"] = freq["itemsets"].str.len()
    return freq


def rules_from_itemsets(freq: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(
        freq, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return rules
    rules["antecedent_len"] = rules["antecedents"].apply(lambda s: len(s))
    rules["consequent_len"] = rules["consequents"].apply(lambda s: len(s))
    # chỉ giữ consequent 1 item cho dễ diễn giải
    rules = rules[rules["consequent_len"] == 1]
    # làm đẹp text
    rules["antecedents"] = rules["antecedents"].apply(
        lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(
        lambda s: ", ".join(sorted(list(s))))
    # lọc lift
    rules = rules[rules["lift"] > MIN_LIFT_FILTER]
    return rules


def auto_tune(basket: pd.DataFrame, algo: str) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Tự thử nhiều min_support & min_conf để số luật nằm trong TARGET_RULES_RANGE."""
    best = (None, None, None, None,
            math.inf)  # (freq, rules, sup, conf, distance)
    target_low, target_high = TARGET_RULES_RANGE

    for sup in INITIAL_SUPPORT_GRID:
        freq = mine_itemsets(basket, algo, sup)
        if freq.empty:
            continue
        for conf in CONFIDENCE_GRID:
            rules = rules_from_itemsets(freq, conf)
            n = len(rules)
            if n == 0:
                continue
            # đánh giá theo gần biên dưới để có bộ luật gọn
            dist = abs(n - target_low)
            if (target_low <= n <= target_high) or dist < best[-1]:
                best = (freq, rules, sup, conf, dist)
                if target_low <= n <= target_high:
                    return best[0], best[1], best[2], best[3]
    # không trúng range thì trả về cấu hình gần nhất
    return best[0], best[1], best[2], best[3]


def ensure_outdir(path="out"):
    os.makedirs(path, exist_ok=True)
    return path

# === 2) MAIN FLOW ===


def main():
    print("==> Đang đọc dữ liệu:", FILE_PATH)
    df = pd.read_csv(FILE_PATH, low_memory=False)

    invoice_col, item_col, qty_col, price_col, date_col = map_columns(df)
    print(
        f"Ánh xạ cột → Invoice: {invoice_col} | Item: {item_col} | Qty: {qty_col or 'N/A'} | Price: {price_col or 'N/A'}")

    df = clean_df(df, invoice_col, item_col, qty_col, price_col)
    print(f"Sau làm sạch: {len(df):,} dòng")

    basket = build_basket(df, invoice_col, item_col)
    print(f"Transactions: {basket.shape[0]:,} | Items: {basket.shape[1]:,}")

    algo = "apriori" if ID_PARITY == "odd" else "fpgrowth"
    print(
        f"Thuật toán: {algo.upper()} (theo quy định ID {'lẻ' if ID_PARITY == 'odd' else 'chẵn'})")

    freq, rules, sup, conf = auto_tune(basket, algo)
    if freq is None or rules is None or len(rules) == 0:
        # fallback: dùng grid default
        sup = INITIAL_SUPPORT_GRID[-1]
        conf = CONFIDENCE_GRID[-1]
        print("Auto-tune không tìm được cấu hình đẹp → dùng fallback.")
        freq = mine_itemsets(basket, algo, sup)
        rules = rules_from_itemsets(freq, conf)

    print(f"min_support={sup} | min_confidence={conf}")
    print(
        f"Frequent itemsets: {len(freq):,} | Association rules: {len(rules):,}")

    outdir = ensure_outdir()
    freq.to_csv(os.path.join(outdir, "frequent_itemsets.csv"), index=False)
    rules.sort_values(["lift", "confidence"], ascending=False)\
         .to_csv(os.path.join(outdir, "association_rules.csv"), index=False)

    # Nếu ID chẵn: thêm phân tích FP-Max cho pattern tối đại (tham khảo)
    if ID_PARITY == "even":
        freq_max = fpmax(basket, min_support=sup, use_colnames=True)
        freq_max.to_csv(os.path.join(
            outdir, "frequent_itemsets_maximal.csv"), index=False)
        print(
            f"FP-Max (maximal patterns): {len(freq_max):,} itemsets (đã lưu).")

    # In Top 10 luật cho slide
    if not rules.empty:
        cols = ["antecedents", "consequents", "support",
                "confidence", "lift", "leverage", "conviction"]
        print("\nTop 10 rules by lift:")
        print(rules.sort_values(["lift", "confidence"], ascending=False)[
              cols].head(10).to_string(index=False))
    else:
        print("Không sinh được luật nào với cấu hình hiện tại. Hãy hạ min_support hoặc min_confidence.")


if __name__ == "__main__":
    main()
