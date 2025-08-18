# -*- coding: utf-8 -*-
"""
Market Basket Analysis (Online Retail I/II) — Memory-Safe Version (BẢN CÓ CHÚ THÍCH)
-----------------------------------------------------------------------------------
Mục tiêu: Khai thác tập phổ biến & luật kết hợp để tìm cặp/bộ mặt hàng thường đi chung,
phục vụ thiết kế bundle/combo và cross-sell.

Bản này giữ nguyên logic từ script gốc, nhưng bổ sung chú thích tiếng Việt
để dễ đọc, dễ bảo trì.  (max_len=2 để tập trung COMBO 2 món theo yêu cầu đề tài)
"""

import os, re
from typing import List, Tuple
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules
import matplotlib.pyplot as plt

# ====================== CONFIG (THAM SỐ CẤU HÌNH) ======================
# FILE_PATH: đường dẫn tới file CSV Online Retail.
# - Ở máy bạn đang để path tuyệt đối. Có thể thay bằng đường dẫn tương đối ("online_retail_II.csv")
#   khi đặt file cùng thư mục với script.
FILE_PATH    = r"E:\MSE\HomeWork\Ky 2\DAM501.8\Test & Project\Project\Project Fix\online_retail_II.csv"

# ITEM_MODE: chọn biểu diễn mặt hàng bằng mã hay tên
# - "stockcode": ÍT nhiễu chính tả -> số unique items thấp -> nhanh/nhẹ hơn.
# - "description": ĐẸP để trình bày (map tên) nhưng dễ nhiễu, tăng số chiều -> nặng hơn.
ITEM_MODE    = "stockcode"            

# ID_PARITY: chọn thuật toán theo yêu cầu môn học
# - "odd"  (lẻ)  -> Apriori
# - "even" (chẵn)-> FP-Max (ghi tập tối đại) + FP-Growth (sinh luật, nhanh hơn)
ID_PARITY    = "even"                  

# Lưới tham số dò (auto-tune) cho min_support và min_confidence.
# - SUPPORT_GRID: bắt đầu từ 0.003 (0.3% giao dịch) hạ dần đến 0.0005 (0.05%)
#   Phù hợp data ~36k giỏ: 0.001 ~ 36 giao dịch; 0.003 ~ 109 giao dịch.
SUPPORT_GRID = [0.003, 0.002, 0.001, 0.0008, 0.0005]

# - CONF_GRID: từ 0.20 -> 0.02 để bắt những luật vừa chắc vừa đủ phổ biến.
CONF_GRID    = [0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]

# - MIN_LIFT: 1.0 để CHỈ giữ luật DƯƠNG (đi kèm nhiều hơn ngẫu nhiên) phục vụ gợi ý.
MIN_LIFT     = 1.0

# - TARGET_RANGE: mục tiêu số luật “đẹp” để thuyết trình (Top 10–20).
TARGET_RANGE = (10, 20)

# - OUTDIR: thư mục xuất kết quả (CSV + ảnh).
OUTDIR       = "out"

# Giới hạn số ITEM trước khi chạy Apriori/FP-Growth để tránh tràn RAM (OOM)
# - USE_TOPK: True -> Chỉ giữ TOP-K item phổ biến nhất theo support.
# - TOPK_ITEMS: K=600 là mức “an toàn” cho máy phổ thông, vẫn đủ đa dạng.
# - Nếu không dùng TOP-K, có thể giữ theo ngưỡng support của item (MIN_ITEM_SUPPORT).
USE_TOPK         = True
TOPK_ITEMS       = 600                 
MIN_ITEM_SUPPORT = 0.002               # 0.2% giao dịch; chỉ dùng khi USE_TOPK=False

# - MAX_LEN: chỉ xét tập dài tối đa = 2 -> tập trung COMBO 2 món (đúng đề tài) và giảm độ phức tạp.
MAX_LEN = 2                            
# =======================================================================


def step(msg: str):
    """In ra một thông báo theo step cho dễ theo dõi tiến trình."""
    print(f"\n🔹 {msg}")


def choose_col(df: pd.DataFrame, candidates):
    """
    Tự động chọn tên cột đầu tiên tìm thấy trong danh sách 'candidates'.
    Lý do: các bản Online Retail I/II có thể khác tên cột (Invoice/InvoiceNo, Price/UnitPrice).
    - df: DataFrame đầu vào
    - candidates: list tên cột có thể có
    Trả về: tên cột sử dụng. Nếu không tìm thấy -> raise KeyError để báo lỗi sớm.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns {candidates} found. Available: {list(df.columns)}")


def normalize_text(s: str) -> str:
    """
    Chuẩn hoá chuỗi mô tả:
    - upper() + strip(): tránh phân biệt hoa/thường, loại khoảng trắng thừa ở đầu/cuối
    - thay thế dấu nháy & thu gọn khoảng trắng bên trong
    Dùng khi ITEM_MODE='description' để giảm nhiễu chính tả.
    """
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', '').replace("'", "")
    return s


def load_and_clean(path: str):
    """
    ĐỌC & LÀM SẠCH DỮ LIỆU
    - Đọc CSV với low_memory=False (an toàn cho kiểu dữ liệu lẫn lộn).
    - Tự chọn tên cột phù hợp (Invoice/InvoiceNo, Price/UnitPrice, ...).
    - Bỏ dòng thiếu dữ liệu chính (dropna).
    - Loại hoá đơn trả hàng (Invoice bắt đầu 'C').
    - Chỉ giữ Quantity>0 và Price>0 để loại giao dịch bất thường.
    - Chọn trường ITEM theo ITEM_MODE (stockcode/description) và chuẩn hoá.
    - Giữ giỏ (Invoice) có ÍT NHẤT 2 MẶT HÀNG -> cần thiết để tạo cặp.
    Trả về:
        df (chỉ còn 2 cột: invoice_col, ITEM) và tên cột hoá đơn (invoice_col).
    """
    step("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    invoice_col = choose_col(df, ["InvoiceNo", "Invoice"])
    desc_col    = choose_col(df, ["Description"])
    code_col    = choose_col(df, ["StockCode", "Stock Code"])
    qty_col     = choose_col(df, ["Quantity"])
    price_col   = choose_col(df, ["UnitPrice", "Price"])

    # Giữ cột cần thiết & loại NA
    df = df[[invoice_col, desc_col, code_col, qty_col, price_col]].dropna()

    # Loại hóa đơn trả hàng (ký tự đầu 'C')
    df[invoice_col] = df[invoice_col].astype(str)
    df = df[~df[invoice_col].str.startswith("C")]

    # Chỉ giữ dòng có số lượng & giá dương
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)].copy()

    # Chọn biểu diễn ITEM
    if ITEM_MODE == "stockcode":
        df["ITEM"] = df[code_col].astype(str).str.strip()
    else:
        df["ITEM"] = df[desc_col].astype(str).map(normalize_text)

    # Chỉ giữ cột invoice & ITEM, loại trùng
    df = df[[invoice_col, "ITEM"]].dropna().drop_duplicates()

    # Giữ hoá đơn có >= 2 ITEM
    sizes = df.groupby(invoice_col)["ITEM"].nunique()
    keep_ids = sizes.index[sizes >= 2]
    df = df[df[invoice_col].isin(keep_ids)].copy()

    print(f"✅ After cleaning: {df[invoice_col].nunique():,} baskets | {len(df):,} rows | {df['ITEM'].nunique():,} unique items")
    return df, invoice_col


def build_basket(df: pd.DataFrame, invoice_col: str):
    """
    TẠO MA TRẬN GIỎ HÀNG (ONE-HOT)
    - Gom các ITEM theo từng hoá đơn -> mỗi hoá đơn là 1 list ITEM không trùng.
    - Dùng TransactionEncoder để one-hot encode -> DataFrame nhị phân (tx x items).
    - Sau đó LỌC ITEM để tránh OOM:
        * USE_TOPK=True  -> giữ TOPK_ITEMS theo support (mức độ phổ biến).
        * USE_TOPK=False -> giữ item có support >= MIN_ITEM_SUPPORT.
    Trả về: 'basket' là ma trận one-hot sau lọc (số chiều giảm đáng kể).
    """
    step("Building transactions & one-hot encoding...")
    # Tạo list ITEM duy nhất cho mỗi invoice
    transactions = df.groupby(invoice_col)["ITEM"].apply(lambda s: sorted(set(s))).tolist()

    # One-hot
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=te.columns_)
    print(f"✅ Encoded basket (pre-filter): {basket.shape[0]} tx × {basket.shape[1]} items")

    # ---- Lọc ITEM để tránh tràn bộ nhớ ----
    # item_support = tỉ lệ giao dịch chứa ITEM đó (trung bình theo cột của ma trận nhị phân)
    item_support = basket.mean(axis=0)

    if USE_TOPK:
        # Giữ TOP-K item có support cao nhất
        top_items = item_support.sort_values(ascending=False).head(TOPK_ITEMS).index
        basket = basket[top_items]
        print(f"✅ Kept top-{TOPK_ITEMS} items by support → now {basket.shape[1]} items")
    else:
        # Giữ item có support >= ngưỡng cố định
        keep = item_support[item_support >= MIN_ITEM_SUPPORT].index
        basket = basket[keep]
        print(f"✅ Kept items with support ≥ {MIN_ITEM_SUPPORT} → now {basket.shape[1]} items")

    return basket


def mine_itemsets_apriori(basket: pd.DataFrame, min_support: float):
    """
    Chạy Apriori để tìm tập phổ biến với:
    - min_support: ngưỡng xuất hiện tối thiểu (tỉ lệ giao dịch).
    - use_colnames=True: trả về itemsets ở dạng tên cột thay vì index.
    - max_len=MAX_LEN (=2): CHỈ xét cặp để phù hợp mục tiêu bundle/combo.
    - low_memory=True: giảm yêu cầu bộ nhớ (đặc biệt khi số item vẫn còn lớn).
    """
    return apriori(basket, min_support=min_support, use_colnames=True, max_len=MAX_LEN, low_memory=True)


def mine_itemsets_fpgrowth(basket: pd.DataFrame, min_support: float):
    """
    Chạy FP-Growth để tìm tập phổ biến (nhanh hơn Apriori trong nhiều trường hợp).
    - Giữ cùng cấu hình max_len=2 để tập trung vào cặp.
    """
    return fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=MAX_LEN)


def rules_from_itemsets(freq: pd.DataFrame, min_conf: float, min_lift: float):
    """
    Sinh luật kết hợp từ các tập phổ biến:
    - metric="confidence", min_threshold=min_conf: chỉ giữ luật có độ tin cậy >= min_conf.
    - Lọc thêm: chỉ giữ luật có hậu quả (consequents) gồm 1 item -> dễ triển khai gợi ý.
    - Lọc dương: giữ luật có lift >= min_lift (thường = 1.0).
    - Chuyển tập frozenset -> chuỗi có thứ tự để in/ghi file đẹp.
    Trả về: DataFrame các luật đã lọc.
    """
    if freq is None or freq.empty: 
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if rules.empty: 
        return rules

    # Hậu quả chỉ 1 item
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1)].copy()
    # Luật dương (lift >= 1)
    rules = rules[rules["lift"] >= min_lift].copy()

    # Làm đẹp để ghi/đọc
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return rules


def auto_tune(basket: pd.DataFrame, algo: str):
    """
    Tự động dò ngưỡng support & confidence trên các lưới đã cấu hình.
    Mục tiêu: số luật trong khoảng TARGET_RANGE (10–20) để thuyết trình.
    - Duyệt SUPPORT_GRID từ cao -> thấp (ưu tiên luật dễ phổ biến trước).
    - Với mỗi support, duyệt CONF_GRID từ cao -> thấp.
    - Nếu số luật rơi vào khoảng mục tiêu -> trả về ngay.
    - Nếu không tìm thấy, trả về tổ hợp cho nhiều luật nhất (best-effort).
    Trả về: (freq, rules, sup, conf)
    """
    low, high = TARGET_RANGE
    best = (None, None, None, None, -1)  # lưu tổ hợp có nhiều luật nhất để fallback

    for sup in SUPPORT_GRID:
        # Chọn thuật toán theo 'algo'
        if algo == "apriori":
            freq = mine_itemsets_apriori(basket, sup)
        else:
            freq = mine_itemsets_fpgrowth(basket, sup)

        if freq is None or freq.empty: 
            continue

        for conf in CONF_GRID:
            rules = rules_from_itemsets(freq, conf, MIN_LIFT)
            n = 0 if rules is None else len(rules)
            if n == 0: 
                continue

            # Đủ đẹp để thuyết trình -> dừng
            if low <= n <= high: 
                return freq, rules, sup, conf

            # Cập nhật best nếu số luật nhiều hơn
            if n > best[-1]:
                best = (freq, rules, sup, conf, n)

    # Không vào khoảng mục tiêu -> trả về tổ hợp nhiều luật nhất
    return best[0], best[1], best[2], best[3]


def save_all(freq, rules, parity):
    """
    Ghi toàn bộ kết quả ra thư mục OUTDIR:
    - frequent_itemsets.csv: tất cả tập phổ biến ứng với ngưỡng chọn.
    - association_rules.csv: toàn bộ luật dương đã sinh (lift >= 1, hậu quả 1 item).
    - top_rules.csv: top 20 luật theo (lift, confidence) để trình bày nhanh.
    - Nếu parity=even: file frequent_itemsets_maximal.csv đã được ghi ở phần main.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    (freq if freq is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "frequent_itemsets.csv"), index=False)
    (rules if rules is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "association_rules.csv"), index=False)
    if rules is not None and not rules.empty:
        rules.sort_values(["lift","confidence"], ascending=False).head(20).to_csv(os.path.join(OUTDIR, "top_rules.csv"), index=False)
    print(f"\n📦 Saved CSVs to ./{OUTDIR}/")


def plot_scatter(rules):
    """
    Vẽ biểu đồ scatter (support vs confidence), kích thước điểm ~ lift:
    - Giúp trực quan vùng luật “ngon” (support vừa phải, confidence cao, lift lớn).
    - Ảnh lưu tại OUTDIR/rules_scatter.png để chèn vào slide.
    """
    if rules is None or rules.empty: 
        return
    plt.figure(figsize=(7,5))
    sizes = (rules["lift"] * 40).clip(10, 400)
    plt.scatter(rules["support"], rules["confidence"], s=sizes, alpha=0.6)
    plt.title("Association Rules: support vs confidence (size ~ lift)")
    plt.xlabel("support"); plt.ylabel("confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "rules_scatter.png"), dpi=150)
    print(" - rules_scatter.png")


def main():
    """
    Luồng chính:
    1) Đọc & làm sạch dữ liệu -> (df, invoice_col)
    2) Tạo ma trận giỏ hàng (one-hot) -> basket (đã lọc item để tránh OOM)
    3) Chọn thuật toán theo ID_PARITY:
       - even -> ghi FP-Max (tối đại) rồi dùng FP-Growth để sinh luật
       - odd  -> Apriori
    4) Auto-tune tìm (support, confidence) phù hợp
    5) Lưu kết quả & vẽ biểu đồ
    6) In top 10 luật để tham khảo nhanh
    """
    df, invoice_col = load_and_clean(FILE_PATH)
    basket = build_basket(df, invoice_col)

    if basket.shape[1] < 2:
        print("❗Not enough items after filtering. Reduce filtering or increase TOPK_ITEMS.")
        return

    if ID_PARITY == "even":
        # FP-Max chỉ để báo cáo tập tối đại (không sinh luật) ở ngưỡng “vừa phải”.
        # - sup_for_max: lấy max(0.001, 5 / số_giao_dịch) để tránh quá nhiều/ít kết quả.
        sup_for_max = max(0.001, 5 / basket.shape[0])
        freq_max = fpmax(basket, min_support=sup_for_max, use_colnames=True, max_len=MAX_LEN)
        os.makedirs(OUTDIR, exist_ok=True)
        (freq_max if freq_max is not None else pd.DataFrame()).to_csv(os.path.join(OUTDIR, "frequent_itemsets_maximal.csv"), index=False)
        print(f"✅ FP-Max saved ({0 if freq_max is None else len(freq_max)} rows).")
        algo = "fpgrowth"
    else:
        algo = "apriori"

    print(f"\n▶ Using algorithm: {algo.upper()} (MAX_LEN={MAX_LEN}, ITEM_MODE={ITEM_MODE}, TOPK_ITEMS={TOPK_ITEMS if USE_TOPK else 'N/A'})")

    # Dò tham số để có số luật “đẹp” trong khoảng TARGET_RANGE
    freq, rules, sup, conf = auto_tune(basket, algo)

    if freq is None or rules is None or rules.empty:
        print("❗No positive rules (lift>=1) within configured ranges. Try increasing TOPK_ITEMS or lowering SUPPORT/CONF grids.")
        return

    print(f"\n⚙️ Best params -> min_support={sup}, min_confidence={conf}")
    print(f"Frequent itemsets: {len(freq)} | Positive rules: {len(rules)}")

    # Lưu & vẽ
    save_all(freq, rules, ID_PARITY)
    plot_scatter(rules)

    # In nhanh top 10 luật
    print("\n📊 Top 10 rules:")
    cols = ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
    print(rules.sort_values(['lift','confidence'], ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()