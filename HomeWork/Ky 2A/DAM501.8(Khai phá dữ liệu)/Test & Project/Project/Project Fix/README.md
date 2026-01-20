# 📊 Dự án Phân tích Khách hàng và Khai phá Dữ liệu - Online Retail

## 📋 Tổng quan dự án

Dự án này tập trung vào việc phân tích dữ liệu bán lẻ trực tuyến (Online Retail Dataset V2) để thực hiện:
- **Phân đoạn khách hàng (Customer Segmentation)** dựa trên RFM analysis
- **Gắn nhãn sản phẩm (Product Tagging)** sử dụng NLP và machine learning
- **Phân tích giỏ hàng (Market Basket Analysis)** để tìm luật kết hợp
- **Trực quan hóa dữ liệu (Data Visualization)** với Plotly

Dự án được chia thành 2 phần chính:
1. **DataVisualize.ipynb**: Notebook phân tích khách hàng và trực quan hóa
2. **Main.py**: Script phân tích giỏ hàng và tìm luật kết hợp

---

## 🎯 Mục tiêu dự án

1. **Làm sạch và tiền xử lý dữ liệu** từ dataset Online Retail V2
2. **Phân loại sản phẩm tự động** dựa trên mô tả sản phẩm
3. **Phân đoạn khách hàng** thành các nhóm có đặc điểm tương đồng
4. **Xây dựng mô hình dự đoán** phân loại khách hàng mới
5. **Tìm luật kết hợp** giữa các sản phẩm để đề xuất combo/bundle
6. **Trực quan hóa kết quả** qua các biểu đồ tương tác

---

## 📁 Cấu trúc dự án

```
Project Fix/
├── DataVisualize.ipynb          # Notebook phân tích chính
├── Main.py                       # Script Market Basket Analysis
├── online_retail_II.csv         # Dataset đầu vào
├── Slide.pptx                    # Slide thuyết trình
└── README.md                     # File mô tả dự án (file này)
```

---

## 🔧 Công nghệ sử dụng

### Thư viện Python chính:
- **Data Processing**: `pandas`, `numpy`
- **NLP**: `texthero`, `nltk`, `sklearn.feature_extraction.text`
- **Machine Learning**: 
  - `sklearn` (Logistic Regression, Random Forest, XGBoost, SVM, etc.)
  - `xgboost`
- **Visualization**: `plotly`, `plotly.express`, `plotly.graph_objects`
- **Market Basket Analysis**: `mlxtend` (Apriori, FP-Growth, FP-Max)
- **Statistical Analysis**: `scipy.stats`

---

## 📊 Nội dung chi tiết

### 1. DataVisualize.ipynb - Phân tích Khách hàng

#### 1.1. Làm sạch dữ liệu (Data Cleaning)
- Xử lý dữ liệu thiếu (22% Customer ID bị thiếu)
- Loại bỏ giá trị trùng lặp
- Xử lý mã đặc biệt (Discount, Postage)
- Xử lý đơn hàng bị hủy (Canceled Orders)

#### 1.2. Gắn nhãn sản phẩm (Product Tagging)
- **Màu sắc sản phẩm**: Trích xuất màu từ mô tả (black, blue, red, etc.)
- **Thiết kế**: Nhận diện pattern/design (vintage, retro, hearts, etc.)
- **Danh mục**: Phân loại sản phẩm thành 10 nhóm chính:
  - Home Decoration
  - Bags
  - Gifts
  - Party
  - Tableware
  - Other
  - ...
- Sử dụng **TF-IDF** và các thuật toán phân loại (Random Forest, XGBoost, etc.)

#### 1.3. Kỹ thuật đặc trưng (Feature Engineering)
- **RFM Analysis**:
  - **Recency**: Thời gian từ lần mua gần nhất
  - **Frequency**: Tần suất mua hàng
  - **Monetary**: Giá trị giao dịch
- **Time Features**: Year, Month, Weekday, Hour, Day
- **Product Category Expenses**: Chi tiêu theo từng danh mục sản phẩm

#### 1.4. Phân đoạn khách hàng (Customer Segmentation)
Phân loại khách hàng thành 9 nhóm dựa trên RFM Score:
- **Best Customers** (111): Khách hàng tốt nhất
- **Loyal Customers**: Khách hàng trung thành
- **Big Spenders**: Khách hàng chi tiêu lớn
- **Good Customers**: Khách hàng tốt
- **Average Customer**: Khách hàng trung bình
- **Not So Good Customers**: Khách hàng không tốt lắm
- **Almost Lost**: Khách hàng sắp mất
- **Lost Customers** (411): Khách hàng đã mất
- **Bad Customers** (444): Khách hàng xấu

#### 1.5. Học có giám sát (Supervised Learning)
- Tách dữ liệu: Khách hàng cũ (train) vs Khách hàng mới (test - 2 tháng cuối)
- So sánh các mô hình: SGD, SVM, Random Forest, XGBoost, MLP, etc.
- **XGBoost** cho kết quả tốt nhất
- Dự đoán phân loại khách hàng mới chỉ với vài giao dịch

#### 1.6. Trực quan hóa dữ liệu (Data Visualization)
- **Tổng doanh thu theo tháng**: Phân tích xu hướng và tính mùa vụ
- **Bản đồ khách hàng**: Phân bố khách hàng theo quốc gia (2009-2011)
- **Phân tích theo phân đoạn**: Doanh thu theo từng nhóm khách hàng
- **Sản phẩm bán chạy**: Top 10 sản phẩm và xu hướng theo thời gian
- **Phân tích thời gian**: Doanh thu theo giờ, ngày trong tuần, ngày trong tháng
- **Phân bố danh mục**: Biểu đồ tròn doanh thu theo danh mục sản phẩm

### 2. Main.py - Market Basket Analysis

#### 2.1. Chức năng chính
- **Tìm tập phổ biến (Frequent Itemsets)**: Sử dụng Apriori hoặc FP-Growth
- **Sinh luật kết hợp (Association Rules)**: Tìm cặp sản phẩm thường mua cùng nhau
- **Tối ưu tự động**: Tự động điều chỉnh min_support và min_confidence
- **Tập trung vào combo 2 món**: max_len=2 để phù hợp với mục tiêu bundle/combo

#### 2.2. Cấu hình
- **ITEM_MODE**: Chọn biểu diễn sản phẩm (stockcode hoặc description)
- **ID_PARITY**: Chọn thuật toán (odd → Apriori, even → FP-Growth + FP-Max)
- **SUPPORT_GRID**: Lưới tham số support (0.003 → 0.0005)
- **CONF_GRID**: Lưới tham số confidence (0.20 → 0.02)
- **TOPK_ITEMS**: Giới hạn số item để tránh tràn bộ nhớ (mặc định: 600)

#### 2.3. Kết quả đầu ra
- `frequent_itemsets.csv`: Tất cả tập phổ biến
- `association_rules.csv`: Tất cả luật kết hợp
- `top_rules.csv`: Top 20 luật tốt nhất
- `frequent_itemsets_maximal.csv`: Tập tối đại (nếu dùng FP-Max)
- `rules_scatter.png`: Biểu đồ scatter (support vs confidence)

---

## 🚀 Hướng dẫn chạy dự án

### Yêu cầu hệ thống
- Python 3.7+
- Jupyter Notebook hoặc JupyterLab

### Cài đặt thư viện

```bash
pip install pandas numpy scipy
pip install texthero nltk
pip install plotly plotly-express
pip install scikit-learn xgboost
pip install mlxtend matplotlib
pip install jupyter-dash
```

### Chạy DataVisualize.ipynb

1. Mở Jupyter Notebook:
```bash
jupyter notebook
```

2. Mở file `DataVisualize.ipynb`

3. Chạy các cell theo thứ tự:
   - Cell 1-4: Cài đặt thư viện
   - Cell 5-6: Import và load dữ liệu
   - Cell 7-25: Làm sạch dữ liệu
   - Cell 26-90: Gắn nhãn sản phẩm
   - Cell 91-133: Kỹ thuật đặc trưng
   - Cell 134-146: Phân đoạn khách hàng
   - Cell 147-175: Học có giám sát
   - Cell 176-212: Trực quan hóa

**Lưu ý**: Cần cập nhật đường dẫn file CSV trong Cell 6:
```python
df = pd.read_csv(r'ĐƯỜNG_DẪN_ĐẾN_FILE\online_retail_II copy.csv')
```

### Chạy Main.py

1. Cập nhật đường dẫn file trong `Main.py`:
```python
FILE_PATH = r"ĐƯỜNG_DẪN_ĐẾN_FILE\online_retail_II.csv"
```

2. Chạy script:
```bash
python Main.py
```

3. Kết quả sẽ được lưu trong thư mục `out/`

---

## 📈 Kết quả chính

### Phân đoạn khách hàng
- Phân loại thành công khách hàng thành 9 nhóm
- Mô hình XGBoost đạt độ chính xác cao trong dự đoán phân loại
- Có thể phân loại khách hàng mới chỉ với vài giao dịch

### Gắn nhãn sản phẩm
- Gắn nhãn được ~60% sản phẩm thông qua phương pháp bán tự động
- Phân loại thành 10 danh mục chính
- Trích xuất thành công màu sắc và thiết kế từ mô tả

### Market Basket Analysis
- Tìm được các cặp sản phẩm thường mua cùng nhau
- Sinh luật kết hợp với lift > 1.0 (luật dương)
- Hỗ trợ thiết kế bundle/combo và cross-sell

### Trực quan hóa
- Biểu đồ tương tác với Plotly
- Phân tích xu hướng theo thời gian
- Phân tích địa lý (bản đồ thế giới)
- Phân tích theo phân đoạn và danh mục

---

## 💡 Ứng dụng thực tế

1. **Marketing cá nhân hóa**: 
   - Gửi email/quảng cáo phù hợp với từng nhóm khách hàng
   - Đề xuất sản phẩm dựa trên màu sắc yêu thích

2. **Đề xuất sản phẩm**:
   - Gợi ý combo/bundle dựa trên luật kết hợp
   - Cross-sell và up-sell hiệu quả

3. **Quản lý khách hàng**:
   - Xác định khách hàng có nguy cơ rời bỏ (Almost Lost, Lost)
   - Tập trung chăm sóc khách hàng tốt nhất (Best Customers)

4. **Quản lý kho**:
   - Dự đoán nhu cầu theo mùa
   - Tối ưu hóa tồn kho theo danh mục

---

## 📝 Ghi chú

- Dataset Online Retail V2 chứa dữ liệu từ năm 2009-2011
- 22% dữ liệu bị mất Customer ID và đã được loại bỏ
- Phân đoạn khách hàng nên được thực hiện với sự hợp tác của team Marketing
- Một số danh mục sản phẩm có thể cần được điều chỉnh lại

---

## 👤 Tác giả

Dự án được phát triển cho môn học **DAM501.8 - Khai phá dữ liệu**

---

## 📄 License

Dự án này được sử dụng cho mục đích học tập và nghiên cứu.

---

## 🔗 Tài liệu tham khảo

- [Online Retail Dataset](https://www.kaggle.com/datasets/mathchi/online-retail-ii-data-set-from-ml-repository)
- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
- [Association Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning)
- [Plotly Documentation](https://plotly.com/python/)
- [TextHero Documentation](https://texthero.org/)

---

**Cập nhật lần cuối**: 2024
