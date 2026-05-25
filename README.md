# Quy Trình Phân Tích Dữ Liệu Olist E-Commerce (Restructured)

Dự án này đã được tái cấu trúc theo chuẩn quy trình khoa học dữ liệu chuyên nghiệp để phân tích hành vi khách hàng, vận hành logistics và dự báo lượng đơn hàng hàng ngày của nền tảng thương mại điện tử Olist (Brazil).

---

## 1. Cấu Trúc Thư Mục Dự Án

Codebase được tổ chức phân lớp rõ ràng để tăng tính bảo trì và tái sử dụng:

```
DA_remake/
│
├── config/                  # Quản lý cấu hình tập trung
│   └── config.yaml          # Định nghĩa đường dẫn dữ liệu và siêu tham số
│
├── data/                    # Tổ chức dữ liệu
│   ├── raw/                 # Dữ liệu thô gốc từ Kaggle (olist_*.csv) và file ngày lễ
│   └── processed/           # Dữ liệu sạch và các bảng phân tích (RFM, Cohort, timeseries)
│
├── notebooks/               # Runner thử nghiệm và trực quan hóa tương tác (Percent Format # %%)
│   ├── 1.0_eda_business_analysis.py   # Phân tích logistics, review score, RFM, Cohort
│   └── 2.0_timeseries_prototyping.py  # Thử nghiệm chi tiết các mô hình dự báo
│
├── src/                     # Mã nguồn lõi dạng module tái sử dụng
│   ├── config_loader.py     # Loader giải quyết đường dẫn tuyệt đối
│   ├── data_cleaning.py     # Tiền xử lý dữ liệu thô thông minh (imputation, địa lý)
│   ├── feature_engineering.py# Tạo đặc trưng chuỗi thời gian & tính RFM, Cohort
│   ├── models.py            # Huấn luyện mô hình, kiểm định chéo và dự báo đệ quy
│   └── visualization.py     # Trực quan hóa cao cấp (cohort heatmap, RFM bar, forecasts)
│
├── reports/                 # Báo cáo kết quả cuối cùng
│   ├── figures/             # Các tệp biểu đồ PNG chất lượng cao
│   └── final_report.md      # Báo cáo kinh doanh & so sánh mô hình chi tiết (Tiếng Việt)
│
├── main.py                  # Script điều phối chạy tự động toàn bộ pipeline
├── requirements.txt         # Các thư viện Python cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng (Tệp này)
```

---

## 2. Cách Vận Hành Dự Án

### Bước 1: Cài đặt môi trường
Đảm bảo bạn đã cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ Pipeline tự động
Chỉ cần chạy một script duy nhất ở thư mục gốc để thực hiện toàn bộ quy trình từ làm sạch, tính toán đặc trưng, chạy kiểm định chéo các mô hình, đến vẽ biểu đồ và xuất báo cáo:
```bash
python3 main.py
```

### Bước 3: Xem Báo cáo & Kết quả phân tích
- Toàn bộ các biểu đồ trực quan hóa được sinh ra tại: [reports/figures/](file:///home/naoh/Documents/projects/DA_remake/reports/figures)
- Báo cáo phân tích kinh doanh chuyên sâu và so sánh mô hình chi tiết được lưu tại: [reports/final_report.md](file:///home/naoh/Documents/projects/DA_remake/reports/final_report.md)

---

## 3. Các Phân Tích & Cải Tiến Kỹ Thuật Chính

1. **Khắc Phục Rò Rỉ Dữ Liệu (Data Leakage):**
   Trong mô hình học máy (XGBoost, LightGBM), việc tạo đặc trưng trễ (`lag_1`, `lag_7`...) đã được viết lại bằng **Dự báo Đệ quy (Recursive Forecasting)**. Khi chạy dự báo 30 ngày ở tập test, mô hình chỉ sử dụng giá trị dự đoán của ngày hôm trước để tạo đặc trưng trễ cho ngày hôm sau, phản ánh đúng thực tế.

2. **Làm Sạch Dữ Liệu Thông Minh:**
   Không dùng `dropna(how='any')` bừa bãi làm mất dữ liệu. Các giá trị thiếu trong bảng sản phẩm được điền khuyết (imputation) để bảo toàn giao dịch mua bán. Tập dữ liệu địa lý khổng lồ (~1 triệu dòng) được gom cụm theo zip code để giảm thời gian xử lý và lưu trữ.

3. **Mô Hình Hóa Toàn Diện & Kiểm Định Chéo:**
   Đánh giá các mô hình **Baseline (Seasonal Naive)**, **SARIMAX**, **Prophet**, **XGBoost** và **LightGBM** trên **Time Series Cross-Validation (5 folds)** thay vì đánh giá trên một tập train/test split cố định duy nhất. Kết quả ghi nhận **Prophet** và **XGBoost** là hai mô hình hoạt động ổn định nhất trên dữ liệu Olist.
# Olist_Analyzing
