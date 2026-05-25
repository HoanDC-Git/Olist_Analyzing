# Báo Cáo Phân Tích Chuyên Sâu & Dự Báo Doanh Số E-Commerce Olist (Đã Cập Nhật Lọc Dữ Liệu)

Báo báo này tổng hợp các kết quả phân tích hành vi khách hàng, hiệu suất vận hành logistics và đánh giá các mô hình dự báo chuỗi thời gian lượng đơn hàng ngày trên nền tảng thương mại điện tử Olist (Brazil).

---

## 1. Tổng Quan Quy Trình & Cải Tiến Kỹ Thuật

Dự án đã được tái cấu trúc thành một pipeline dạng module chuyên nghiệp, loại bỏ hoàn toàn các lỗi nghiêm trọng ở dự án cũ:

1. **Tránh Rò Rỉ Dữ Liệu (Leakage-Free):** Trong XGBoost và LightGBM, việc tính toán lag/rolling features đã được thực hiện dạng **Dự báo Đệ quy (Recursive Forecasting)**, không sử dụng giá trị thực tế tương lai của tập test.
2. **Làm Sạch Dữ Liệu Thông Minh:** Không áp dụng `dropna(how='any')` bừa bãi gây mất sản phẩm và đơn hàng liên quan; điền khuyết (imputation) giá trị thiếu hợp lý.
3. **Loại Bỏ Điểm Kỳ Dị Cutoff Dữ Liệu:** Đã lọc dữ liệu kết thúc vào ngày **20/08/2018** (thay vì 31/08/2018) để loại bỏ hiện tượng database cutoff lỗi ở những ngày cuối tháng 8 (khi số đơn hàng giảm giả tạo từ ~250 đơn xuống 11 đơn do dừng cập nhật).

---

## 2. Phân Tích Khám Phá & Insight Kinh Doanh (EDA)

### 2.1. Phân Tích Khách Hàng RFM & Định Nghĩa Phân Khúc

Phân tích RFM (Recency, Frequency, Monetary) trên **94,663** khách hàng duy nhất cho thấy Olist đang gặp thách thức lớn trong việc giữ chân khách hàng.

Dưới đây là định nghĩa chi tiết và tỷ lệ của từng nhóm khách hàng:

| Phân khúc                                        | Định nghĩa kỹ thuật (R_Score, F_Score) | Ý nghĩa kinh doanh & Đặc điểm                                                                               |    Tỷ lệ khách hàng (%)     |
| :----------------------------------------------- | :------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :-------------------------: |
| **Champions** _(Khách hàng VIP)_                 | $R \ge 4$ và $F \ge 4$                 | Mua gần đây nhất, mua thường xuyên nhất và chi tiêu nhiều nhất. Cần có chính sách chăm sóc đặc biệt.        |     **0.14%** (129 KH)      |
| **Loyal Customers** _(KH trung thành)_           | $R \ge 3$ và $F \ge 3$                 | Mua hàng khá thường xuyên, chi tiêu tốt. Phản hồi tốt với các chương trình khuyến mãi.                      |    **1.86%** (1,762 KH)     |
| **Recent Customers** _(KH mới)_                  | $R \ge 4$ và $F < 3$                   | Mua gần đây nhưng tần suất thấp (mới mua lần đầu hoặc lần hai). Cần marketing kích thích mua lần tiếp theo. |   **38.91%** (36,835 KH)    |
| **About to Sleep** _(KH sắp ngủ đông)_           | $R = 3$ và $F < 3$                     | Recency và Frequency dưới trung bình. Dễ bị đối thủ lôi kéo nếu không được tiếp cận lại kịp thời.           |   **19.12%** (18,103 KH)    |
| **Customers Needing Attention** _(KH cần chú ý)_ | $R = 2$                                | Đã khá lâu chưa quay lại mua hàng. Cần các chiến dịch kích hoạt lại (reactivation) bằng mã giảm giá.        |   **19.99%** (18,929 KH)    |
| **Can't Lose Them** _(Không thể để mất)_         | $R = 1$ và $F \ge 3$                   | Từng mua rất nhiều và thường xuyên nhưng đã rất lâu chưa quay lại. Có nguy cơ cao đã chuyển sang đối thủ.   |     **0.48%** (457 KH)      |
| **At Risk** _(KH gặp nguy cơ)_                   | $R = 1$ và $F = 2$                     | Đã lâu không mua hàng và tần suất mua thấp. Khả năng cao sẽ rời bỏ sàn.                                     | **—** _(Gộp vào nhóm Lost)_ |
| **Lost** _(Khách hàng đã mất)_                   | $R = 1$ và $F = 1$                     | Chỉ mua duy nhất 1 lần từ rất lâu trước đây và không quay lại. Gần như không thể kích hoạt lại.             |   **19.45%** (18,414 KH)    |

> [!IMPORTANT]
> **Insight then chốt:** Nhóm khách hàng mua một lần rồi rời đi (**Recent, About to Sleep, Needing Attention, Lost**) chiếm hơn **97%** tổng số khách hàng. Tỷ lệ trung thành (**Champions & Loyal**) chỉ chiếm vỏn vẹn **2.0%**. Điều này phản ánh Olist có chi phí chuyển đổi khách hàng (CAC) rất cao nhưng giá trị vòng đời khách hàng (CLV) cực kỳ thấp.

---

### 2.2. Phân Tích Cohort Retention (Tỷ Lệ Giữ Chân)

Biểu đồ nhiệt Retention minh chứng rõ nét cho thách thức trên:

- Tỷ lệ quay lại mua hàng ở tháng thứ nhất (Month 1) của mọi nhóm cohort đều **dưới 0.8%** (thường từ 0.3% - 0.6%).
- Đến tháng thứ ba (Month 3), tỷ lệ giữ chân tiệm cận về **0%**.
- Olist thực chất là một nền tảng giao dịch một lần (one-time transactional platform), chưa xây dựng được lòng trung thành của khách hàng.

---

### 2.3. Hiệu Suất Logistics & Đánh Giá Khách Hàng (Review Score)

- **Top 3 bang giao hàng chậm nhất:** Roraima (RR - 29.5 ngày), Amapá (AP - 26.7 ngày), Amazonas (AM - 26.0 ngày).
- **Bang giao hàng nhanh nhất:** São Paulo (SP - 8.3 ngày).
- **Mối tương quan với Đánh giá:**
  - Đơn hàng giao đúng hạn hoặc sớm đạt điểm trung bình **4.29 / 5.0** (với 62.3% đánh giá 5 sao).
  - Đơn hàng giao trễ khiến điểm trung bình tụt thảm hại xuống **2.27 / 5.0** (với **53.7% đánh giá 1 sao**).
  - Trễ hẹn logistics là nguyên nhân hàng đầu tàn phá uy tín thương hiệu của Olist.

---

## 3. Kết Quả Dự Báo Chuỗi Thời Gian Sau Khi Lọc Dữ Liệu Cutoff

Bằng cách loại bỏ phần dữ liệu bị lỗi do database cutoff sau ngày **20/08/2018**, chúng tôi đã loại bỏ nhiễu giảm sút giả tạo ở tập test cuối cùng.

### 3.1. So Sánh Sai Số Giữa Bộ Dữ Liệu Cũ và Bộ Dữ Liệu Lọc Mới (Tập Test 30 Ngày Cuối)

Khi dịch chuyển cửa sổ kiểm thử ra khỏi vùng cutoff lỗi (chuyển sang giai đoạn từ `21/07/2018` đến `20/08/2018` thay vì đến `29/08/2018`), độ chính xác dự báo tăng vượt trội:

| Mô hình                       | MAPE ban đầu (Có lỗi Cutoff) | MAPE mới (Đã lọc bỏ Cutoff) | Tỷ lệ giảm sai số (%) |
| :---------------------------- | :--------------------------: | :-------------------------: | :-------------------: |
| **XGBoost (Đệ quy)**          |            123.7%            |          **16.9%**          | **Giảm 86.3% sai số** |
| **LightGBM (Đệ quy)**         |            172.9%            |          **21.1%**          | **Giảm 87.8% sai số** |
| **Baseline (Seasonal Naive)** |            135.9%            |          **20.2%**          | **Giảm 85.1% sai số** |
| **SARIMAX**                   |            125.1%            |          **22.6%**          | **Giảm 81.9% sai số** |
| **Prophet**                   |            106.8%            |          **24.1%**          | **Giảm 77.4% sai số** |

> [!TIP]
> Việc loại bỏ vùng nhiễu giúp sai số MAPE của tất cả các mô hình giảm từ mức >100% xuống dưới **25%**.

---

### 3.2. Đánh Giá Hiệu Năng Kiểm Định Chéo Trung Bình (Average 5-Fold CV)

Để có cái nhìn khách quan không phụ thuộc vào 1 fold duy nhất, dưới đây là kết quả kiểm định chéo trung bình trên 5 folds sau khi cấu trúc lại dữ liệu:

| Mô hình                       | Trung bình RMSE | Trung bình MAE | Trung bình MAPE |  Thứ hạng hiệu năng   |
| :---------------------------- | :-------------: | :------------: | :-------------: | :-------------------: |
| **XGBoost (Đệ quy)**          |    **49.98**    |   **39.95**    |   **20.45%**    | **Hạng 1 (Tốt nhất)** |
| **Baseline (Seasonal Naive)** |      60.73      |     48.95      |     25.50%      |        Hạng 2         |
| **Prophet**                   |      66.38      |     53.60      |     28.58%      |        Hạng 3         |
| **SARIMAX**                   |      66.99      |     54.73      |     29.44%      |        Hạng 4         |
| **LightGBM (Đệ quy)**         |      72.21      |     58.65      |     31.89%      |        Hạng 5         |

### 3.3. So Sánh Chi Tiết: XGBoost Dự Đoán vs Số Lượng Đơn Thực Tế

Biểu đồ dưới đây so sánh trực quan sai lệch hàng ngày giữa số lượng đơn hàng thực tế và số lượng đơn hàng dự đoán bởi mô hình **XGBoost (Học máy đệ quy)** trên tập kiểm thử 30 ngày sạch (từ 21/07/2018 đến 19/08/2018):

![XGBoost vs Actual](figures/05_xgboost_vs_actual.png)

*Mô tả biểu đồ:*
- Đường màu xanh đen đại diện cho số lượng đơn hàng thực tế hàng ngày.
- Đường màu đỏ đại diện cho dự báo đệ quy của XGBoost.
- Vùng màu đỏ nhạt biểu thị khoảng sai lệch (Error) giữa thực tế và dự báo. Ta thấy mô hình bám rất sát các đỉnh và đáy chu kỳ tuần của thị trường, mang lại MAPE cực kỳ ấn tượng là **16.89%** cho giai đoạn này.

---

## 4. Phân Tích Điểm Mạnh, Yếu Của Từng Mô Hình Trong Dự Án Này

### 4.1. XGBoost với Đặc Trưng Trễ Đệ Quy (Hạng 1)

- **Tại sao hoạt động tốt nhất?** Sau khi loại bỏ nhiễu cutoff, mô hình cây XGBoost phát huy tối đa sức mạnh nhờ các đặc trưng tự hồi quy (`lag_1`, `lag_7`, `lag_14`, `lag_30`) và đặc trưng động (`rolling_mean_7`, `rolling_mean_30`). Cùng với các đặc trưng lịch biểu (`day_of_week`), XGBoost học được mối quan hệ phi tuyến phức tạp của chu kỳ tuần rất tốt. Nhờ thiết lập đệ quy, mô hình không bị rò rỉ thông tin trong tập test mà vẫn duy trì MAPE ở mức **20.45%**.
- **Điểm yếu:** Đòi hỏi tính toán đệ quy phức tạp trong sản xuất và không thể tự ngoại suy xu hướng dài hạn nếu xu hướng tiếp tục tăng vượt ngưỡng lịch sử.

### 4.2. Baseline (Seasonal Naive) (Hạng 2)

- **Tại sao hoạt động tốt?** Với công thức cực kỳ đơn giản $y_t = y_{t-7}$, mô hình tận dụng triệt để tính chu kỳ tuần lặp đi lặp lại rất đều đặn của hành vi mua sắm trực tuyến (lượng mua thứ 2 luôn giống thứ 2 tuần trước). Điểm MAPE **25.50%** là một benchmark rất cao mà không phải mô hình phức tạp nào cũng vượt qua được.
- **Điểm yếu:** Hoàn toàn bất lực trước các ngày lễ di động hoặc biến động đột biến như Black Friday (khi cú nhảy vọt 4 lần không diễn ra trùng ngày giữa các năm).

### 4.3. Prophet (Hạng 3)

- **Tại sao tốt?** Khớp tốt xu hướng phi tuyến tính dài hạn và bóc tách các thành phần mùa vụ cộng tính hiệu quả. Việc xử lý Black Friday bằng trọng số ngoại sinh giúp Prophet không bị lệch dự báo sau biến động lớn.
- **Tại sao xếp dưới XGBoost?** Prophet đôi khi quá "mượt mà" (smooth), nó tập trung vào xu hướng dài hạn nên các dao động ngắn hạn hàng ngày không bám sát và nhạy bén bằng mô hình dựa trên các lags ngắn hạn như XGBoost.

### 4.4. SARIMAX (Hạng 4)

- **Tại sao kém?** Bản chất tuyến tính của SARIMAX không khớp tốt với các cú sốc biến động cực đoan phi tuyến tính. Nó cũng rất nhạy cảm với các tham số bậc tự hồi quy cố định, dễ bị trôi dự báo khi gặp các đợt biến động mạnh kéo dài.

### 4.5. LightGBM (Hạng 5)

- **Tại sao kém hơn XGBoost?** LightGBM tối ưu hóa phân nhánh theo chiều sâu lá (leaf-wise), cực kỳ thích hợp cho các bộ dữ liệu siêu lớn (hàng triệu dòng). Với tập dữ liệu chuỗi thời gian ngắn (~560 dòng ngày huấn luyện), thuật toán này dễ bị overfitting hoặc dự báo các giá trị trung bình quá mức an toàn, dẫn đến sai số cao hơn so với XGBoost vốn tối ưu hơn cho tập dữ liệu vừa và nhỏ.

---

## 5. Khuyến Nghị Chiến Lược Cho Olist

1. **Kế hoạch Giao hàng Đa kênh & Kho bãi (Fulfillment Center):**
   Khuyến khích các sellers ở São Paulo gửi hàng ký kho tại các trung tâm phân phối của Olist ở vùng Đông Bắc/Bắc để rút ngắn thời gian giao hàng xuống dưới 10 ngày (thay vì 25-30 ngày hiện tại), bảo vệ điểm đánh giá 5 sao của sàn.
2. **Chiến dịch Tiếp thị Lại (Retargeting):**
   Tập trung tiếp cận nhóm khách hàng khổng lồ "Recent Customers" (38.9%) bằng coupon mua hàng lần 2 trong vòng 30 ngày để nâng tỷ lệ giữ chân khách hàng (Month 1 retention) vốn đang ở mức báo động (<1%).
3. **Capacity Planning với XGBoost:**
   Sử dụng dự báo của XGBoost (sai số ~20%) để lên kế hoạch phân bổ nhân sự xử lý đơn hàng và làm việc trước với các bên vận chuyển đối tác, sẵn sàng cho các đợt sóng mua sắm trong tương lai.
