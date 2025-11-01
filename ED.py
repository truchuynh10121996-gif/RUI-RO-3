# app.py — Streamlit PD + Phân tích Gemini (FIX YÊU CẦU ẨN GIAO DIỆN)

# =========================
# THƯ VIỆN BẮT BUỘC VÀ BỔ SUNG
# ... (Phần thư viện giữ nguyên)
# =========================
from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# Thư viện Machine Learning và Mô hình
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# ... (Phần Thư viện Gemini, Hàm get_ai_analysis, và Hàm compute_ratios_from_three_sheets giữ nguyên) ...

# =========================
# UI & TRAIN MODEL
# =========================
np.random.seed(0)
# st.title("DỰ BÁO THAM SỐ PD") 
# => Loại bỏ st.title
# st.write("## Dự báo xác suất vỡ nợ của khách hàng_PD")
# => Loại bỏ st.write này, vì nó đã được chuyển vào tab "Xây dựng mô hình" ở các yêu cầu trước.

st.title("HỆ THỐNG PHÂN TÍCH TÍN DỤNG DOANH NGHIỆP") # Thêm lại title chung cho ứng dụng

# Hiển thị trạng thái thư viện AI
st.caption("🔎 Trạng thái Gemini: " + ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai."))

# Load dữ liệu huấn luyện (CSV có default, X_1..X_14)
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df = None

uploaded_file = st.file_uploader("Tải CSV dữ liệu huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

if df is None:
    st.info("Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

# Kiểm tra cột cần thiết
required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Thiếu cột: {missing}")
    st.stop()

st.write(df[[f"X_{i}" for i in range(1, 15)]].describe())

# Train model
X = df.drop(columns=['default'])
y = df['default'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

# Dự báo & đánh giá (Giữ nguyên logic)
y_pred_in = model.predict(X_train)
y_proba_in = model.predict_proba(X_train)[:, 1]
y_pred_out = model.predict(X_test)
y_proba_out = model.predict_proba(X_test)[:, 1]

metrics_in = {
    "accuracy_in": accuracy_score(y_train, y_pred_in),
    "precision_in": precision_score(y_train, y_pred_in, zero_division=0),
    "recall_in": recall_score(y_train, y_pred_in, zero_division=0),
    "f1_in": f1_score(y_train, y_pred_in, zero_division=0),
    "auc_in": roc_auc_score(y_train, y_proba_in),
}
metrics_out = {
    "accuracy_out": accuracy_score(y_test, y_pred_out),
    "precision_out": precision_score(y_test, y_pred_out, zero_division=0),
    "recall_out": recall_score(y_test, y_pred_out, zero_division=0),
    "f1_out": f1_score(y_test, y_pred_out, zero_division=0),
    "auc_out": roc_auc_score(y_test, y_proba_out),
}

menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':    
    # ... (Giữ nguyên logic Mục tiêu) ...
    st.subheader("Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số X1–X14.")
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except Exception:
            st.warning(f"Không tìm thấy {img}")

elif choice == 'Xây dựng mô hình':
    # ... (Giữ nguyên logic Xây dựng mô hình) ...
    st.subheader("Xây dựng mô hình")

    st.write("##### 1) Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3)) 

    st.write("##### 2) Trực quan hóa dữ liệu")
    col = st.text_input('Nhập tên biến X muốn vẽ', value='X_1')
    if col in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x=col, y='default', alpha=0.4, ax=ax)
            x_range = np.linspace(df[col].min(), df[col].max(), 100)
            X_temp = df[[col]].copy()
            y_temp = df['default']
            lr_temp = LogisticRegression(max_iter=1000)
            lr_temp.fit(X_temp, y_temp)
            x_test = pd.DataFrame({col: x_range})
            y_curve = lr_temp.predict_proba(x_test)[:, 1]
            ax.plot(x_range, y_curve, color='red', linewidth=2)
            ax.set_ylabel('Xác suất default')
            ax.set_xlabel(col)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")
    else:
        st.warning("Biến không tồn tại trong dữ liệu.")

    st.write("##### 3) Kết quả đánh giá")
    dt = pd.DataFrame([metrics_in | metrics_out])
    st.dataframe(dt)

    st.write("##### 4) Ma trận nhầm lẫn (test)")
    cm = confusion_matrix(y_test, y_pred_out)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig2, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig2)
    plt.close()

elif choice == 'Sử dụng mô hình để dự báo':
    # Chỉ hiển thị tiêu đề chính, ẩn tiêu đề phụ "Dự báo xác suất vỡ nợ..."
    st.subheader("Sử dụng mô hình để dự báo & phân tích AI (3 sheet)") 
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")

    up_xlsx = st.file_uploader("Tải ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")
    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet (Giữ nguyên logic)
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown("### Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))
        
        # Tạo payload data cho AI (Giữ nguyên logic)
        data_for_ai = ratios_df.iloc[0].to_dict()

        # (Tuỳ chọn) dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        # PHẦN NÀY ĐÃ ĐƯỢC BỌC TRONG st.expander NÊN TÍNH NĂNG ĐÃ BỊ ẨN MẶC ĐỊNH.
        # Ta chỉ cần đảm bảo không có tiêu đề phụ nào hiển thị PD trước nó.
        if set(X.columns) == set(ratios_df.columns):
            # Giữ nguyên st.expander để ẩn thông tin chi tiết về PD
            with st.expander("Xác suất vỡ nợ dự báo (Tính năng phụ)"):
                try:
                    # Logic tính toán PD (Giữ nguyên logic cốt lõi)
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    show = ratios_df.copy()
                    show["pd"] = probs
                    show["pred_default"] = preds
                    st.dataframe(show.style.format({"pd": "{:.3f}"}))
                except Exception as e:
                    st.warning(f"Lỗi khi tính PD: {e}")
        
        # Gemini Phân tích & khuyến nghị (Giữ nguyên logic)
        st.markdown("### Phân tích AI & đề xuất CHO VAY/KHÔNG CHO VAY")
        
        # Thêm các chỉ số PD nếu đã tính được vào payload
        # Note: 'probs' chỉ tồn tại nếu khối try bên trên thành công.
        if 'probs' in locals():
            data_for_ai['PD_Probability'] = probs[0]
            data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"

        if st.button("Yêu cầu AI Phân tích"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
