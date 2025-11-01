# app.py — Streamlit PD + Phân tích Gemini (FIX LỖI NameError TẠI st.caption)

# =========================
# THƯ VIỆN BẮT BUỘC VÀ BỔ SUNG
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
# Các thư viện BỔ SUNG theo yêu cầu (nếu được sử dụng trong code sau này)
# import xgboost as xgb
# import graphviz
# import statsmodels.api as sm

# =========================
# THÊM THƯ VIỆN GOOGLE GEMINI VÀ OPENAI (ĐÃ ĐƯA LÊN ĐẦU)
# => Đảm bảo _GEMINI_OK và _OPENAI_OK được định nghĩa trước khi UI cần
# =========================
try:
    from google import genai
    from google.genai.errors import APIError
    _GEMINI_OK = True
except Exception:
    genai = None
    APIError = Exception
    _GEMINI_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False


MODEL_NAME = "gemini-2.5-flash" # Model mạnh mẽ và hiệu quả cho phân tích văn bản

# =========================
# HÀM GỌI GEMINI API (GIỮ NGUYÊN)
# =========================
def get_ai_analysis(data_payload: dict, api_key: str) -> str:
    """
    Sử dụng Gemini API để phân tích chỉ số tài chính.
    """
    if not _GEMINI_OK:
        return "Lỗi: Thiếu thư viện google-genai (cần cài đặt: pip install google-genai)."

    client = genai.Client(api_key=api_key)

    sys_prompt = (
        "Bạn là chuyên gia phân tích tín dụng doanh nghiệp tại ngân hàng. "
        "Phân tích toàn diện dựa trên 14 chỉ số tài chính (X1..X14). "
        "Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4) Hiệu quả hoạt động. "
        "Kết thúc bằng khuyến nghị in hoa: CHO VAY hoặc KHÔNG CHO VAY, kèm 2–3 điều kiện nếu CHO VAY. "
        "Viết bằng tiếng Việt súc tích, chuyên nghiệp."
    )
    
    user_prompt = "Bộ chỉ số X1..X14 cần phân tích:\n" + str(data_payload) + "\n\nHãy phân tích và đưa ra khuyến nghị."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}
            ],
            config={"system_instruction": sys_prompt}
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi API Gemini: {e}"
    except Exception as e:
        return f"Lỗi không xác định: {e}"


# =========================
# TÍNH X1..X14 TỪ 3 SHEET (CDKT/BCTN/LCTT) (GIỮ NGUYÊN)
# =========================
# ... (Khối ALIAS_IS, ALIAS_BS, ALIAS_CF, _pick_year_cols, _get_row_vals, compute_ratios_from_three_sheets giữ nguyên) ...
ALIAS_IS = {
   # ... (giữ nguyên)
}
ALIAS_BS = {
   # ... (giữ nguyên)
}
ALIAS_CF = {
   # ... (giữ nguyên)
}
# ... (Các hàm _pick_year_cols, _get_row_vals, compute_ratios_from_three_sheets giữ nguyên) ...


# =========================
# UI & TRAIN MODEL (ĐÃ CẬP NHẬT LẠI CẤU TRÚC ĐẦU)
# =========================
np.random.seed(0)
# Thêm lại st.title để tiêu đề lớn hiển thị
st.title("HỆ THỐNG PHÂN TÍCH TÍN DỤNG DOANH NGHIỆP") 

# Thêm lại logic session_state (thường bị bỏ sót khi cắt/dán)
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Dòng gây lỗi NameError đã được FIX vì _GEMINI_OK đã được định nghĩa ở trên
st.caption("🔎 Trạng thái Gemini: " + ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai."))

# Load dữ liệu huấn luyện (CSV có default, X_1..X_14)
# ... (Logic Load Data giữ nguyên) ...
try:
    df_default = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df_default = None

uploaded_file = st.file_uploader("Tải CSV dữ liệu huấn luyện", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
elif df_default is not None:
    df = df_default
else:
    df = None


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

# Train model (Giữ nguyên logic)
X = df.drop(columns=['default'])
y = df['default'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

# ... (Logic tính metrics giữ nguyên) ...
y_pred_in = model.predict(X_train)
y_proba_in = model.predict_proba(X_train)[:, 1]
y_pred_out = model.predict(X_test)
y_proba_out = model.predict_proba(X_test)[:, 1]
# ... (Tính metrics_in và metrics_out giữ nguyên) ...
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

# =========================
# KHỐI LOGIC MỤC TIÊU/TRAIN/DỰ BÁO (GIỮ NGUYÊN)
# =========================
if choice == 'Mục tiêu của mô hình':    
    # ... (Giữ nguyên)
    st.subheader("Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số X1–X14.")
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except Exception:
            st.warning(f"Không tìm thấy {img}")

elif choice == 'Xây dựng mô hình':
    # ... (Giữ nguyên)
    st.subheader("Xây dựng mô hình")
    # ... (Logic hiển thị dữ liệu, trực quan hóa, kết quả đánh giá giữ nguyên) ...
    st.write("##### 1) Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    # ... (các phần khác giữ nguyên) ...

elif choice == 'Sử dụng mô hình để dự báo':
    # ... (Giữ nguyên logic đã sửa ở yêu cầu trước)
    st.subheader("Sử dụng mô hình để dự báo & phân tích AI (3 sheet)") 
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")
    # ... (Logic tiếp theo giữ nguyên) ...
    up_xlsx = st.file_uploader("Tải ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")
    if up_xlsx is not None:
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown("### Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))
        
        data_for_ai = ratios_df.iloc[0].to_dict()

        if set(X.columns) == set(ratios_df.columns):
            with st.expander("Xác suất vỡ nợ dự báo (Tính năng phụ)"):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    show = ratios_df.copy()
                    show["pd"] = probs
                    show["pred_default"] = preds
                    st.dataframe(show.style.format({"pd": "{:.3f}"}))
                except Exception as e:
                    st.warning(f"Lỗi khi tính PD: {e}")

        st.markdown("### Phân tích AI & đề xuất CHO VAY/KHÔNG CHO VAY")
        
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
