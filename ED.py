# app.py — Streamlit PD + Phân tích Gemini (CẬP NHẬT THƯ VIỆN)

# =========================
# THƯ VIỆN BẮT BUỘC VÀ BỔ SUNG
# (Cần đảm bảo các gói này được cài đặt, ví dụ trong requirements.txt)
# =========================
from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
# THÊM THƯ VIỆN GOOGLE GEMINI VÀ OPENAI (CHO TƯƠNG THÍCH VỚI REQ CŨ)
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
# FEATURE LABELS - CHỈ SỐ TÀI CHÍNH
# =========================
# NOTE: Cập nhật tên chỉ số cho phù hợp với nghiệp vụ thực tế của ngân hàng
FEATURE_LABELS = {
    "X_1": "Biên lợi nhuận gộp",
    "X_2": "Biên lợi nhuận trước thuế",
    "X_3": "ROA (Lợi nhuận trên tổng tài sản)",
    "X_4": "ROE (Lợi nhuận trên vốn chủ sở hữu)",
    "X_5": "Tỷ số nợ trên tổng tài sản",
    "X_6": "Tỷ số nợ trên vốn chủ sở hữu",
    "X_7": "Tỷ số thanh toán hiện hành",
    "X_8": "Tỷ số thanh toán nhanh",
    "X_9": "Khả năng thanh toán lãi vay",
    "X_10": "Khả năng thanh toán nợ gốc",
    "X_11": "Tỷ số tiền mặt trên vốn chủ sở hữu",
    "X_12": "Vòng quay hàng tồn kho",
    "X_13": "Kỳ thu tiền bình quân (ngày)",
    "X_14": "Hiệu suất sử dụng tài sản",
}

def get_feature_display_name(feature_code):
    """Trả về tên hiển thị đầy đủ: 'X1 – Biên lợi nhuận gộp'"""
    if feature_code in FEATURE_LABELS:
        # Chuyển X_1 thành X1
        code_display = feature_code.replace("_", "")
        return f"{code_display} – {FEATURE_LABELS[feature_code]}"
    return feature_code

# =========================
# HÀM GỌI GEMINI API
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
# TÍNH X1..X14 TỪ 3 SHEET (CDKT/BCTN/LCTT)
# =========================

# Alias các dòng quan trọng trong từng sheet
ALIAS_IS = {
    "doanh_thu_thuan": ["Doanh thu thuần", "Doanh thu bán hàng", "Doanh thu thuần về bán hàng và cung cấp dịch vụ"],
    "gia_von": ["Giá vốn hàng bán"],
    "loi_nhuan_gop": ["Lợi nhuận gộp"],
    "chi_phi_lai_vay": ["Chi phí lãi vay", "Chi phí tài chính (trong đó: chi phí lãi vay)"],
    "loi_nhuan_truoc_thue": ["Tổng lợi nhuận kế toán trước thuế", "Lợi nhuận trước thuế", "Lợi nhuận trước thuế thu nhập DN"],
}
ALIAS_BS = {
    "tong_tai_san": ["Tổng tài sản"],
    "von_chu_so_huu": ["Vốn chủ sở hữu", "Vốn CSH"],
    "no_phai_tra": ["Nợ phải trả"],
    "tai_san_ngan_han": ["Tài sản ngắn hạn"],
    "no_ngan_han": ["Nợ ngắn hạn"],
    "hang_ton_kho": ["Hàng tồn kho"],
    "tien_tdt": ["Tiền và các khoản tương đương tiền", "Tiền và tương đương tiền"],
    "phai_thu_kh": ["Phải thu ngắn hạn của khách hàng", "Phải thu khách hàng"],
    "no_dai_han_den_han": ["Nợ dài hạn đến hạn trả", "Nợ dài hạn đến hạn"],
}
ALIAS_CF = {
    "khau_hao": ["Khấu hao TSCĐ", "Khấu hao", "Chi phí khấu hao"],
}

def _pick_year_cols(df: pd.DataFrame):
    """Chọn 2 cột năm gần nhất từ sheet (ưu tiên cột có nhãn là năm)."""
    numeric_years = []
    for c in df.columns[1:]:
        try:
            y = int(float(str(c).strip()))
            if 1990 <= y <= 2100:
                numeric_years.append((y, c))
        except Exception:
            continue
    if numeric_years:
        numeric_years.sort(key=lambda x: x[0])
        return numeric_years[-2][1], numeric_years[-1][1]
    # fallback: 2 cột cuối
    cols = df.columns[-2:]
    return cols[0], cols[1]

def _get_row_vals(df: pd.DataFrame, aliases: list[str]):
    """Tìm dòng theo alias (contains, không phân biệt hoa/thường). Trả về (prev, cur) theo 2 cột năm gần nhất."""
    label_col = df.columns[0]
    prev_col, cur_col = _pick_year_cols(df)
    mask = False
    for alias in aliases:
        mask = mask | df[label_col].astype(str).str.contains(alias, case=False, na=False)
    rows = df[mask]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]

    def to_num(x):
        try:
            return float(str(x).replace(",", "").replace(" ", ""))
        except Exception:
            return np.nan

    return to_num(row[prev_col]), to_num(row[cur_col])

def compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
    """Đọc 3 sheet CDKT/BCTN/LCTT và tính X1..X14 theo yêu cầu."""
    # Đọc 3 sheet; cần openpyxl trong requirements
    bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
    is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
    cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")

    # ---- KQKD (BCTN)
    DTT_prev, DTT_cur    = _get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
    GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
    LNG_prev, LNG_cur    = _get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
    LNTT_prev, LNTT_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_truoc_thue"])
    LV_prev, LV_cur      = _get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])

    # ---- CĐKT (CDKT)
    TTS_prev, TTS_cur      = _get_row_vals(bs, ALIAS_BS["tong_tai_san"])
    VCSH_prev, VCSH_cur    = _get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
    NPT_prev, NPT_cur      = _get_row_vals(bs, ALIAS_BS["no_phai_tra"])
    TSNH_prev, TSNH_cur    = _get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
    NNH_prev, NNH_cur      = _get_row_vals(bs, ALIAS_BS["no_ngan_han"])
    HTK_prev, HTK_cur      = _get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
    Tien_prev, Tien_cur    = _get_row_vals(bs, ALIAS_BS["tien_tdt"])
    KPT_prev, KPT_cur      = _get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
    NDH_prev, NDH_cur      = _get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])

    # ---- LCTT (LCTT) – lấy Khấu hao nếu có
    KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])

    # Chuẩn hoá số âm thường thấy ở GVHB, chi phí lãi vay, khấu hao
    if pd.notna(GVHB_cur): GVHB_cur = abs(GVHB_cur)
    if pd.notna(LV_cur):    LV_cur    = abs(LV_cur)
    if pd.notna(KH_cur):    KH_cur    = abs(KH_cur)

    # Trung bình đầu/cuối kỳ
    def avg(a, b):
        if pd.isna(a) and pd.isna(b): return np.nan
        if pd.isna(a): return b
        if pd.isna(b): return a
        return (a + b) / 2.0
    TTS_avg  = avg(TTS_cur,  TTS_prev)
    VCSH_avg = avg(VCSH_cur, VCSH_prev)
    HTK_avg  = avg(HTK_cur,  HTK_prev)
    KPT_avg  = avg(KPT_cur,  KPT_prev)

    # EBIT ~ LNTT + chi phí lãi vay (nếu thiếu EBIT riêng)
    EBIT_cur = (LNTT_cur + LV_cur) if (pd.notna(LNTT_cur) and pd.notna(LV_cur)) else np.nan
    # Nợ dài hạn đến hạn trả: có file không ghi -> set 0
    NDH_cur = 0.0 if pd.isna(NDH_cur) else NDH_cur

    def div(a, b):
        return np.nan if (b is None or pd.isna(b) or b == 0) else a / b

    # ==== TÍNH X1..X14 ====
    X1  = div(LNG_cur, DTT_cur)                      # Biên LN gộp
    X2  = div(LNTT_cur, DTT_cur)                     # Biên LNTT
    X3  = div(LNTT_cur, TTS_avg)                     # ROA (trước thuế)
    X4  = div(LNTT_cur, VCSH_avg)                    # ROE (trước thuế)
    X5  = div(NPT_cur,  TTS_cur)                     # Nợ/Tài sản
    X6  = div(NPT_cur,  VCSH_cur)                    # Nợ/VCSH
    X7  = div(TSNH_cur, NNH_cur)                     # Thanh toán hiện hành
    X8  = div((TSNH_cur - HTK_cur) if pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)  # Nhanh
    X9  = div(EBIT_cur, LV_cur)                      # Khả năng trả lãi
    X10 = div((EBIT_cur + (KH_cur if pd.notna(KH_cur) else 0.0)),
              (LV_cur + NDH_cur) if pd.notna(LV_cur) else np.nan)  # Khả năng trả nợ gốc
    X11 = div(Tien_cur, VCSH_cur)                     # Tiền/VCSH
    X12 = div(GVHB_cur, HTK_avg)                     # Vòng quay HTK
    turnover = div(DTT_cur, KPT_avg)                # Vòng quay phải thu
    X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else np.nan  # Kỳ thu tiền BQ
    X14 = div(DTT_cur, TTS_avg)                      # Hiệu suất sử dụng tài sản

    ratios = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14]],
                          columns=[f"X_{i}" for i in range(1, 15)])
    return ratios

# =========================
# UI HELPER FUNCTIONS
# =========================
def load_css(file_path):
    """Load CSS file into Streamlit"""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_path}")

def render_header():
    """Render Agribank-style header with logo"""
    try:
        # Create header with logo
        st.markdown("""
        <div class="agribank-header">
            <img src="data:image/jpeg;base64,{}" width="80" height="80" alt="Agribank Logo">
            <h1>Đánh giá rủi ro tín dụng khách hàng doanh nghiệp</h1>
        </div>
        """.format(_get_logo_base64()), unsafe_allow_html=True)
    except Exception:
        # Fallback without logo
        st.markdown("""
        <div class="agribank-header">
            <h1>Đánh giá rủi ro tín dụng khách hàng doanh nghiệp</h1>
        </div>
        """, unsafe_allow_html=True)

def _get_logo_base64():
    """Get base64 encoded logo"""
    import base64
    try:
        with open("logo-agribank.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def render_metric_card(title, value, icon="📊"):
    """Render a metric card with styling"""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <p>{value}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# UI & TRAIN MODEL
# =========================
np.random.seed(0)

# Load CSS theme
load_css("ui/theme.css")

# Render header
render_header()

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

# Dự báo & đánh giá
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

# =========================
# TABS LAYOUT - 3 TAB NGANG
# =========================
tab1, tab2, tab3 = st.tabs([
    "📈 Sử dụng mô hình dự báo",
    "📊 Phân tích dữ liệu",
    "📚 Tài liệu & Hướng dẫn"
])

# TAB 1: SỬ DỤNG MÔ HÌNH DỰ BÁO (Default tab)
with tab1:
    st.subheader("Sử dụng mô hình để dự báo & phân tích AI (3 sheet)")
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")

    up_xlsx = st.file_uploader("Tải ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")
    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown("### Kết quả tính toán 14 chỉ số tài chính")

        # Create a display dataframe with readable labels
        display_df = ratios_df.copy()
        display_df.columns = [get_feature_display_name(col) for col in display_df.columns]

        # Display in a styled container
        st.markdown('<div class="feature-table">', unsafe_allow_html=True)
        st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tạo payload data cho AI
        data_for_ai = ratios_df.iloc[0].to_dict()

        # Metric cards for key indicators
        st.markdown("### 📊 Các chỉ số quan trọng")
        col1, col2, col3 = st.columns(3)

        with col1:
            render_metric_card(
                "Biên lợi nhuận gộp (X1)",
                f"{ratios_df['X_1'].iloc[0]:.2%}" if pd.notna(ratios_df['X_1'].iloc[0]) else "N/A",
                "💰"
            )
        with col2:
            render_metric_card(
                "ROA (X3)",
                f"{ratios_df['X_3'].iloc[0]:.2%}" if pd.notna(ratios_df['X_3'].iloc[0]) else "N/A",
                "📈"
            )
        with col3:
            render_metric_card(
                "Tỷ số thanh toán (X7)",
                f"{ratios_df['X_7'].iloc[0]:.2f}" if pd.notna(ratios_df['X_7'].iloc[0]) else "N/A",
                "💵"
            )

        # (Tuỳ chọn) dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        if set(X.columns) == set(ratios_df.columns):
            with st.expander("🔍 Xác suất vỡ nợ dự báo (nếu đã huấn luyện ở trên)"):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    show = ratios_df.copy()
                    show["pd"] = probs
                    show["pred_default"] = preds
                    st.dataframe(show.style.format({"pd": "{:.3f}"}), use_container_width=True)
                except Exception as e:
                    st.warning(f"Không dự báo được PD: {e}")

        # Gemini Phân tích & khuyến nghị
        st.markdown("### 🤖 Phân tích AI & đề xuất CHO VAY/KHÔNG CHO VAY")

        # Thêm các chỉ số PD nếu đã tính được vào payload
        if 'probs' in locals():
            data_for_ai['PD_Probability'] = probs[0]
            data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"

        if st.button("🚀 Yêu cầu AI Phân tích", use_container_width=True):
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

# TAB 2: PHÂN TÍCH DỮ LIỆU
with tab2:
    st.subheader("Phân tích dữ liệu & Xây dựng mô hình")

    st.write("##### 1) Hiển thị dữ liệu")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Dữ liệu đầu")
        st.dataframe(df.head(3), use_container_width=True)
    with col_b:
        st.caption("Dữ liệu cuối")
        st.dataframe(df.tail(3), use_container_width=True)

    st.write("##### 2) Trực quan hóa dữ liệu")

    # Feature selection
    col = st.selectbox(
        'Chọn biến X muốn phân tích',
        options=[f"X_{i}" for i in range(1, 15)],
        format_func=lambda x: get_feature_display_name(x),
        index=0
    )

    if col in df.columns:
        try:
            # Plotly scatter plot with logistic curve
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Create scatter plot
            fig = px.scatter(
                df,
                x=col,
                y='default',
                opacity=0.5,
                labels={col: get_feature_display_name(col), 'default': 'Xác suất vỡ nợ'},
                color='default',
                color_continuous_scale=['#D4AF37', '#800000']
            )

            # Add logistic regression curve
            x_range = np.linspace(df[col].min(), df[col].max(), 100)
            X_temp = df[[col]].copy()
            y_temp = df['default']
            lr_temp = LogisticRegression(max_iter=1000)
            lr_temp.fit(X_temp, y_temp)
            x_test = pd.DataFrame({col: x_range})
            y_curve = lr_temp.predict_proba(x_test)[:, 1]

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_curve,
                    mode='lines',
                    name='Logistic Regression',
                    line=dict(color='#800000', width=3)
                )
            )

            # Update layout with Plotly 5 compatible syntax
            fig.update_layout(
                title={
                    'text': f'Phân tích {get_feature_display_name(col)}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title=get_feature_display_name(col),
                yaxis_title='Xác suất vỡ nợ',
                height=500,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")
    else:
        st.warning("Biến không tồn tại trong dữ liệu.")

    st.write("##### 3) Kết quả đánh giá mô hình")

    # Display metrics in cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Accuracy (Test)", f"{metrics_out['accuracy_out']:.3f}", "🎯")
    with col2:
        render_metric_card("Precision (Test)", f"{metrics_out['precision_out']:.3f}", "✅")
    with col3:
        render_metric_card("Recall (Test)", f"{metrics_out['recall_out']:.3f}", "🔍")
    with col4:
        render_metric_card("F1 Score (Test)", f"{metrics_out['f1_out']:.3f}", "⚖️")
    with col5:
        render_metric_card("AUC (Test)", f"{metrics_out['auc_out']:.3f}", "📊")

    # Full metrics table
    with st.expander("Xem chi tiết các chỉ số đánh giá"):
        dt = pd.DataFrame([metrics_in | metrics_out])
        st.dataframe(dt, use_container_width=True)

    st.write("##### 4) Ma trận nhầm lẫn (Test Set)")
    cm = confusion_matrix(y_test, y_pred_out)

    # Create Plotly heatmap for confusion matrix
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
        x=['Non-Default', 'Default'],
        y=['Non-Default', 'Default'],
        color_continuous_scale=['#FAFAFA', '#800000'],
        text_auto=True
    )
    fig_cm.update_layout(
        title={
            'text': 'Ma trận nhầm lẫn - Test Set',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Additional analysis charts
    st.write("##### 5) Biểu đồ phân tích bổ sung")

    chart_type = st.radio(
        "Chọn loại biểu đồ",
        ["Phân bố chỉ số", "So sánh giá trị trung bình", "Correlation Matrix"],
        horizontal=True
    )

    if chart_type == "Phân bố chỉ số":
        selected_feature = st.selectbox(
            'Chọn chỉ số',
            options=[f"X_{i}" for i in range(1, 15)],
            format_func=lambda x: get_feature_display_name(x),
            key="hist_select"
        )
        if selected_feature in df.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_hist = px.histogram(
                df,
                x=selected_feature,
                color='default',
                marginal="box",
                nbins=30,
                labels={selected_feature: get_feature_display_name(selected_feature)},
                color_discrete_map={0: '#D4AF37', 1: '#800000'}
            )
            fig_hist.update_layout(
                title={
                    'text': f'Phân bố {get_feature_display_name(selected_feature)}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=500
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif chart_type == "So sánh giá trị trung bình":
        # Calculate means by default status
        means_df = df.groupby('default')[[f"X_{i}" for i in range(1, 15)]].mean().T
        means_df.columns = ['Non-Default', 'Default']
        means_df['Feature'] = [get_feature_display_name(f"X_{i}") for i in range(1, 15)]

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_bar = px.bar(
            means_df,
            x='Feature',
            y=['Non-Default', 'Default'],
            barmode='group',
            labels={'value': 'Giá trị trung bình', 'Feature': 'Chỉ số'},
            color_discrete_map={'Non-Default': '#D4AF37', 'Default': '#800000'}
        )
        fig_bar.update_layout(
            title={
                'text': 'So sánh giá trị trung bình các chỉ số theo trạng thái',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif chart_type == "Correlation Matrix":
        # Calculate correlation matrix
        corr_matrix = df[[f"X_{i}" for i in range(1, 15)]].corr()

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=[get_feature_display_name(f"X_{i}") for i in range(1, 15)],
            y=[get_feature_display_name(f"X_{i}") for i in range(1, 15)],
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto='.2f'
        )
        fig_corr.update_layout(
            title={
                'text': 'Ma trận tương quan giữa các chỉ số',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=700,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: TÀI LIỆU & HƯỚNG DẪN
with tab3:
    st.subheader("Tài liệu & Hướng dẫn sử dụng")

    st.markdown("### 📖 Giới thiệu hệ thống")
    st.markdown("""
    **Hệ thống đánh giá rủi ro tín dụng khách hàng doanh nghiệp** là công cụ hỗ trợ quyết định
    cho vay dựa trên phân tích 14 chỉ số tài chính quan trọng.
    """)

    # Display images
    st.markdown("### 🖼️ Minh họa mô hình")
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img, use_column_width=True)
        except Exception:
            st.info(f"Hình minh họa {img} sẽ được cập nhật sau")

    st.markdown("### 📊 Chi tiết 14 chỉ số tài chính")

    # Create a nice table with feature descriptions
    features_info = []
    for i in range(1, 15):
        feature_code = f"X_{i}"
        features_info.append({
            "Mã": f"X{i}",
            "Tên chỉ số": FEATURE_LABELS[feature_code],
            "Nhóm": _get_feature_group(i)
        })

    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("### 📝 Hướng dẫn sử dụng")
    st.markdown("""
    #### Bước 1: Chuẩn bị dữ liệu
    - Chuẩn bị file Excel (.xlsx) chứa 3 sheet:
        - **CDKT**: Cân đối kế toán
        - **BCTN**: Báo cáo thu nhập
        - **LCTT**: Lưu chuyển tiền tệ

    #### Bước 2: Tải file và phân tích
    - Chuyển sang tab **"Sử dụng mô hình dự báo"**
    - Tải file Excel lên hệ thống
    - Hệ thống sẽ tự động tính toán 14 chỉ số tài chính

    #### Bước 3: Xem kết quả
    - Xem bảng kết quả 14 chỉ số với tên đầy đủ
    - Xem các chỉ số quan trọng được highlight
    - Xem xác suất vỡ nợ dự báo

    #### Bước 4: Phân tích AI
    - Nhấn nút **"Yêu cầu AI Phân tích"**
    - Đọc kết quả phân tích và khuyến nghị từ Gemini AI
    - Ra quyết định cho vay dựa trên phân tích tổng hợp
    """)

    st.markdown("### ⚙️ Cấu hình API")
    st.info("""
    Để sử dụng tính năng phân tích AI, cần cấu hình **GEMINI_API_KEY** trong Streamlit Secrets.
    Liên hệ quản trị viên hệ thống để được hỗ trợ.
    """)

def _get_feature_group(index):
    """Helper function to categorize features into groups"""
    if index in [1, 2, 3, 4]:
        return "Khả năng sinh lời"
    elif index in [5, 6]:
        return "Cơ cấu nợ"
    elif index in [7, 8, 9, 10, 11]:
        return "Thanh khoản"
    elif index in [12, 13, 14]:
        return "Hiệu quả hoạt động"
    return "Khác"
