# app.py — Streamlit PD + Phân tích Gemini (GIAO DIỆN HIỆN ĐẠI)

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    roc_curve,
    ConfusionMatrixDisplay,
)

# =========================
# THÊM THƯ VIỆN GOOGLE GEMINI
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


MODEL_NAME = "gemini-2.5-flash"

# =========================
# THIẾT LẬP TRANG
# =========================
st.set_page_config(
    page_title="Agribank - Dự báo PD",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS HIỆN ĐẠI VÀ CHUYÊN NGHIỆP
# =========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Poppins:wght@400;500;600;700&display=swap');

    /* Màu chủ đạo Agribank */
    :root {
        --primary-green: #00923F;
        --primary-gold: #FFB81C;
        --dark-green: #006837;
        --light-green: #E8F5E9;
        --bg-gradient: linear-gradient(135deg, #00923F 0%, #006837 100%);
        --gold-gradient: linear-gradient(135deg, #FFB81C 0%, #FFA000 100%);
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15);
    }

    /* Reset và base */
    * {
        font-family: 'Roboto', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-gradient);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* Header with gradient */
    .main-header {
        background: var(--bg-gradient);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        text-align: center;
        animation: slideDown 0.6s ease-out;
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }

    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-green);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }

    .custom-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-5px);
    }

    .custom-card h3 {
        color: var(--primary-green);
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow);
        text-align: center;
        border: 2px solid var(--light-green);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }

    .metric-card:hover {
        border-color: var(--primary-green);
        transform: scale(1.05);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-green);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button styling */
    .stButton > button {
        background: var(--bg-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: var(--shadow-lg) !important;
        transform: translateY(-2px) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed var(--primary-green);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--light-green);
        border-radius: 8px;
        font-weight: 600;
        color: var(--dark-green) !important;
    }

    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }

    .logo-container img {
        border-radius: 10px;
        box-shadow: var(--shadow);
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--primary-green) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-gradient);
        color: white;
    }

    /* Section divider */
    .section-divider {
        height: 3px;
        background: var(--gold-gradient);
        margin: 2rem 0;
        border-radius: 2px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .status-success {
        background: #E8F5E9;
        color: #2E7D32;
    }

    .status-warning {
        background: #FFF3E0;
        color: #E65100;
    }
</style>
""", unsafe_allow_html=True)

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
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <h1>🏦 AGRIBANK - HỆ THỐNG DỰ BÁO XÁC SUẤT VỠ NỢ</h1>
    <p>Dự báo tham số PD (Probability of Default) cho khách hàng doanh nghiệp</p>
</div>
""", unsafe_allow_html=True)

# Logo sidebar
with st.sidebar:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    if os.path.exists("logo-agribank.jpg"):
        st.image("logo-agribank.jpg", width=200)
    st.markdown('</div>', unsafe_allow_html=True)

# Hiển thị trạng thái thư viện AI
if _GEMINI_OK:
    st.sidebar.markdown('<div class="status-badge status-success">✅ Gemini AI: Sẵn sàng</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge status-warning">⚠️ Gemini AI: Chưa cài đặt</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# =========================
# LOAD DỮ LIỆU & TRAIN MODEL
# =========================
np.random.seed(0)

# Load dữ liệu huấn luyện
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df = None

uploaded_file = st.sidebar.file_uploader("📁 Tải CSV dữ liệu huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

if df is None:
    st.info("📊 Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

# Kiểm tra cột cần thiết
required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Thiếu cột: {missing}")
    st.stop()

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
    "Accuracy": accuracy_score(y_train, y_pred_in),
    "Precision": precision_score(y_train, y_pred_in, zero_division=0),
    "Recall": recall_score(y_train, y_pred_in, zero_division=0),
    "F1-Score": f1_score(y_train, y_pred_in, zero_division=0),
    "AUC": roc_auc_score(y_train, y_proba_in),
}
metrics_out = {
    "Accuracy": accuracy_score(y_test, y_pred_out),
    "Precision": precision_score(y_test, y_pred_out, zero_division=0),
    "Recall": recall_score(y_test, y_pred_out, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred_out, zero_division=0),
    "AUC": roc_auc_score(y_test, y_proba_out),
}

# =========================
# MENU
# =========================
menu = ["🎯 Mục tiêu của mô hình", "🔧 Xây dựng mô hình", "🔮 Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('📋 Danh mục tính năng', menu)

# =========================
# TRANG 1: MỤC TIÊU
# =========================
if choice == '🎯 Mục tiêu của mô hình':
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Mục tiêu của mô hình")
    st.markdown("""
    #### Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp

    Hệ thống sử dụng **14 chỉ số tài chính (X1-X14)** để đánh giá:

    - 📈 **Khả năng sinh lời**: Biên lợi nhuận, ROA, ROE
    - 💰 **Thanh khoản**: Tỷ lệ thanh toán hiện hành, thanh toán nhanh
    - 📊 **Cơ cấu nợ**: Tỷ lệ nợ/tài sản, nợ/vốn chủ sở hữu
    - ⚙️ **Hiệu quả hoạt động**: Vòng quay hàng tồn kho, kỳ thu tiền

    **Công nghệ AI:**
    - 🤖 Machine Learning: Logistic Regression
    - 🧠 Gemini AI: Phân tích chuyên sâu và đề xuất cho vay
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dashboard Summary với KPIs
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Thống kê Tổng quan")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tổng mẫu</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tỷ lệ vỡ nợ</div>
            <div class="metric-value">{(y.sum() / len(y) * 100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics_out['Accuracy']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AUC Score</div>
            <div class="metric-value">{metrics_out['AUC']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{metrics_out['F1-Score']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Giải thích chi tiết các chỉ số
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📖 Giải thích các chỉ số tài chính")

    with st.expander("📊 Nhóm chỉ số Khả năng sinh lời (X1-X4)"):
        st.markdown("""
        - **X1 - Biên lợi nhuận gộp**: (Lợi nhuận gộp / Doanh thu) - Đo lường hiệu quả kinh doanh cốt lõi
        - **X2 - Biên lợi nhuận trước thuế**: (LNTT / Doanh thu) - Đo lường khả năng sinh lời tổng thể
        - **X3 - ROA**: (LNTT / Tài sản TB) - Hiệu quả sử dụng tài sản để sinh lời
        - **X4 - ROE**: (LNTT / VCSH TB) - Lợi nhuận trên vốn chủ sở hữu
        """)

    with st.expander("💰 Nhóm chỉ số Cơ cấu nợ (X5-X6)"):
        st.markdown("""
        - **X5 - Tỷ lệ nợ/Tài sản**: Đo lường mức độ đòn bẩy tài chính
        - **X6 - Tỷ lệ nợ/VCSH**: Phản ánh cơ cấu nguồn vốn và rủi ro tài chính
        """)

    with st.expander("🔄 Nhóm chỉ số Thanh khoản (X7-X11)"):
        st.markdown("""
        - **X7 - Thanh toán hiện hành**: (TSNH / Nợ NH) - Khả năng trả nợ ngắn hạn
        - **X8 - Thanh toán nhanh**: ((TSNH - HTK) / Nợ NH) - Thanh khoản loại bỏ hàng tồn kho
        - **X9 - Khả năng trả lãi**: (EBIT / Chi phí lãi vay)
        - **X10 - Khả năng trả nợ gốc**: ((EBIT + Khấu hao) / (Lãi vay + Nợ DH đến hạn))
        - **X11 - Tiền/VCSH**: Tỷ lệ tiền mặt trên vốn chủ
        """)

    with st.expander("⚙️ Nhóm chỉ số Hiệu quả hoạt động (X12-X14)"):
        st.markdown("""
        - **X12 - Vòng quay hàng tồn kho**: (Giá vốn / HTK TB) - Tốc độ luân chuyển hàng tồn
        - **X13 - Kỳ thu tiền bình quân**: 365 / (Doanh thu / Phải thu TB) - Thời gian thu hồi công nợ
        - **X14 - Hiệu suất tài sản**: (Doanh thu / Tài sản TB) - Hiệu quả sử dụng tài sản
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Hiển thị hình ảnh minh họa
    col1, col2, col3 = st.columns(3)
    images = [("hinh2.jpg", col1), ("LogReg_1.png", col2), ("hinh3.png", col3)]

    for img, col in images:
        if os.path.exists(img):
            with col:
                st.image(img, use_container_width=True)

# =========================
# TRANG 2: XÂY DỰNG MÔ HÌNH
# =========================
elif choice == '🔧 Xây dựng mô hình':
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Xây dựng và đánh giá mô hình")
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs cho các phần
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "📈 Trực quan hóa", "🎯 Kết quả đánh giá", "🔍 Ma trận nhầm lẫn"])

    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Dữ liệu huấn luyện")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Tổng số mẫu", len(df))
        with col2:
            st.metric("✅ Không vỡ nợ", (y == 0).sum())
        with col3:
            st.metric("❌ Vỡ nợ", (y == 1).sum())

        st.markdown("##### Dữ liệu mẫu đầu")
        st.dataframe(df.head(5), use_container_width=True)

        st.markdown("##### Thống kê mô tả")
        st.dataframe(df[[f"X_{i}" for i in range(1, 15)]].describe(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Trực quan hóa mối quan hệ giữa biến và xác suất vỡ nợ")

        # Sub-tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Phân tích từng biến", "🔥 Correlation Heatmap", "📈 Distribution Plots"])

        with viz_tab1:
            col = st.selectbox('Chọn biến X muốn phân tích', [f'X_{i}' for i in range(1, 15)])

            if col in df.columns:
                try:
                    # Tạo biểu đồ với Plotly
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Scatter Plot', 'Logistic Regression Curve'))

                    # Scatter plot
                    colors = ['#00923F' if v == 0 else '#FFB81C' for v in df['default']]
                    fig.add_trace(
                        go.Scatter(x=df[col], y=df['default'], mode='markers',
                                  marker=dict(color=colors, size=8, opacity=0.6),
                                  name='Data points'),
                        row=1, col=1
                    )

                    # Logistic regression curve
                    x_range = np.linspace(df[col].min(), df[col].max(), 100)
                    X_temp = df[[col]].copy()
                    y_temp = df['default']
                    lr_temp = LogisticRegression(max_iter=1000)
                    lr_temp.fit(X_temp, y_temp)
                    x_test = pd.DataFrame({col: x_range})
                    y_curve = lr_temp.predict_proba(x_test)[:, 1]

                    fig.add_trace(
                        go.Scatter(x=x_range, y=y_curve, mode='lines',
                                  line=dict(color='#00923F', width=3),
                                  name='Probability curve'),
                        row=1, col=2
                    )

                    fig.update_layout(height=400, showlegend=True)
                    fig.update_xaxes(title_text=col, row=1, col=1)
                    fig.update_xaxes(title_text=col, row=1, col=2)
                    fig.update_yaxes(title_text="Default", row=1, col=1)
                    fig.update_yaxes(title_text="Probability", row=1, col=2)

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Lỗi khi vẽ biểu đồ: {e}")

        with viz_tab2:
            st.markdown("##### Ma trận tương quan giữa các chỉ số tài chính")

            # Compute correlation matrix
            corr_matrix = df[[f'X_{i}' for i in range(1, 15)]].corr()

            # Create heatmap with Plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                title='Correlation Heatmap - 14 Chỉ số Tài chính',
                height=600,
                xaxis_title='',
                yaxis_title=''
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **Giải thích**: Màu xanh đậm = tương quan dương mạnh, Màu đỏ đậm = tương quan âm mạnh")

        with viz_tab3:
            st.markdown("##### Phân phối của các chỉ số tài chính")

            # Select variables to plot
            selected_vars = st.multiselect(
                'Chọn các biến muốn xem phân phối (tối đa 6)',
                [f'X_{i}' for i in range(1, 15)],
                default=[f'X_{i}' for i in range(1, 7)]
            )

            if selected_vars:
                # Create subplot grid
                n_vars = len(selected_vars)
                n_cols = 3
                n_rows = (n_vars + n_cols - 1) // n_cols

                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=selected_vars
                )

                for idx, var in enumerate(selected_vars):
                    row = idx // n_cols + 1
                    col = idx % n_cols + 1

                    # Histogram for non-default
                    fig.add_trace(
                        go.Histogram(
                            x=df[df['default'] == 0][var],
                            name='Non-Default',
                            marker_color='#00923F',
                            opacity=0.7,
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )

                    # Histogram for default
                    fig.add_trace(
                        go.Histogram(
                            x=df[df['default'] == 1][var],
                            name='Default',
                            marker_color='#FFB81C',
                            opacity=0.7,
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )

                fig.update_layout(
                    height=300 * n_rows,
                    showlegend=True,
                    barmode='overlay'
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Kết quả đánh giá mô hình")

        # Metrics cards
        st.markdown("##### 📊 Tập huấn luyện (In-Sample)")
        cols = st.columns(5)
        for idx, (metric_name, value) in enumerate(metrics_in.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric_name}</div>
                    <div class="metric-value">{value:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("##### 🎯 Tập kiểm tra (Out-of-Sample)")
        cols = st.columns(5)
        for idx, (metric_name, value) in enumerate(metrics_out.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric_name}</div>
                    <div class="metric-value">{value:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        # ROC Curve và Feature Importance
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📉 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba_out)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                    name=f'ROC (AUC = {metrics_out["AUC"]:.3f})',
                                    line=dict(color='#00923F', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random',
                                    line=dict(color='gray', width=2, dash='dash')))
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### 📊 Feature Importance")

            # Get feature importance from logistic regression coefficients
            feature_names = [f"X_{i}" for i in range(1, 15)]
            coefficients = model.coef_[0]

            # Create dataframe for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=True)

            # Create horizontal bar chart
            fig = go.Figure()
            colors = ['#FFB81C' if c < 0 else '#00923F' for c in importance_df['Coefficient']]

            fig.add_trace(go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Coefficient'],
                orientation='h',
                marker=dict(color=colors),
                text=np.round(importance_df['Coefficient'], 3),
                textposition='outside'
            ))

            fig.update_layout(
                xaxis_title='Coefficient Value',
                yaxis_title='Features',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **Giải thích**: Xanh = tăng rủi ro vỡ nợ, Vàng = giảm rủi ro vỡ nợ")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix)")

        cm = confusion_matrix(y_test, y_pred_out)

        # Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: Non-Default', 'Predicted: Default'],
            y=['Actual: Non-Default', 'Actual: Default'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Greens',
            showscale=True
        ))

        fig.update_layout(
            title='Confusion Matrix - Test Set',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TRANG 3: SỬ DỤNG MÔ HÌNH
# =========================
elif choice == '🔮 Sử dụng mô hình để dự báo':
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 Sử dụng mô hình để dự báo & Phân tích AI")
    st.info("📋 File Excel phải có đủ 3 sheet: **CDKT** (Cân đối kế toán), **BCTN** (Báo cáo thu nhập), **LCTT** (Lưu chuyển tiền tệ)")
    st.markdown('</div>', unsafe_allow_html=True)

    up_xlsx = st.file_uploader("📂 Tải hồ sơ doanh nghiệp (ho_so_dn.xlsx)", type=["xlsx"], key="ho_so_dn")

    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet
        try:
            with st.spinner('🔄 Đang xử lý dữ liệu từ 3 sheet...'):
                ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"❌ Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Kết quả tính toán 14 chỉ số tài chính")

        # Hiển thị bảng với styling
        styled_df = ratios_df.style.format("{:.4f}").background_gradient(cmap='Greens')
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tạo payload data cho AI
        data_for_ai = ratios_df.iloc[0].to_dict()

        # Dự báo PD
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Dự báo xác suất vỡ nợ (PD)")

        if set(X.columns) == set(ratios_df.columns):
            try:
                probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                preds = (probs >= 0.5).astype(int)

                # Hiển thị kết quả với metrics lớn
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Xác suất vỡ nợ (PD)</div>
                        <div class="metric-value">{probs[0]:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    status = "VỠ NỢ ❌" if preds[0] == 1 else "AN TOÀN ✅"
                    color = "#FFB81C" if preds[0] == 1 else "#00923F"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dự báo</div>
                        <div class="metric-value" style="color: {color};">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    risk_level = "CAO" if probs[0] > 0.7 else "TRUNG BÌNH" if probs[0] > 0.3 else "THẤP"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Mức độ rủi ro</div>
                        <div class="metric-value">{risk_level}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Biểu đồ gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probs[0] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Xác suất vỡ nợ (%)", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#00923F" if probs[0] < 0.5 else "#FFB81C"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#E8F5E9'},
                            {'range': [30, 70], 'color': '#FFF3E0'},
                            {'range': [70, 100], 'color': '#FFEBEE'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Thêm PD vào payload cho AI
                data_for_ai['PD_Probability'] = probs[0]
                data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"

            except Exception as e:
                st.warning(f"⚠️ Không dự báo được PD: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Gemini AI Analysis
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Phân tích AI & Đề xuất cho vay")
        st.markdown("Sử dụng **Gemini AI** để phân tích chuyên sâu và đưa ra khuyến nghị cho vay")

        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])

        with col_btn1:
            analyze_button = st.button("🚀 Yêu cầu AI Phân tích", use_container_width=True)

        with col_btn2:
            # Download data button
            if 'probs' in locals() and len(ratios_df) > 0:
                download_df = ratios_df.copy()
                if 'probs' in locals():
                    download_df['PD_Probability'] = probs[0]
                    download_df['Prediction'] = "Vỡ nợ" if preds[0] == 1 else "An toàn"

                csv = download_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải xuống kết quả",
                    data=csv,
                    file_name=f"du_bao_pd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if analyze_button:
            api_key = st.secrets.get("GEMINI_API_KEY")

            if api_key:
                with st.spinner('🧠 Đang gửi dữ liệu và chờ Gemini AI phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)

                    st.markdown("#### 📋 Kết quả Phân tích từ Gemini AI")

                    # Store in session state for persistence
                    st.session_state['ai_analysis'] = ai_result

                    # Display with better formatting
                    formatted_result = ai_result.replace('\n', '<br>')
                    st.markdown(f"""
                    <div style="background: white;
                                padding: 2rem;
                                border-radius: 15px;
                                border-left: 5px solid #FFB81C;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                line-height: 1.8;
                                font-size: 1.05rem;">
                        {formatted_result}
                    </div>
                    """, unsafe_allow_html=True)

                    # Download AI analysis
                    st.download_button(
                        label="📄 Tải xuống phân tích AI",
                        data=ai_result.encode('utf-8'),
                        file_name=f"phan_tich_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            else:
                st.error("❌ Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình **'GEMINI_API_KEY'** trong Streamlit Secrets.")

        # Display previous analysis if exists
        elif 'ai_analysis' in st.session_state:
            st.markdown("#### 📋 Kết quả Phân tích từ Gemini AI (đã lưu)")
            formatted_saved = st.session_state['ai_analysis'].replace('\n', '<br>')
            st.markdown(f"""
            <div style="background: white;
                        padding: 2rem;
                        border-radius: 15px;
                        border-left: 5px solid #FFB81C;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        line-height: 1.8;
                        font-size: 1.05rem;">
                {formatted_saved}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("📂 Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet: CDKT, BCTN, LCTT) để tính toán các chỉ số, dự báo PD và nhận phân tích AI.")

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="font-size: 0.9rem;">
        © 2024 Agribank - Ngân hàng Nông nghiệp và Phát triển Nông thôn Việt Nam<br>
        Hệ thống Dự báo Xác suất Vỡ nợ (PD) - Phiên bản 2.0
    </p>
</div>
""", unsafe_allow_html=True)
