# app.py — Streamlit PD + Phân tích Gemini (GIAO DIỆN CHUYÊN NGHIỆP)

from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

# =========================
# THƯ VIỆN GEMINI
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
# CUSTOM CSS - GIAO DIỆN CHUYÊN NGHIỆP
# =========================
st.set_page_config(
    page_title="Hệ thống Đánh giá Rủi ro Tín dụng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --background-light: #f8f9fa;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    /* Risk level badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2ecc71 0%, #f39c12 50%, #e74c3c 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HÀM GỌI GEMINI API
# =========================
def get_ai_analysis(data_payload: dict, api_key: str) -> str:
    """Sử dụng Gemini API để phân tích chỉ số tài chính."""
    if not _GEMINI_OK:
        return "❌ Lỗi: Thiếu thư viện google-genai (cần cài đặt: pip install google-genai)."

    client = genai.Client(api_key=api_key)

    sys_prompt = (
        "Bạn là chuyên gia phân tích tín dụng doanh nghiệp hàng đầu tại ngân hàng với 15 năm kinh nghiệm. "
        "Phân tích toàn diện dựa trên 14 chỉ số tài chính (X1..X14) và xác suất vỡ nợ (PD). "
        "Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4) Hiệu quả hoạt động. "
        "Kết thúc bằng khuyến nghị rõ ràng: **CHO VAY** hoặc **KHÔNG CHO VAY**, kèm 2–3 điều kiện cụ thể. "
        "Viết bằng tiếng Việt chuyên nghiệp, sử dụng markdown để format đẹp với headers, bullet points."
    )
    
    user_prompt = f"""
Phân tích hồ sơ tín dụng với các thông tin sau:

**DỮ LIỆU TÀI CHÍNH:**
{str(data_payload)}

Hãy đưa ra phân tích chi tiết theo cấu trúc:
- **Tổng quan**: Đánh giá tổng thể tình hình doanh nghiệp
- **Điểm mạnh**: 3-4 điểm nổi bật
- **Điểm yếu**: 3-4 vấn đề cần lưu ý
- **Phân tích chuyên sâu**: Theo 4 khía cạnh (sinh lời, thanh khoản, nợ, hiệu quả)
- **Mức độ rủi ro**: THẤP / TRUNG BÌNH / CAO
- **KHUYẾN NGHỊ CUỐI CÙNG**: CHO VAY hoặc KHÔNG CHO VAY (in đậm, in hoa)
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}],
            config={"system_instruction": sys_prompt, "temperature": 0.3, "max_output_tokens": 2048}
        )
        return response.text
    except APIError as e:
        return f"❌ Lỗi gọi API Gemini: {e}"
    except Exception as e:
        return f"❌ Lỗi không xác định: {e}"

# =========================
# TÍNH X1..X14 TỪ 3 SHEET
# =========================
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
    cols = df.columns[-2:]
    return cols[0], cols[1]

def _get_row_vals(df: pd.DataFrame, aliases: list[str]):
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

@st.cache_data
def compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
    """Đọc 3 sheet CDKT/BCTN/LCTT và tính X1..X14 theo yêu cầu."""
    try:
        # Đọc 3 sheet
        bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
        is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
        cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Lỗi đọc file Excel: {e}. Kiểm tra file có đủ 3 sheet CDKT, BCTN, LCTT")

    # ---- BCTN (Báo cáo thu nhập)
    DTT_prev, DTT_cur    = _get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
    GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
    LNG_prev, LNG_cur    = _get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
    LNTT_prev, LNTT_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_truoc_thue"])
    LV_prev, LV_cur      = _get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])

    # ---- CDKT (Cân đối kế toán)
    TTS_prev, TTS_cur      = _get_row_vals(bs, ALIAS_BS["tong_tai_san"])
    VCSH_prev, VCSH_cur    = _get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
    NPT_prev, NPT_cur      = _get_row_vals(bs, ALIAS_BS["no_phai_tra"])
    TSNH_prev, TSNH_cur    = _get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
    NNH_prev, NNH_cur      = _get_row_vals(bs, ALIAS_BS["no_ngan_han"])
    HTK_prev, HTK_cur      = _get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
    Tien_prev, Tien_cur    = _get_row_vals(bs, ALIAS_BS["tien_tdt"])
    KPT_prev, KPT_cur      = _get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
    NDH_prev, NDH_cur      = _get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])

    # ---- LCTT (Lưu chuyển tiền tệ)
    KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])

    # Chuẩn hoá số âm (giá vốn, chi phí thường âm trong báo cáo)
    if pd.notna(GVHB_cur): 
        GVHB_cur = abs(GVHB_cur)
    if pd.notna(LV_cur):
        LV_cur = abs(LV_cur)
    if pd.notna(KH_cur):
        KH_cur = abs(KH_cur)

    # Hàm tính trung bình
    def avg(a, b):
        if pd.isna(a) and pd.isna(b): 
            return np.nan
        if pd.isna(a): 
            return b
        if pd.isna(b): 
            return a
        return (a + b) / 2.0
    
    # Tính trung bình đầu cuối kỳ
    TTS_avg  = avg(TTS_cur,  TTS_prev)
    VCSH_avg = avg(VCSH_cur, VCSH_prev)
    HTK_avg  = avg(HTK_cur,  HTK_prev)
    KPT_avg  = avg(KPT_cur,  KPT_prev)

    # Tính EBIT
    EBIT_cur = (LNTT_cur + LV_cur) if (pd.notna(LNTT_cur) and pd.notna(LV_cur)) else np.nan
    
    # Nợ dài hạn đến hạn (nếu không có thì = 0)
    NDH_cur = 0.0 if pd.isna(NDH_cur) else NDH_cur

    # Hàm chia an toàn
    def div(a, b):
        if b is None or pd.isna(b) or b == 0:
            return np.nan
        if a is None or pd.isna(a):
            return np.nan
        return float(a) / float(b)

    # ==== TÍNH X1..X14 ====
    X1  = div(LNG_cur, DTT_cur)                      # Biên LN gộp
    X2  = div(LNTT_cur, DTT_cur)                     # Biên LNTT
    X3  = div(LNTT_cur, TTS_avg)                     # ROA (trước thuế)
    X4  = div(LNTT_cur, VCSH_avg)                    # ROE (trước thuế)
    X5  = div(NPT_cur,  TTS_cur)                     # Nợ/Tài sản
    X6  = div(NPT_cur,  VCSH_cur)                    # Nợ/VCSH
    X7  = div(TSNH_cur, NNH_cur)                     # Thanh toán hiện hành
    
    # X8: Thanh toán nhanh
    TSNH_tru_HTK = None
    if pd.notna(TSNH_cur) and pd.notna(HTK_cur):
        TSNH_tru_HTK = TSNH_cur - HTK_cur
    X8  = div(TSNH_tru_HTK, NNH_cur)
    
    X9  = div(EBIT_cur, LV_cur)                      # Khả năng trả lãi
    
    # X10: Khả năng trả nợ gốc
    tu_so_X10 = None
    if pd.notna(EBIT_cur):
        KH_val = KH_cur if pd.notna(KH_cur) else 0.0
        tu_so_X10 = EBIT_cur + KH_val
    
    mau_so_X10 = None
    if pd.notna(LV_cur):
        mau_so_X10 = LV_cur + NDH_cur
    
    X10 = div(tu_so_X10, mau_so_X10)
    
    X11 = div(Tien_cur, VCSH_cur)                    # Tiền/VCSH
    X12 = div(GVHB_cur, HTK_avg)                     # Vòng quay HTK
    
    # X13: Kỳ thu tiền BQ
    turnover = div(DTT_cur, KPT_avg)
    X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else np.nan
    
    X14 = div(DTT_cur, TTS_avg)                      # Hiệu suất sử dụng tài sản

    # Tạo DataFrame kết quả
    ratios = pd.DataFrame(
        [[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14]],
        columns=[f"X_{i}" for i in range(1, 15)]
    )
    
    return ratios

# =========================
# GIAO DIỆN CHÍNH
# =========================
np.random.seed(0)

# Header chuyên nghiệp
st.markdown("""
<div class="main-header">
    <h1>🏦 HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG</h1>
    <p>Powered by Machine Learning & Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ CÀI ĐẶT HỆ THỐNG")
    st.markdown(f"""
    <div class="metric-card">
        <h3>🤖 Trạng thái AI</h3>
        <p>{'✅ Gemini: Sẵn sàng' if _GEMINI_OK else '⚠️ Gemini: Chưa cài đặt'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 THÔNG TIN MÔ HÌNH")
    st.info("""
    **Model**: Logistic Regression  
    **Features**: 14 chỉ số tài chính  
    **AI Engine**: Google Gemini 2.5
    """)

# Load CSV
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df = None

uploaded_file = st.file_uploader("📤 **Tải dữ liệu huấn luyện (CSV)**", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

if df is None:
    st.warning("⚠️ Vui lòng tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Thiếu cột: {missing}")
    st.stop()

with st.expander("📊 Xem thống kê mô tả dữ liệu"):
    st.dataframe(df[[f"X_{i}" for i in range(1, 15)]].describe(), use_container_width=True)

# Train model
X = df.drop(columns=['default'])
y = df['default'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

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

# MENU
menu = ["🎯 Mục tiêu", "🔧 Xây dựng mô hình", "🚀 Dự báo & Phân tích"]
choice = st.sidebar.radio('📋 **CHỨC NĂNG**', menu)

if choice == '🎯 Mục tiêu':
    st.markdown("## 🎯 Mục tiêu của Hệ thống")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Dự báo Xác suất Vỡ nợ (PD)
        
        Hệ thống sử dụng **Machine Learning** kết hợp **Gemini AI** để:
        
        - ✅ Tính toán 14 chỉ số tài chính từ 3 báo cáo (CDKT, BCTN, LCTT)
        - ✅ Dự báo xác suất vỡ nợ với độ chính xác cao
        - ✅ Phân tích chuyên sâu bởi AI
        - ✅ Đưa ra khuyến nghị cho vay rõ ràng
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 CÁC CHỈ SỐ</h3>
            <p><b>X1-X4:</b> Sinh lời</p>
            <p><b>X5-X6:</b> Đòn bẩy</p>
            <p><b>X7-X11:</b> Thanh khoản</p>
            <p><b>X12-X14:</b> Hiệu quả</p>
        </div>
        """, unsafe_allow_html=True)
    
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img, use_column_width=True)
        except:
            pass

elif choice == '🔧 Xây dựng mô hình':
    st.markdown("## 🔧 Xây dựng & Đánh giá Mô hình")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "📈 Trực quan", "🎯 Đánh giá", "🔍 Ma trận"])
    
    with tab1:
        st.markdown("### Dữ liệu huấn luyện")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔝 Top 3 records**")
            st.dataframe(df.head(3), use_column_width=True)
        with col2:
            st.markdown("**🔽 Bottom 3 records**")
            st.dataframe(df.tail(3), use_column_width=True)
    
    with tab2:
        st.markdown("### Trực quan hóa mối quan hệ")
        col = st.selectbox('Chọn biến X để vẽ', [f"X_{i}" for i in range(1, 15)])
        if col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=col, y='default', alpha=0.4, ax=ax)
                x_range = np.linspace(df[col].min(), df[col].max(), 100)
                X_temp = df[[col]].copy()
                y_temp = df['default']
                lr_temp = LogisticRegression(max_iter=1000)
                lr_temp.fit(X_temp, y_temp)
                x_test = pd.DataFrame({col: x_range})
                y_curve = lr_temp.predict_proba(x_test)[:, 1]
                ax.plot(x_range, y_curve, color='red', linewidth=3, label='Logistic Curve')
                ax.set_ylabel('Xác suất default', fontsize=12)
                ax.set_xlabel(col, fontsize=12)
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
    
    with tab3:
        st.markdown("### Kết quả đánh giá mô hình")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Accuracy (Test)", f"{metrics_out['accuracy_out']:.1%}")
        with col2:
            st.metric("🎯 Precision (Test)", f"{metrics_out['precision_out']:.1%}")
        with col3:
            st.metric("🎯 Recall (Test)", f"{metrics_out['recall_out']:.1%}")
        with col4:
            st.metric("🎯 F1-Score (Test)", f"{metrics_out['f1_out']:.1%}")
        with col5:
            st.metric("🎯 AUC (Test)", f"{metrics_out['auc_out']:.3f}")
        
        st.markdown("---")
        dt = pd.DataFrame([metrics_in | metrics_out])
        st.dataframe(dt.style.format("{:.4f}"), use_column_width=True)
    
    with tab4:
        st.markdown("### Ma trận nhầm lẫn (Test Set)")
        cm = confusion_matrix(y_test, y_pred_out)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig2, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig2)
        plt.close()

elif choice == '🚀 Dự báo & Phân tích':
    st.markdown("## 🚀 Dự báo Rủi ro & Phân tích AI")
    
    st.markdown("""
    <div class="info-box">
        📁 <b>Yêu cầu:</b> File Excel phải có đủ 3 sheet: <b>CDKT</b> | <b>BCTN</b> | <b>LCTT</b>
    </div>
    """, unsafe_allow_html=True)
    
    up_xlsx = st.file_uploader("📂 **Tải hồ sơ doanh nghiệp (Excel)**", type=["xlsx"], key="ho_so_dn")
    
    if up_xlsx is not None:
        try:
            with st.spinner('⏳ Đang xử lý dữ liệu...'):
                ratios_df = compute_ratios_from_three_sheets(up_xlsx)
            st.success("✅ Tính toán X1-X14 thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi tính X1…X14: {e}")
            st.stop()

        # Tabs cho kết quả
        tab1, tab2, tab3 = st.tabs(["📊 Chỉ số tài chính", "🎯 Dự báo PD", "🤖 Phân tích AI"])
        
        with tab1:
            st.markdown("### 📊 Bộ chỉ số tài chính X1-X14")
            st.dataframe(ratios_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1), 
                        use_column_width=True)
            
            with st.expander("ℹ️ Giải thích chi tiết các chỉ số"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **📈 Khả năng sinh lời:**
                    - **X1**: Biên lợi nhuận gộp
                    - **X2**: Biên lợi nhuận trước thuế
                    - **X3**: ROA (Sinh lời trên tài sản)
                    - **X4**: ROE (Sinh lời trên vốn CSH)
                    
                    **💰 Cơ cấu nợ:**
                    - **X5**: Tỷ lệ Nợ/Tài sản
                    - **X6**: Tỷ lệ Nợ/VCSH
                    - **X9**: Khả năng trả lãi
                    - **X10**: Khả năng trả nợ gốc
                    """)
                with col2:
                    st.markdown("""
                    **💧 Thanh khoản:**
                    - **X7**: Tỷ lệ thanh toán hiện hành
                    - **X8**: Tỷ lệ thanh toán nhanh
                    - **X11**: Tỷ lệ Tiền/VCSH
                    
                    **⚡ Hiệu quả hoạt động:**
                    - **X12**: Vòng quay hàng tồn kho
                    - **X13**: Kỳ thu tiền bình quân (ngày)
                    - **X14**: Hiệu suất sử dụng tài sản
                    """)
        
        with tab2:
            st.markdown("### 🎯 Kết quả Dự báo Xác suất Vỡ nợ (PD)")
            
            data_for_ai = ratios_df.iloc[0].to_dict()
            
            if set(X.columns) == set(ratios_df.columns):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    # Metrics chuyên nghiệp
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 Xác suất PD</h3>
                            <div class="metric-value">{probs[0]:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        pred_text = "VỠ NỢ" if preds[0] == 1 else "AN TOÀN"
                        pred_color = "#e74c3c" if preds[0] == 1 else "#2ecc71"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>✅ Kết luận</h3>
                            <div class="metric-value" style="color: {pred_color};">{pred_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        if probs[0] < 0.3:
                            risk_level = "THẤP"
                            risk_class = "risk-low"
                            risk_icon = "🟢"
                        elif probs[0] < 0.5:
                            risk_level = "TRUNG BÌNH"
                            risk_class = "risk-medium"
                            risk_icon = "🟡"
                        else:
                            risk_level = "CAO"
                            risk_class = "risk-high"
                            risk_icon = "🔴"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⚠️ Mức độ rủi ro</h3>
                            <span class="risk-badge {risk_class}">{risk_icon} {risk_level}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        confidence = max(probs[0], 1-probs[0])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎯 Độ tin cậy</h3>
                            <div class="metric-value">{confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar với màu gradient
                    st.markdown("---")
                    st.markdown("#### 📏 Thang đánh giá rủi ro")
                    st.progress(probs[0])
                    
                    col_left, col_mid, col_right = st.columns([1,1,1])
                    with col_left:
                        st.caption("🟢 0% - An toàn")
                    with col_mid:
                        st.caption("🟡 30-50% - Cảnh báo")
                    with col_right:
                        st.caption("🔴 >50% - Nguy hiểm")
                    
                    st.markdown("---")
                    
                    # Đánh giá chi tiết
                    if probs[0] < 0.3:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ ĐÁNH GIÁ: RỦI RO THẤP</h4>
                            <p>Doanh nghiệp có tình hình tài chính tốt, khả năng trả nợ cao. Đề xuất <b>PHÊ DUYỆT CHO VAY</b> với điều kiện chuẩn.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif probs[0] < 0.5:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ ĐÁNH GIÁ: RỦI RO TRUNG BÌNH</h4>
                            <p>Cần xem xét kỹ lưỡng. Đề xuất <b>CHO VAY CÓ ĐIỀU KIỆN</b>: Yêu cầu tài sản đảm bảo, giám sát chặt chẽ, hạn mức vay phù hợp.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="danger-box">
                            <h4>🚫 ĐÁNH GIÁ: RỦI RO CAO</h4>
                            <p>Doanh nghiệp có nguy cơ vỡ nợ cao. Đề xuất <b>TỪ CHỐI CHO VAY</b> hoặc yêu cầu tài sản thế chấp giá trị cao (>150% giá trị khoản vay).</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Bảng chi tiết
                    st.markdown("#### 📋 Bảng chi tiết đầy đủ")
                    show = ratios_df.copy()
                    show["PD (%)"] = probs * 100
                    show["Dự báo"] = ["🔴 VỠ NỢ" if p == 1 else "🟢 AN TOÀN" for p in preds]
                    show["Mức rủi ro"] = [f"{risk_icon} {risk_level}"]
                    
                    st.dataframe(show.style.format({
                        **{f"X_{i}": "{:.4f}" for i in range(1, 15)},
                        "PD (%)": "{:.2f}%"
                    }).background_gradient(subset=['PD (%)'], cmap='RdYlGn_r'), 
                    use_column_width=True)
                    
                    # Lưu data cho AI
                    data_for_ai['PD_Probability'] = probs[0]
                    data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
                    data_for_ai['Risk_Level'] = risk_level
                    
                except Exception as e:
                    st.error(f"❌ Không dự báo được PD: {e}")
            else:
                st.error("⚠️ Cấu trúc dữ liệu không khớp với mô hình huấn luyện!")
        
        with tab3:
            st.markdown("### 🤖 Phân tích Chuyên sâu bằng Gemini AI")
            
            st.markdown("""
            <div class="info-box">
                💡 <b>AI sẽ phân tích:</b> Khả năng sinh lời, Thanh khoản, Cơ cấu nợ, Hiệu quả hoạt động và đưa ra khuyến nghị cuối cùng.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 **Phân tích bằng Gemini AI**", type="primary", use_container_width=True):
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if api_key:
                    with st.spinner('⏳ Gemini AI đang phân tích hồ sơ tín dụng... Vui lòng đợi 10-15 giây'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        
                        st.markdown("---")
                        st.markdown("### 📋 BÁO CÁO PHÂN TÍCH TỪ GEMINI AI")
                        
                        # Hiển thị kết quả trong box đẹp
                        st.markdown(f"""
                        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            {ai_result}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Tải báo cáo (Text)",
                            data=ai_result,
                            file_name=f"bao_cao_phan_tich_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("""
                    ❌ **Lỗi:** Không tìm thấy GEMINI_API_KEY trong Streamlit Secrets.
                    
                    **Hướng dẫn:**
                    1. Lấy API key tại: https://aistudio.google.com/apikey
                    2. Thêm vào Settings → Secrets: `GEMINI_API_KEY = "your-key-here"`
                    """)
    else:
        # Hướng dẫn khi chưa upload
        st.markdown("""
        <div class="info-box">
            <h3>📁 Chưa có file dữ liệu</h3>
            <p>Vui lòng tải file <b>ho_so_dn.xlsx</b> để bắt đầu phân tích rủi ro tín dụng.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 **Hướng dẫn chi tiết**"):
            st.markdown("""
            ### 📂 Cấu trúc file Excel yêu cầu:
            
            File Excel phải có **đúng 3 sheet** với tên cụ thể:
            
            #### 1️⃣ Sheet **CDKT** (Cân đối kế toán)
            Các chỉ tiêu cần có:
            - Tổng tài sản
            - Vốn chủ sở hữu
            - Nợ phải trả
            - Tài sản ngắn hạn
            - Nợ ngắn hạn
            - Hàng tồn kho
            - Tiền và tương đương tiền
            - Phải thu khách hàng
            - Nợ dài hạn đến hạn trả
            
            #### 2️⃣ Sheet **BCTN** (Báo cáo thu nhập)
            Các chỉ tiêu cần có:
            - Doanh thu thuần
            - Giá vốn hàng bán
            - Lợi nhuận gộp
            - Chi phí lãi vay
            - Lợi nhuận trước thuế
            
            #### 3️⃣ Sheet **LCTT** (Lưu chuyển tiền tệ)
            Các chỉ tiêu cần có:
            - Khấu hao TSCĐ
            
            ---
            
            ### 🔑 Cấu hình Gemini API:
            
            1. **Lấy API Key miễn phí:**
               - Truy cập: https://aistudio.google.com/apikey
               - Đăng nhập bằng Google Account
               - Tạo API Key mới
            
            2. **Thêm vào Streamlit:**
               - Vào Settings → Secrets
               - Thêm dòng: `GEMINI_API_KEY = "your-api-key-here"`
               - Save và restart app
            
            ---
            
            ### ⚡ Lưu ý quan trọng:
            - File phải có định dạng **.xlsx** (không hỗ trợ .xls)
            - Tên sheet phải **chính xác** (CDKT, BCTN, LCTT)
            - Dữ liệu phải có **ít nhất 2 năm** (năm trước và năm sau)
            - Các chỉ tiêu có thể viết hoa/thường, hệ thống tự nhận diện
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p>🏦 <b>Hệ thống Đánh giá Rủi ro Tín dụng</b></p>
    <p>Powered by <b>Machine Learning</b> & <b>Google Gemini AI</b></p>
    <p><i>© 2025 - Phiên bản 2.0</i></p>
</div>
""", unsafe_allow_html=True)
