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

# Thư viện GOOGLE GEMINI VÀ OPENAI (Giữ nguyên logic kiểm tra thư viện)
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
# CẤU HÌNH TRANG (NÂNG CẤP GIAO DIỆN)
# =========================
st.set_page_config(
    page_title="Credit Risk PD & Gemini Analysis",
    page_icon="🏛️",
    layout="wide", # <--- Giữ nguyên layout wide
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh cho MÀU SẮC, PHÔNG CHỮ, HIỆU ỨNG ĐỘNG - ĐÃ ĐỔI SANG TÔNG ĐỎ NHẠT
st.markdown("""
<style>
/* Ẩn menu và footer mặc định */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Thiết lập font chữ và màu nền tổng thể */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* ------------------------------------------------------------------------------------------------
| THAY ĐỔI 1: Dải Banner Cho Tiêu đề Chính (TÔNG ĐỎ NHẠT/HỒNG - CANH GIỮA)
------------------------------------------------------------------------------------------------ */
.banner-title-container {
    /* Màu chuyển sắc nhẹ nhàng - Tông Hồng Nhạt */
    background: linear-gradient(90deg, #ffe0e0, #fff5f5, #ffe0e0); 
    padding: 20px 30px; 
    border-radius: 10px; 
    box-shadow: 0 4px 12px rgba(255, 136, 136, 0.3); /* Shadow nổi bật */
    margin-bottom: 20px; 
    text-align: center; /* **CANH GIỮA TIÊU ĐỀ** */
}
/* Đảm bảo h1 trong banner sử dụng màu sắc đồng bộ và animation */
.banner-title-container h1 {
    color: #d9534f !important; /* Đỏ Mềm */
    font-weight: 900 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    animation: wave 2s infinite alternate; /* THÊM CHUYỂN ĐỘNG NHẸ */
}

@keyframes wave {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}

.banner-title-container h3 {
    color: #e38282 !important; /* Hồng tươi hơn */
    font-weight: 600 !important;
    margin-top: -10px;
    border-bottom: none;
}
/* ------------------------------------------------------------------------------------------------ */

/* Tiêu đề cấp 2 (Sub-header) */
h3 {
    color: #d9534f; /* Đỏ Mềm */
    border-bottom: 2px solid #ffeded;
    padding-bottom: 5px;
}

/* Thẻ chính metrics - Thiết kế box hiện đại */
div[data-testid="metric-container"] {
    border: 1px solid #ff8888; /* Border nổi bật - Hồng */
    border-radius: 12px;
    padding: 10px;
    box-shadow: 4px 4px 10px rgba(217, 83, 79, 0.15);
    background-color: #fff7f7; /* Nền trắng hồng nhạt */
}
/* Màu chữ metric */
div[data-testid="stMetricValue"] {
    color: #d9534f; /* Đỏ Mềm */
    font-size: 1.8rem;
}
/* Màu chữ delta (rủi ro cao - đỏ, rủi ro thấp - xanh) */
div[data-testid="stMetricDelta"] {
    font-weight: bold;
}
/* Đảm bảo màu đỏ cho rủi ro cao (inverse) */
div[data-testid="stMetricDelta"][data-delta-color="inverse"] {
    color: #d9534f !important; /* Đỏ mềm */
}
/* Đảm bảo màu xanh cho rủi ro thấp (normal) - Giữ xanh nhưng tông mềm */
div[data-testid="stMetricDelta"][data-delta-color="normal"] {
    color: #4CAF50 !important; /* Xanh Lá An toàn */
}

/* ------------------------------------------------------------------------------------------------
| THAY ĐỔI 2: Styling cho Tabs Ngang (Tông Đỏ/Hồng)
------------------------------------------------------------------------------------------------ */
/* Style cho từng nút Tab (chưa được chọn) */
button[data-testid="stTab"] {
    background-color: #fff7f7; /* Nền nhẹ */
    border: 1px solid #ffcccc;
    border-radius: 8px 8px 0 0 !important; /* Bo góc trên */
    transition: all 0.3s ease;
    font-weight: 600;
    color: #8c2a2a; /* Màu chữ nâu đỏ */
    padding: 10px 20px;
    margin-right: 5px;
}

/* Hiệu ứng Hover */
button[data-testid="stTab"]:hover {
    background-color: #ffeded; /* Hồng nhạt hơn khi hover */
    color: #d9534f; /* Đỏ đậm hơn */
    border-color: #ff8888;
    transform: translateY(-2px); /* Hiệu ứng nhấc lên nhẹ */
}

/* Style cho Tab đang được chọn (Active) */
button[data-testid="stTab"][aria-selected="true"] {
    background-color: #ff8888 !important; /* Màu Hồng nổi bật */
    color: white !important; /* Chữ trắng */
    border-color: #d9534f !important;
    border-bottom: 2px solid white !important;
    box-shadow: 0 4px 8px rgba(217, 83, 79, 0.4); /* Thêm shadow nhẹ */
    transform: translateY(0px);
}

/* Kéo thanh Tab lên cao, nằm dưới banner */
[data-testid="stVerticalBlock"]:has(.stTabs) {
    margin-top: -60px; 
    z-index: 1000; 
}
/* ------------------------------------------------------------------------------------------------ */

/* Sidebar - Làm nổi bật phần upload file */
[data-testid="stSidebar"] {
    background-color: #ffeded; /* Hồng nhạt cho sidebar */
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #d9534f; /* Đỏ mềm */
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
}

/* Nút bấm Phân tích AI - Hiệu ứng nhấn */
button[kind="primary"] {
    background-color: #ff8888;
    border-color: #ff8888;
    transition: background-color 0.3s ease, transform 0.1s ease;
}
button[kind="primary"]:hover {
    background-color: #d9534f;
    border-color: #d9534f;
}
button[kind="primary"]:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)


# =========================
# HÀM GỌI GEMINI API (GIỮ NGUYÊN LOGIC)
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
        "Phân tích toàn diện dựa trên 14 chỉ số tài chính được cung cấp và PD nếu có. "
        "Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4) Hiệu quả hoạt động. "
        "Kết thúc bằng khuyến nghị in hoa: CHO VAY hoặc KHÔNG CHO VAY, kèm 2–3 điều kiện nếu CHO VAY. "
        "Viết bằng tiếng Việt súc tích, chuyên nghiệp."
    )
    
    # Gửi tên tiếng Việt dễ hiểu hơn cho AI
    user_prompt = "Bộ chỉ số tài chính và PD cần phân tích:\n" + str(data_payload) + "\n\nHãy phân tích và đưa ra khuyến nghị."

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
# TÍNH X1..X14 TỪ 3 SHEET (CDKT/BCTN/LCTT) - SỬ DỤNG TÊN TIẾNG VIỆT (GIỮ NGUYÊN)
# =========================

# Bảng ánh xạ Tên chỉ số tiếng Việt
COMPUTED_COLS = [
    "Biên Lợi nhuận Gộp (X1)", "Biên Lợi nhuận Tr.Thuế (X2)", "ROA Tr.Thuế (X3)", 
    "ROE Tr.Thuế (X4)", "Tỷ lệ Nợ/TTS (X5)", "Tỷ lệ Nợ/VCSH (X6)", 
    "Thanh toán Hiện hành (X7)", "Thanh toán Nhanh (X8)", "Khả năng Trả lãi (X9)", 
    "Khả năng Trả nợ Gốc (X10)", "Tỷ lệ Tiền/VCSH (X11)", "Vòng quay HTK (X12)", 
    "Kỳ thu tiền BQ (X13)", "Hiệu suất Tài sản (X14)"
]

# Alias các dòng quan trọng trong từng sheet (GIỮ NGUYÊN)
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
    """Tìm dòng theo alias. Trả về (prev, cur) theo 2 cột năm gần nhất."""
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
            # Xóa dấu phẩy, khoảng trắng
            return float(str(x).replace(",", "").replace(" ", ""))
        except Exception:
            return np.nan

    return to_num(row[prev_col]), to_num(row[cur_col])

def compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
    """Đọc 3 sheet CDKT/BCTN/LCTT và tính X1..X14 theo yêu cầu."""
    bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
    is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
    cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")

    # ---- Tính toán các biến số tài chính (GIỮ NGUYÊN CÁCH TÍNH)
    DTT_prev, DTT_cur         = _get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
    GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
    LNG_prev, LNG_cur         = _get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
    LNTT_prev, LNTT_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_truoc_thue"])
    LV_prev, LV_cur           = _get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])
    TTS_prev, TTS_cur         = _get_row_vals(bs, ALIAS_BS["tong_tai_san"])
    VCSH_prev, VCSH_cur       = _get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
    NPT_prev, NPT_cur         = _get_row_vals(bs, ALIAS_BS["no_phai_tra"])
    TSNH_prev, TSNH_cur       = _get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
    NNH_prev, NNH_cur         = _get_row_vals(bs, ALIAS_BS["no_ngan_han"])
    HTK_prev, HTK_cur         = _get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
    Tien_prev, Tien_cur       = _get_row_vals(bs, ALIAS_BS["tien_tdt"])
    KPT_prev, KPT_cur         = _get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
    NDH_prev, NDH_cur         = _get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])
    KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])

    if pd.notna(GVHB_cur): GVHB_cur = abs(GVHB_cur)
    if pd.notna(LV_cur):      LV_cur     = abs(LV_cur)
    if pd.notna(KH_cur):      KH_cur     = abs(KH_cur)

    def avg(a, b):
        if pd.isna(a) and pd.isna(b): return np.nan
        if pd.isna(a): return b
        if pd.isna(b): return a
        return (a + b) / 2.0
    TTS_avg    = avg(TTS_cur,    TTS_prev)
    VCSH_avg = avg(VCSH_cur, VCSH_prev)
    HTK_avg    = avg(HTK_cur,    HTK_prev)
    KPT_avg    = avg(KPT_cur,    KPT_prev)

    EBIT_cur = (LNTT_cur + LV_cur) if (pd.notna(LNTT_cur) and pd.notna(LV_cur)) else np.nan
    NDH_cur = 0.0 if pd.isna(NDH_cur) else NDH_cur

    def div(a, b):
        return np.nan if (b is None or pd.isna(b) or b == 0) else a / b

    # ==== TÍNH X1..X14 ==== (GIỮ NGUYÊN CÔNG THỨC)
    X1  = div(LNG_cur, DTT_cur)
    X2  = div(LNTT_cur, DTT_cur)
    X3  = div(LNTT_cur, TTS_avg)
    X4  = div(LNTT_cur, VCSH_avg)
    X5  = div(NPT_cur,  TTS_cur)
    X6  = div(NPT_cur,  VCSH_cur)
    X7  = div(TSNH_cur, NNH_cur)
    X8  = div((TSNH_cur - HTK_cur) if pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)
    X9  = div(EBIT_cur, LV_cur)
    X10 = div((EBIT_cur + (KH_cur if pd.notna(KH_cur) else 0.0)), (LV_cur + NDH_cur) if pd.notna(LV_cur) else np.nan)
    X11 = div(Tien_cur, VCSH_cur)
    X12 = div(GVHB_cur, HTK_avg)
    turnover = div(DTT_cur, KPT_avg)
    X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else np.nan
    X14 = div(DTT_cur, TTS_avg)

    # Khởi tạo DataFrame với tên cột tiếng Việt mới
    ratios = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14]],
                          columns=COMPUTED_COLS)
                          
    # Thêm cột X_1..X_14 ẩn để phục vụ việc dự báo mô hình
    ratios[[f"X_{i}" for i in range(1, 15)]] = ratios.values
    return ratios

# =========================
# UI & TRAIN MODEL
# =========================
np.random.seed(0)

# ------------------------------------------------------------------------------------------------
# THAY ĐỔI MỚI: Tách Tab ra ngoài và thay đổi CSS để kéo Tab lên trên
# ------------------------------------------------------------------------------------------------

# 1. Hiển thị dải Banner/Tiêu đề
st.markdown('<div class="banner-title-container">', unsafe_allow_html=True)
st.title("🏛️ HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG DOANH NGHIỆP")
st.write("### Dự báo Xác suất Vỡ nợ (PD) & Phân tích Tài chính nâng cao")
st.markdown('</div>', unsafe_allow_html=True)

# 2. Định nghĩa các Tabs (sẽ được kéo lên)
tab_predict, tab_build, tab_goal = st.tabs([
    "🚀 Sử dụng mô hình để dự báo", 
    "🛠️ Xây dựng mô hình", 
    "🎯 Mục tiêu của mô hình"
])

# Load dữ liệu huấn luyện (CSV có default, X_1..X_14) - Giữ nguyên logic load data
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
    MODEL_COLS = [f"X_{i}" for i in range(1, 15)]
except Exception:
    df = None

# DI CHUYỂN UPLOADER VỀ ĐẦU SIDEBAR (Không còn selectbox)
uploaded_file = st.sidebar.file_uploader("📂 Tải CSV Dữ liệu Huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    MODEL_COLS = [f"X_{i}" for i in range(1, 15)]
    
# --- Logic xử lý khi chưa có data huấn luyện ---
if df is None:
    st.sidebar.info("💡 Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14) để xây dựng mô hình.")
    
    # Logic cho các tab khi thiếu data huấn luyện
    with tab_predict:
        st.header("⚡ Dự báo PD & Phân tích AI cho Hồ sơ mới")
        st.warning("⚠️ **Không thể dự báo PD**. Vui lòng tải file **CSV Dữ liệu Huấn luyện** ở sidebar để xây dựng mô hình Logistic Regression.")
        up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx**", type=["xlsx"], key="ho_so_dn")
        if up_xlsx is None:
            st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14 và phân tích AI.")

    with tab_goal:
        st.header("🎯 Mục tiêu của Mô hình")
        st.info("Ứng dụng này cần dữ liệu huấn luyện để bắt đầu hoạt động.")
    
    with tab_build:
          st.header("🛠️ Xây dựng & Đánh giá Mô hình LogReg")
          st.error("❌ **Không thể xây dựng mô hình**. Vui lòng tải file **CSV Dữ liệu Huấn luyện** ở sidebar để bắt đầu.")
          
    st.stop()
# ------------------------------------------------------------------------------------------------

# Hiển thị trạng thái thư viện AI (Sử dụng cột để bố trí đẹp hơn)
col_ai_status, col_date = st.columns([3, 1])
with col_ai_status:
    ai_status = ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai.")
    st.caption(f"🔎 Trạng thái Gemini AI: **<span style='color: #d9534f; font-weight: bold;'>{ai_status}</span>**", unsafe_allow_html=True)
with col_date:
    st.caption(f"📅 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.divider()

# Kiểm tra cột cần thiết
required_cols = ['default'] + MODEL_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Thiếu cột: **{missing}**. Vui lòng kiểm tra lại file CSV huấn luyện.")
    st.stop()


# Train model (GIỮ NGUYÊN)
X = df[MODEL_COLS] # Chỉ lấy các cột X_1..X_14
y = df['default'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

# Dự báo & đánh giá (GIỮ NGUYÊN)
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

# --- CÁC PHẦN UI DỰA TRÊN TABS ---

with tab_goal:
    st.header("🎯 Mục tiêu của Mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số $\\text{X1}–\\text{X14}$ (tính từ Bảng Cân đối Kế toán, Báo cáo Kết quả Kinh doanh và Báo cáo Lưu chuyển Tiền tệ).")
    
    with st.expander("🖼️ Mô tả trực quan mô hình"):
        st.markdown("Đây là các hình ảnh minh họa cho mô hình Hồi quy Logistic và các giai đoạn đánh giá rủi ro.")
        
        # ------------------------------------------------------------------
        # FIX LỖI HIỂN THỊ ẢNH (Sử dụng URL Placeholder cố định)
        # ------------------------------------------------------------------
        image_files = ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]
        
        for img in image_files:
            placeholder_text = img.replace('.jpg', '').replace('.png', '').upper()
            # FIX: Loại bỏ try/except không cần thiết và hiển thị ảnh URL Placeholder trực tiếp
            st.image(
                f"https://placehold.co/800x400/d9534f/ffffff?text={placeholder_text}_PLACEHOLDER",
                caption=f"Ảnh minh họa mô hình: {placeholder_text}",
                use_column_width=True
            )
        # ------------------------------------------------------------------

with tab_build:
    st.header("🛠️ Xây dựng & Đánh giá Mô hình LogReg")
    st.info("Mô hình Hồi quy Logistic đã được huấn luyện trên **20% dữ liệu Test (chưa thấy)**.")
    
    # Hiển thị Metrics quan trọng bằng st.metric
    st.subheader("1. Tổng quan Kết quả Đánh giá (Test Set)")
    col_acc, col_auc, col_f1 = st.columns(3)
    
    col_acc.metric(label="Độ chính xác (Accuracy)", value=f"{metrics_out['accuracy_out']:.2%}")
    # Đảm bảo logic delta vẫn đúng
    col_auc.metric(label="Diện tích dưới đường cong (AUC)", value=f"{metrics_out['auc_out']:.3f}", delta=f"{metrics_in['auc_in'] - metrics_out['auc_out']:.3f}", delta_color="inverse")
    col_f1.metric(label="Điểm F1-Score", value=f"{metrics_out['f1_out']:.3f}")
    
    st.divider()

    # Thống kê chi tiết & Biểu đồ
    st.subheader("2. Dữ liệu và Trực quan hóa")
    
    with st.expander("📊 Thống kê Mô tả và Dữ liệu Mẫu"):
        st.markdown("##### Thống kê Mô tả các biến $X_1..X_{14}$")
        st.dataframe(df[MODEL_COLS].describe().style.format("{:.4f}"))
        st.markdown("##### 6 Dòng dữ liệu huấn luyện mẫu (Đầu/Cuối)")
        st.dataframe(pd.concat([df.head(3), df.tail(3)]))

    st.markdown("##### Biểu đồ Phân tán (Scatter Plot) với Đường Hồi quy Logisitc")
    col = st.selectbox('🔍 Chọn biến X muốn vẽ', options=MODEL_COLS, index=0, key="select_build_col")
    
    # Biểu đồ Scatter Plot và Đường Hồi quy Logisitc (GIỮ NGUYÊN LOGIC)
    if col in df.columns:
        try:
            # Dùng Streamlit.pyplot để đảm bảo tích hợp tốt hơn
            fig, ax = plt.subplots(figsize=(10, 6))
            # Màu mới cho scatter plot
            sns.scatterplot(data=df, x=col, y='default', alpha=0.6, ax=ax, hue='default', palette=['#ff8888', '#d9534f']) 
            
            # Vẽ đường logistic regression theo 1 biến
            x_range = np.linspace(df[col].min(), df[col].max(), 100).reshape(-1, 1)
            X_temp = df[[col]].copy()
            y_temp = df['default']
            lr_temp = LogisticRegression(max_iter=1000)
            lr_temp.fit(X_temp, y_temp)
            x_test = pd.DataFrame({col: x_range[:, 0]})
            y_curve = lr_temp.predict_proba(x_test)[:, 1]
            ax.plot(x_range, y_curve, color='#d9534f', linewidth=3, label='Đường LogReg') # Màu đỏ mềm
            
            ax.set_title(f'Quan hệ giữa {col} và Xác suất Vỡ nợ', fontsize=14)
            ax.set_ylabel('Xác suất default (1: Default)', fontsize=12)
            ax.set_xlabel(col, fontsize=12)
            ax.legend(title='Default')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")
    else:
        st.warning("Biến không tồn tại trong dữ liệu.")
    
    st.divider()

    st.subheader("3. Ma trận Nhầm lẫn và Bảng Metrics Chi tiết")
    col_cm, col_metrics_table = st.columns(2)
    
    with col_cm:
        st.markdown("##### Ma trận Nhầm lẫn (Test Set)")
        cm = confusion_matrix(y_test, y_pred_out)
        # Sử dụng cmap màu đỏ/cam
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Default (0)', 'Default (1)'])
        fig2, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.OrRd) # Sử dụng OrRd (Orange Red)
        st.pyplot(fig2)
        plt.close(fig2)
        
    with col_metrics_table:
        st.markdown("##### Bảng Metrics Chi tiết")
        dt = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
            "Train Set": [metrics_in['accuracy_in'], metrics_in['precision_in'], metrics_in['recall_in'], metrics_in['f1_in'], metrics_in['auc_in']],
            "Test Set": [metrics_out['accuracy_out'], metrics_out['precision_out'], metrics_out['recall_out'], metrics_out['f1_out'], metrics_out['auc_out']],
        }).set_index("Metric")
        # Thêm styling để làm nổi bật kết quả tốt nhất
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #ffeded' if v else '' for v in is_max]

        st.dataframe(dt.style.format("{:.4f}").apply(highlight_max, axis=1), use_container_width=True)

with tab_predict:
    # Trang này được hiển thị mặc định
    st.header("⚡ Dự báo PD & Phân tích AI cho Hồ sơ mới")
    
    # Sử dụng st.container và st.expander để tổ chức khu vực upload
    input_container = st.container(border=True)
    with input_container:
        st.markdown("##### 📥 Tải lên Hồ sơ Doanh nghiệp (Excel)")
        st.caption("File phải có đủ **3 sheet**: **CDKT** (Bảng Cân đối Kế toán) ; **BCTN** (Báo cáo Kết quả Kinh doanh) ; **LCTT** (Báo cáo Lưu chuyển Tiền tệ).")
        up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx**", type=["xlsx"], key="ho_so_dn_main", label_visibility="collapsed")
    
    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet (GIỮ NGUYÊN)
        try:
            # Hiển thị thanh tiến trình giả lập (thêm hiệu ứng động)
            with st.spinner('Đang đọc và xử lý dữ liệu tài chính...'):
                ratios_df = compute_ratios_from_three_sheets(up_xlsx)
            
            # Tách riêng 14 cột tiếng Việt (hiển thị) và 14 cột tiếng Anh (dự báo)
            # ratios_display là DataFrame 1 cột: Index (Tên chỉ số) | Giá trị
            ratios_display = ratios_df[COMPUTED_COLS].T.rename(columns={0: 'Giá trị'})
            ratios_predict = ratios_df[MODEL_COLS]
            
        except Exception as e:
            st.error(f"❌ Lỗi tính chỉ số tài chính: Vui lòng kiểm tra lại cấu trúc 3 sheet trong file Excel. Chi tiết lỗi: {e}")
            st.stop()

        st.divider()
        st.markdown("### 1. 🔢 Các Chỉ số Tài chính Đã tính")
        
        # Tạo payload data cho AI (Sử dụng tên tiếng Việt)
        data_for_ai = ratios_display.to_dict()['Giá trị']
        
        # (Tuỳ chọn) dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        probs = np.nan
        preds = np.nan
        # Kiểm tra mô hình có sẵn sàng dự báo không (đã train và cột khớp)
        if set(X.columns) == set(ratios_predict.columns):
            try:
                # Đảm bảo thứ tự cột cho predict đúng như thứ tự cột huấn luyện
                probs = model.predict_proba(ratios_predict[X.columns])[:, 1]
                preds = (probs >= 0.5).astype(int)
                # Thêm PD vào payload AI
                data_for_ai['Xác suất Vỡ nợ (PD)'] = probs[0]
                data_for_ai['Dự đoán PD'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
            except Exception as e:
                # Nếu có lỗi dự báo, chỉ cảnh báo, không dừng app
                st.warning(f"Không dự báo được PD: {e}")
        
        # ------------------------------------------------------------------------------------------------
        # ĐIỀU CHỈNH CỦA CHUYÊN GIA PYTHON: Bỏ .T để hiển thị đúng Tên Biến | Con số
        # ------------------------------------------------------------------------------------------------
        pd_col_1, pd_col_2, pd_col_pd = st.columns([2, 2, 1]) # Chia làm 3 cột, 2 cột giữa hiển thị ratios, 1 cột cuối hiển thị PD
        
        ratios_list = ratios_display.index.tolist()
        mid_point = len(ratios_list) // 2
        # ratios_display đã có cấu trúc đúng: Index (Tên biến) | Giá trị (Con số)
        ratios_part1 = ratios_display.iloc[:mid_point]
        ratios_part2 = ratios_display.iloc[mid_point:]
        
        # Hàm styling (Cập nhật màu sắc)
        def color_ratios(val):
            """Ánh xạ màu dựa trên tên chỉ số và giá trị (tạm thời để hiển thị đẹp)"""
            # Chỉ số Thanh khoản (X7, X8) - Green/Yellow
            if "Thanh toán" in val.name and val.values[0] < 1.0: return ['background-color: #ffcccc' for _ in val] # Dưới 1: Báo động đỏ
            if "Thanh toán" in val.name and val.values[0] > 1.5: return ['background-color: #ccffcc' for _ in val] # Trên 1.5: Tốt
            # Chỉ số Nợ (X5, X6) - Red/Green
            if "Tỷ lệ Nợ/" in val.name and val.values[0] > 1.0: return ['background-color: #ffcccc' for _ in val] # Trên 1: Rủi ro cao
            if "Tỷ lệ Nợ/" in val.name and val.values[0] < 0.5: return ['background-color: #ccffcc' for _ in val] # Dưới 0.5: Tốt
            # Chỉ số Sinh lời (X1, X2, X3, X4) - Red/Green
            if "Lợi nhuận" in val.name or "ROA" in val.name or "ROE" in val.name:
                if val.values[0] <= 0: return ['background-color: #ffcccc' for _ in val]
                if val.values[0] > 0.1: return ['background-color: #ccffcc' for _ in val]
            return [''] * len(val)

        with pd_col_1:
             # Đảm bảo hiển thị Tên biến | Giá trị
             st.markdown("##### **Chỉ số Tài chính (1/2)**") 
             st.dataframe(
                 ratios_part1.style.apply(color_ratios, axis=1).format("{:.4f}").set_properties(**{'font-size': '14px'}),
                 use_container_width=True
             )

        with pd_col_2:
            # Đảm bảo hiển thị Tên biến | Giá trị
            st.markdown("##### **Chỉ số Tài chính (2/2)**")
            st.dataframe(
                ratios_part2.style.apply(color_ratios, axis=1).format("{:.4f}").set_properties(**{'font-size': '14px'}),
                use_container_width=True
            )
        
        with pd_col_pd:
            pd_value = f"{probs[0]:.2%}" if pd.notna(probs) else "N/A"
            pd_delta = "⬆️ Rủi ro cao" if pd.notna(preds) and preds[0] == 1 else "⬇️ Rủi ro thấp"
            
            st.metric(
                label="**Xác suất Vỡ nợ (PD)**",
                value=pd_value,
                delta=pd_delta if pd.notna(probs) else None,
                # Đảo ngược màu sắc delta cho PD: Rủi ro cao là màu đỏ (inverse), rủi ro thấp là màu xanh (normal)
                delta_color=("inverse" if pd.notna(preds) and preds[0] == 1 else "normal")
            )
        # ------------------------------------------------------------------------------------------------

        st.divider()

        # Khu vực Phân tích AI
        st.markdown("### 2. 🧠 Phân tích AI & Khuyến nghị Tín dụng")
        
        # ------------------------------------------------------------------------------------------------
        # BỔ SUNG: Biểu đồ trực quan hóa 14 chỉ số đã tính (Bar Chart)
        # ------------------------------------------------------------------------------------------------
        st.markdown("#### 📈 Trực quan hóa các Chỉ số Tài chính")
        # Chuẩn bị data cho biểu đồ
        chart_data = ratios_display.copy()
        chart_data.columns = ['Giá trị']
        chart_data['Chỉ số'] = chart_data.index.str.replace(r' \(X\d+\)', '', regex=True)
        chart_data.reset_index(drop=True, inplace=True)

        # Phân loại màu sắc theo nhóm chỉ số
        def get_group(index_name):
            if any(k in index_name for k in ["Lợi nhuận", "ROA", "ROE"]): return "A. Sinh lời"
            if any(k in index_name for k in ["Nợ/", "Nợ/"]): return "B. Cơ cấu vốn"
            if any(k in index_name for k in ["Thanh toán", "Tiền/VCSH"]): return "C. Thanh khoản"
            return "D. Hiệu quả hoạt động"

        chart_data['Nhóm'] = chart_data['Chỉ số'].apply(get_group)
        # Màu mới: Đỏ mềm cho giá trị âm, Xanh an toàn cho giá trị dương
        chart_data['Màu'] = np.where(chart_data['Giá trị'] >= 0, '#4CAF50', '#d9534f') 

        # Matplotlib/Seaborn Bar Plot
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Sắp xếp lại dữ liệu theo Nhóm để biểu đồ dễ đọc hơn
            chart_data.sort_values(by='Nhóm', ascending=False, inplace=True) 

            # Vẽ biểu đồ thanh ngang
            sns.barplot(
                x='Giá trị', 
                y='Chỉ số', 
                data=chart_data, 
                palette=chart_data['Màu'].tolist(), # Dùng màu tùy chỉnh (Xanh/Đỏ)
                ax=ax,
                # Thêm màu theo Nhóm để phân biệt
                hue='Nhóm',
                hue_order=sorted(chart_data['Nhóm'].unique()),
                dodge=False
            )
            
            ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7) # Đường 0 rõ ràng
            ax.set_title('Trực quan hóa 14 Chỉ số Tài chính Quan trọng', fontsize=16, color='#d9534f', fontweight='bold')
            ax.set_xlabel('Giá trị Chỉ số', fontsize=12)
            ax.set_ylabel('Tên Chỉ số', fontsize=12)
            
            # Tạo legend mới cho Nhóm (Cập nhật màu legend)
            from matplotlib.patches import Patch
            group_map = {
                'A. Sinh lời': '#4CAF50', 
                'B. Cơ cấu vốn': '#d9534f', 
                'C. Thanh khoản': '#FFC107', 
                'D. Hiệu quả hoạt động': '#90CAF9'
            }
            legend_elements = [Patch(facecolor=group_map.get(g, '#999999'), label=g) 
                               for g in sorted(chart_data['Nhóm'].unique())]
            
            ax.legend(handles=legend_elements, title='Nhóm Chỉ số', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Không thể vẽ biểu đồ chỉ số: {e}")
        
        st.divider()
        # ------------------------------------------------------------------------------------------------


        ai_container = st.container(border=True)
        with ai_container:
            st.markdown("Sử dụng Gemini AI để phân tích toàn diện các chỉ số và đưa ra khuyến nghị chuyên nghiệp.")
            
            if st.button("✨ Yêu cầu AI Phân tích & Đề xuất", use_container_width=True, type="primary"):
                # Kiểm tra API Key: ưu tiên lấy từ secrets
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    # Thêm thanh tiến trình đẹp mắt
                    progress_bar = st.progress(0, text="Đang gửi dữ liệu và chờ Gemini phân tích...")
                    for percent_complete in range(100):
                        import time
                        time.sleep(0.01) # Giả lập thời gian xử lý
                        progress_bar.progress(percent_complete + 1, text=f"Đang gửi dữ liệu và chờ Gemini phân tích... {percent_complete+1}%")
                    
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    progress_bar.empty() # Xóa thanh tiến trình
                    
                    st.markdown("**Kết quả Phân tích Chi tiết từ Gemini AI:**")
                    
                    if "KHÔNG CHO VAY" in ai_result.upper():
                        st.error("🚨 **KHUYẾN NGHỊ CUỐI CÙNG: KHÔNG CHO VAY**")
                        st.snow() 
                    elif "CHO VAY" in ai_result.upper():
                        st.success("✅ **KHUYẾN NGHỊ CUỐI CÙNG: CHO VAY**")
                        st.balloons() 
                    else:
                        st.info("💡 **KHUYẾN NGHỊ CUỐI CÙNG**")
                        
                    st.info(ai_result)
                else:
                    st.error("❌ **Lỗi Khóa API**: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
