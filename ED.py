# app_upgraded_i18n.py — Streamlit PD + Phân tích Gemini (Giao diện Tên Tiếng Việt)

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh nhẹ
st.markdown("""
<style>
/* Tăng độ đậm tiêu đề chính */
h1 {
    font-weight: 700;
    color: #1E90FF; /* Màu xanh dương hiện đại */
}
/* Thẻ chính metrics */
div[data-testid="metric-container"] {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
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
# TÍNH X1..X14 TỪ 3 SHEET (CDKT/BCTN/LCTT) - SỬ DỤNG TÊN TIẾNG VIỆT
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
    DTT_prev, DTT_cur    = _get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
    GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
    LNG_prev, LNG_cur    = _get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
    LNTT_prev, LNTT_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_truoc_thue"])
    LV_prev, LV_cur      = _get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])
    TTS_prev, TTS_cur      = _get_row_vals(bs, ALIAS_BS["tong_tai_san"])
    VCSH_prev, VCSH_cur    = _get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
    NPT_prev, NPT_cur      = _get_row_vals(bs, ALIAS_BS["no_phai_tra"])
    TSNH_prev, TSNH_cur    = _get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
    NNH_prev, NNH_cur      = _get_row_vals(bs, ALIAS_BS["no_ngan_han"])
    HTK_prev, HTK_cur      = _get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
    Tien_prev, Tien_cur    = _get_row_vals(bs, ALIAS_BS["tien_tdt"])
    KPT_prev, KPT_cur      = _get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
    NDH_prev, NDH_cur      = _get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])
    KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])

    if pd.notna(GVHB_cur): GVHB_cur = abs(GVHB_cur)
    if pd.notna(LV_cur):   LV_cur    = abs(LV_cur)
    if pd.notna(KH_cur):   KH_cur    = abs(KH_cur)

    def avg(a, b):
        if pd.isna(a) and pd.isna(b): return np.nan
        if pd.isna(a): return b
        if pd.isna(b): return a
        return (a + b) / 2.0
    TTS_avg  = avg(TTS_cur,  TTS_prev)
    VCSH_avg = avg(VCSH_cur, VCSH_prev)
    HTK_avg  = avg(HTK_cur,  HTK_prev)
    KPT_avg  = avg(KPT_cur,  KPT_prev)

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

# Ẩn menu và footer mặc định của Streamlit (Tăng tính chuyên nghiệp)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title("🏛️ HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG DOANH NGHIỆP")
st.write("### Dự báo Xác suất Vỡ nợ (PD) & Phân tích Tài chính nâng cao")

# Hiển thị trạng thái thư viện AI (Sử dụng cột để bố trí đẹp hơn)
col_ai_status, col_date = st.columns([3, 1])
with col_ai_status:
    ai_status = ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai.")
    st.caption(f"🔎 Trạng thái Gemini AI: **{ai_status}**")
with col_date:
    st.caption(f"📅 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.divider()

# Load dữ liệu huấn luyện (CSV có default, X_1..X_14) - Giữ nguyên logic load data
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
    # Tên cột cho việc huấn luyện (phải giữ nguyên X_1..X_14)
    MODEL_COLS = [f"X_{i}" for i in range(1, 15)]
except Exception:
    df = None

uploaded_file = st.sidebar.file_uploader("📂 Tải CSV Dữ liệu Huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    MODEL_COLS = [f"X_{i}" for i in range(1, 15)]

# --- Logic xử lý khi chưa có data huấn luyện ---
if df is None:
    st.sidebar.info("💡 Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14) để xây dựng mô hình.")
    
    # Sử dụng logic mặc định để hiển thị trang dự báo ngay cả khi thiếu data, nhưng cảnh báo.
    menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
    choice = st.sidebar.selectbox('🚀 Danh mục Tính năng', menu, index=2) # <--- ĐIỀU CHỈNH CHÍNH
    
    if choice == 'Sử dụng mô hình để dự báo':
        st.header("⚡ Dự báo PD & Phân tích AI cho Hồ sơ mới")
        st.warning("⚠️ **Không thể dự báo PD**. Vui lòng tải file **CSV Dữ liệu Huấn luyện** ở sidebar để xây dựng mô hình Logistic Regression.")
        up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx**", type=["xlsx"], key="ho_so_dn")
        if up_xlsx is None:
            st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14 và phân tích AI.")
    # Các phần logic còn lại không đổi
    elif choice == 'Mục tiêu của mô hình':
        st.header("🎯 Mục tiêu của Mô hình")
        st.info("Ứng dụng này cần dữ liệu huấn luyện để bắt đầu hoạt động.")
    elif choice == 'Xây dựng mô hình':
         st.header("🛠️ Xây dựng & Đánh giá Mô hình LogReg")
         st.error("❌ **Không thể xây dựng mô hình**. Vui lòng tải file **CSV Dữ liệu Huấn luyện** ở sidebar để bắt đầu.")
        
    st.stop()


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

# Sử dụng Sidebar để chọn tính năng
menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
# ***** ĐIỀU CHỈNH CHỦ YẾU: ĐẶT index=2 LÀM MẶC ĐỊNH *****
choice = st.sidebar.selectbox('🚀 Danh mục Tính năng', menu, index=2)


# --- CÁC PHẦN UI ĐƯỢC TỔ CHỨC LẠI ĐẸP HƠN (GIỮ NGUYÊN) ---

if choice == 'Mục tiêu của mô hình':    
    st.header("🎯 Mục tiêu của Mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số $\text{X1}–\text{X14}$ (tính từ Bảng Cân đối Kế toán, Báo cáo Kết quả Kinh doanh và Báo cáo Lưu chuyển Tiền tệ).")
    
    with st.expander("🖼️ Mô tả trực quan mô hình"):
        st.markdown("Đây là các hình ảnh minh họa cho mô hình Hồi quy Logistic và các giai đoạn đánh giá rủi ro.")
        for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
            try:
                st.image(img)
            except Exception:
                st.warning(f"Không tìm thấy {img}")

elif choice == 'Xây dựng mô hình':
    st.header("🛠️ Xây dựng & Đánh giá Mô hình LogReg")
    st.info("Mô hình Hồi quy Logistic đã được huấn luyện trên **20% dữ liệu Test (chưa thấy)**.")
    
    # Hiển thị Metrics quan trọng bằng st.metric
    st.subheader("1. Tổng quan Kết quả Đánh giá (Test Set)")
    col_acc, col_auc, col_f1 = st.columns(3)
    
    col_acc.metric(label="Độ chính xác (Accuracy)", value=f"{metrics_out['accuracy_out']:.2%}")
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
    col = st.selectbox('🔍 Chọn biến X muốn vẽ', options=MODEL_COLS, index=0)

    # Biểu đồ Scatter Plot và Đường Hồi quy Logisitc (GIỮ NGUYÊN LOGIC)
    if col in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=col, y='default', alpha=0.6, ax=ax, hue='default', palette=['green', 'red'])
            
            # Vẽ đường logistic regression theo 1 biến
            x_range = np.linspace(df[col].min(), df[col].max(), 100).reshape(-1, 1)
            X_temp = df[[col]].copy()
            y_temp = df['default']
            lr_temp = LogisticRegression(max_iter=1000)
            lr_temp.fit(X_temp, y_temp)
            x_test = pd.DataFrame({col: x_range[:, 0]})
            y_curve = lr_temp.predict_proba(x_test)[:, 1]
            ax.plot(x_range, y_curve, color='blue', linewidth=3, label='Đường LogReg')
            
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Default (0)', 'Default (1)'])
        fig2, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        st.pyplot(fig2)
        plt.close(fig2)
        
    with col_metrics_table:
        st.markdown("##### Bảng Metrics Chi tiết")
        dt = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
            "Train Set": [metrics_in['accuracy_in'], metrics_in['precision_in'], metrics_in['recall_in'], metrics_in['f1_in'], metrics_in['auc_in']],
            "Test Set": [metrics_out['accuracy_out'], metrics_out['precision_out'], metrics_out['recall_out'], metrics_out['f1_out'], metrics_out['auc_out']],
        }).set_index("Metric")
        st.dataframe(dt.style.format("{:.4f}"))

elif choice == 'Sử dụng mô hình để dự báo':
    # Trang này được hiển thị mặc định khi index=2
    st.header("⚡ Dự báo PD & Phân tích AI cho Hồ sơ mới")
    
    # Sử dụng st.container và st.expander để tổ chức khu vực upload
    input_container = st.container(border=True)
    with input_container:
        st.markdown("##### 📥 Tải lên Hồ sơ Doanh nghiệp (Excel)")
        st.caption("File phải có đủ **3 sheet**: **CDKT** (Bảng Cân đối Kế toán) ; **BCTN** (Báo cáo Kết quả Kinh doanh) ; **LCTT** (Báo cáo Lưu chuyển Tiền tệ).")
        up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx**", type=["xlsx"], key="ho_so_dn", label_visibility="collapsed")
    
    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet (GIỮ NGUYÊN)
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
            
            # Tách riêng 14 cột tiếng Việt (hiển thị) và 14 cột tiếng Anh (dự báo)
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
                probs = model.predict_proba(ratios_predict[X.columns])[:, 1]
                preds = (probs >= 0.5).astype(int)
                # Thêm PD vào payload AI
                data_for_ai['Xác suất Vỡ nợ (PD)'] = probs[0]
                data_for_ai['Dự đoán PD'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
            except Exception as e:
                # Nếu có lỗi dự báo, chỉ cảnh báo, không dừng app
                st.warning(f"Không dự báo được PD: {e}")
        
        # Hiển thị X1-X14 và PD trong 2 cột
        col_ratios, col_pd = st.columns([3, 1])
        
        with col_ratios:
            # Hiển thị bảng chỉ số tiếng Việt
            st.dataframe(ratios_display.style.format("{:.4f}"), use_container_width=True)
            
        with col_pd:
            pd_value = f"{probs[0]:.2%}" if pd.notna(probs) else "N/A"
            pd_delta = "⬆️ Rủi ro cao" if pd.notna(preds) and preds[0] == 1 else "⬇️ Rủi ro thấp"
            
            st.metric(
                label="**Xác suất Vỡ nợ (PD)**",
                value=pd_value,
                delta=pd_delta if pd.notna(probs) else None,
                delta_color=("inverse" if pd.notna(preds) and preds[0] == 1 else "normal")
            )
            
        st.divider()

        # Khu vực Phân tích AI
        st.markdown("### 2. 🧠 Phân tích AI & Khuyến nghị Tín dụng")
        
        ai_container = st.container(border=True)
        with ai_container:
            st.markdown("Sử dụng Gemini AI để phân tích toàn diện các chỉ số và đưa ra khuyến nghị chuyên nghiệp.")
            
            if st.button("✨ Yêu cầu AI Phân tích & Đề xuất", use_container_width=True, type="primary"):
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                    
                    if "KHÔNG CHO VAY" in ai_result.upper():
                        st.error("🚨 **KHUYẾN NGHỊ CUỐI CÙNG: KHÔNG CHO VAY**")
                    elif "CHO VAY" in ai_result.upper():
                        st.success("✅ **KHUYẾN NGHỊ CUỐI CÙNG: CHO VAY**")
                    else:
                        st.info("💡 **KHUYẾN NGHỊ CUỐI CÙNG**")
                        
                    st.markdown("**Kết quả Phân tích Chi tiết từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("❌ **Lỗi Khóa API**: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
