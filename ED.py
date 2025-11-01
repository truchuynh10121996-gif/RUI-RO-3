# ED.py — Streamlit PD + Phân tích Gemini (Phiên bản Chuyên nghiệp)

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

# =========================
# THÊM THƯ VI VIỆN GOOGLE GEMINI
# =========================
try:
    from google import genai
    from google.genai.errors import APIError
    _GEMINI_OK = True
except Exception:
    genai = None
    APIError = Exception
    _GEMINI_OK = False

# Giữ lại logic OpenAI (nếu có) nhưng không dùng
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False


MODEL_NAME = "gemini-2.5-flash" 

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
        if len(numeric_years) >= 2:
            return numeric_years[-2][1], numeric_years[-1][1]
        elif len(numeric_years) == 1:
            # Nếu chỉ có 1 năm, dùng cột cuối cùng làm cột hiện tại, cột trước là cột cuối cùng thứ 2
            cols = df.columns[-2:]
            return cols[0], numeric_years[0][1] # Giả định cột trước năm đó là cột cuối cùng thứ 2
    # fallback: 2 cột cuối
    cols = df.columns[-2:]
    return cols[0], cols[1]

def _get_row_vals(df: pd.DataFrame, aliases: list[str]):
    """Tìm dòng theo alias (contains, không phân biệt hoa/thường). Trả về (prev, cur) theo 2 cột năm gần nhất."""
    if df.empty:
        return np.nan, np.nan
        
    label_col = df.columns[0]
    
    # Đảm bảo có ít nhất 2 cột ngoài cột label
    if len(df.columns) < 3:
        return np.nan, np.nan
        
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

    return to_num(row.get(prev_col, np.nan)), to_num(row.get(cur_col, np.nan))

def compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
    """Đọc 3 sheet CDKT/BCTN/LCTT và tính X1..X14 theo yêu cầu."""
    # Đọc 3 sheet; cần openpyxl trong requirements
    try:
        bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
        is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
        cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")
    except ValueError as e:
        # Bắt lỗi nếu thiếu sheet
        raise ValueError(f"Lỗi: File Excel thiếu một trong ba sheet bắt buộc (CDKT, BCTN, LCTT). Chi tiết: {e}")
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file Excel: {e}")

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
    if pd.notna(LV_cur):   LV_cur   = abs(LV_cur)
    if pd.notna(KH_cur):   KH_cur   = abs(KH_cur)

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
    X1  = div(LNG_cur, DTT_cur)                         # Biên LN gộp
    X2  = div(LNTT_cur, DTT_cur)                        # Biên LNTT
    X3  = div(LNTT_cur, TTS_avg)                        # ROA (trước thuế)
    X4  = div(LNTT_cur, VCSH_avg)                       # ROE (trước thuế)
    X5  = div(NPT_cur,  TTS_cur)                        # Nợ/Tài sản
    X6  = div(NPT_cur,  VCSH_cur)                       # Nợ/VCSH
    X7  = div(TSNH_cur, NNH_cur)                        # Thanh toán hiện hành
    X8  = div((TSNH_cur - HTK_cur) if pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)  # Thanh toán nhanh
    X9  = div(EBIT_cur, LV_cur)                         # Khả năng trả lãi
    X10 = div((EBIT_cur + (KH_cur if pd.notna(KH_cur) else 0.0)),
                 (LV_cur + NDH_cur) if pd.notna(LV_cur) else np.nan)  # Khả năng trả nợ gốc
    X11 = div(Tien_cur, VCSH_cur)                       # Tiền/VCSH
    X12 = div(GVHB_cur, HTK_avg)                        # Vòng quay HTK
    turnover = div(DTT_cur, KPT_avg)                    # Vòng quay phải thu
    X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else np.nan  # Kỳ thu tiền BQ
    X14 = div(DTT_cur, TTS_avg)                         # Hiệu suất sử dụng tài sản

    ratios = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14]],
                         columns=[f"X_{i}" for i in range(1, 15)])
    return ratios

# =========================
# UI & TRAIN MODEL (PHẦN NÂNG CẤP GIAO DIỆN)
# =========================

# 1. Cấu hình Trang và CSS Tùy chỉnh
st.set_page_config(
    page_title="Hệ thống Phân tích & Dự báo PD Doanh nghiệp",
    page_icon="🏦",
    layout="wide", # Sử dụng toàn bộ chiều rộng màn hình
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh để tối ưu hóa Tabs và Metrics
st.markdown("""
<style>
/* Đảm bảo tab trông hiện đại hơn */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
.stTabs [aria-selected="true"] {
    border-bottom: 4px solid #007bff; /* Màu xanh chuyên nghiệp */
    color: #007bff;
}
/* Thiết kế Metric rõ ràng, nhấn mạnh số liệu */
.stMetric > div:nth-child(2) > div:nth-child(1) {
    font-size: 2.5rem; 
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

np.random.seed(0)

st.title("🏦 PHÂN TÍCH VÀ DỰ BÁO PD DOANH NGHIỆP")
st.markdown("""
<div style="padding: 10px 0 20px 0;">
    <span style="font-size: 1.1em; color: #555;">Công cụ dự báo Xác suất Vỡ nợ (PD) dựa trên chỉ số tài chính và phân tích chuyên sâu bởi Gemini AI.</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# 2. Xử lý Dữ liệu ở Sidebar và Giai đoạn Huấn luyện

# Đưa phần tải dữ liệu huấn luyện vào Sidebar
st.sidebar.header("⚙️ Cấu hình Dữ liệu Huấn luyện")
uploaded_file = st.sidebar.file_uploader(
    "1. Tải CSV Dữ liệu Huấn luyện", 
    type=['csv'], 
    help="File CSV phải có cột 'default' (mục tiêu) và X_1...X_14"
)
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
    elif os.path.exists('DATASET.csv'): # Giữ lại cơ chế tải file default nếu có
        df = pd.read_csv('DATASET.csv', encoding='latin-1')
    else:
        df = None
except Exception:
    df = None

# Hiển thị trạng thái AI trong Sidebar
st.sidebar.markdown("---")
st.sidebar.caption("🔎 Trạng thái AI: " + ("✅ Gemini sẵn sàng" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai."))
st.sidebar.info("Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets để sử dụng chức năng AI.")

if df is None:
    st.info("⚠️ Mô hình PD chưa được huấn luyện. Vui lòng tải file CSV huấn luyện để bắt đầu.")
    st.stop()

# Kiểm tra cột cần thiết
required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Dữ liệu huấn luyện bị thiếu cột: {missing}")
    st.stop()

# Huấn luyện mô hình (Logic giữ nguyên)
X = df.drop(columns=['default'])
y = df['default'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

# Tính Metrics (Logic giữ nguyên)
y_pred_in = model.predict(X_train)
y_proba_in = model.predict_proba(X_train)[:, 1]
y_pred_out = model.predict(X_test)
y_proba_out = model.predict_proba(X_test)[:, 1]

metrics_in = {
   "accuracy_in": accuracy_score(y_train, y_pred_in), "precision_in": precision_score(y_train, y_pred_in, zero_division=0),
   "recall_in": recall_score(y_train, y_pred_in, zero_division=0), "f1_in": f1_score(y_train, y_pred_in, zero_division=0),
   "auc_in": roc_auc_score(y_train, y_proba_in),
}
metrics_out = {
   "accuracy_out": accuracy_score(y_test, y_pred_out), "precision_out": precision_score(y_test, y_pred_out, zero_division=0),
   "recall_out": recall_score(y_test, y_pred_out, zero_division=0), "f1_out": f1_score(y_test, y_pred_out, zero_division=0),
   "auc_out": roc_auc_score(y_test, y_proba_out),
}


# 3. Sử dụng Tab Navigation (thay thế cho st.sidebar.selectbox)
tab1, tab2, tab3 = st.tabs(["💡 Tổng quan Dashboard", "🔬 Đánh giá Mô hình PD", "🔎 Dự báo & Phân tích AI"])


# --- TAB 1: Tổng quan Dashboard ---
with tab1:
    st.header("Tóm tắt Hiệu suất Mô hình")
    st.markdown("Dự báo **Xác suất Vỡ nợ (PD)** của khách hàng doanh nghiệp dựa trên bộ chỉ số tài chính (X1–X14).")
    
    # Hiển thị Metric quan trọng bằng st.metric
    col_acc, col_auc, col_f1 = st.columns(3)
    
    with col_acc:
        st.metric(label="Độ chính xác (Test Set)", value=f"{metrics_out['accuracy_out']:.2%}", delta="Tỷ lệ dự báo đúng")
    with col_auc:
        st.metric(label="AUC (Test Set)", value=f"{metrics_out['auc_out']:.3f}", delta=f"Train AUC: {metrics_in['auc_in']:.3f}")
    with col_f1:
        st.metric(label="F1 Score (Test Set)", value=f"{metrics_out['f1_out']:.2f}", delta="Cân bằng Precision/Recall")
    
    st.markdown("---")
    st.subheader("Phân phối Dữ liệu Đầu vào")
    st.dataframe(df[[f"X_{i}" for i in range(1, 15)]].describe().T.style.format("{:.3f}"))
    
    # Đoạn code hiển thị ảnh minh họa cũ
    # for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
    #     try:
    #         st.image(img)
    #     except Exception:
    #         pass # Bỏ qua lỗi nếu không tìm thấy file

# --- TAB 2: Xây dựng Mô hình (Trực quan hóa & Đánh giá chi tiết) ---
with tab2:
    st.header("Phân tích Sâu Mô hình Hồi quy Logistic")
    
    st.subheader("1. Trực quan hóa Biến và Đường Hồi quy Đơn biến")
    col_meta, col_vis = st.columns([1, 2])
    
    with col_meta:
        col = st.selectbox('Chọn Biến X muốn vẽ', options=[f"X_{i}" for i in range(1, 15)], key='vis_var')
        st.markdown(f"**Ý nghĩa:** Phân tích quan hệ giữa **{col}** và xác suất Default.")
        
    with col_vis:
        if col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                # Scatter plot data points
                sns.scatterplot(data=df, x=col, y='default', alpha=0.5, ax=ax, hue='default', palette={0: '#1f77b4', 1: '#d62728'}, legend=False)
                
                # Vẽ đường logistic regression
                x_range = np.linspace(df[col].min(), df[col].max(), 100).reshape(-1, 1)
                lr_temp = LogisticRegression(max_iter=1000)
                lr_temp.fit(df[[col]], df['default'])
                y_curve = lr_temp.predict_proba(x_range)[:, 1]
                ax.plot(x_range, y_curve, color='black', linestyle='--', linewidth=2, label='Đường Hồi quy Log')
                
                ax.set_ylabel('Xác suất Default')
                ax.set_xlabel(col)
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ: {e}")
    
    st.markdown("---")
    st.subheader("2. Ma trận Nhầm lẫn và Hiệu suất Chi tiết")
    col_cm, col_metrics_detail = st.columns([1, 2])
    
    with col_cm:
        st.markdown("**Ma trận Nhầm lẫn (Test Set)**")
        cm = confusion_matrix(y_test, y_pred_out)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Default (0)', 'Default (1)'])
        fig2, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        st.pyplot(fig2)
        plt.close()
        
    with col_metrics_detail:
        st.markdown("**Bảng so sánh Hiệu suất (Train vs Test)**")
        dt_in = pd.Series(metrics_in).rename(lambda x: x.replace('_in', '')).to_frame('Train Set')
        dt_out = pd.Series(metrics_out).rename(lambda x: x.replace('_out', '')).to_frame('Test Set')
        dt = pd.concat([dt_in, dt_out], axis=1).T
        st.dataframe(dt.style.format("{:.4f}"))

# --- TAB 3: Dự báo & Phân tích AI ---
with tab3:
    st.header("Thẩm định Hộ sơ Tín dụng và Khuyến nghị")
    
    st.caption("Tải File Excel của khách hàng (chứa 3 sheet: **CDKT ; BCTN ; LCTT**) để tính toán X1-X14.")
    
    up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx**", type=["xlsx"], key="ho_so_dn_analysis")
    
    if up_xlsx is not None:
        # Tính X1..X14
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14. Đảm bảo file Excel có đủ 3 sheet và đúng định dạng: {e}")
            st.stop()

        st.markdown("### 1. Chỉ số Tài chính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))
        
        data_for_ai = ratios_df.iloc[0].to_dict()
        
        # Dự báo PD trong Container làm nổi bật
        with st.container(border=True):
            st.subheader("2. Kết quả Dự báo Xác suất Vỡ nợ (PD)")
            
            if set(X.columns) == set(ratios_df.columns):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    col_pd, col_pred = st.columns(2)
                    
                    # Cập nhật payload cho Gemini
                    data_for_ai['PD_Probability'] = f"{probs[0]:.4f}"
                    status_text = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
                    data_for_ai['PD_Prediction'] = status_text
                    
                    with col_pd:
                        st.metric(label="Xác suất Vỡ nợ (PD)", value=f"{probs[0]:.3f}", delta="Ngưỡng 0.5")
                    with col_pred:
                        if preds[0] == 1:
                            st.error(f"🚨 RỦI RO CAO: {status_text}", icon="🚨")
                        else:
                            st.success(f"✅ RỦI RO THẤP: {status_text}", icon="✅")
                            
                except Exception as e:
                    st.warning(f"Không dự báo được PD: Lỗi {e}")
            else:
                st.warning("Mô hình PD chưa sẵn sàng hoặc cấu trúc cột không khớp.")
                
        # Phân tích AI
        st.markdown("### 3. Khuyến nghị và Phân tích chuyên sâu từ Gemini AI")
        
        if st.button("✨ Yêu cầu Gemini AI Phân tích Tín dụng", use_container_width=True, type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    # Dựa vào kết quả để dùng màu sắc phù hợp (Success/Error/Info)
                    if "KHÔNG CHO VAY" in ai_result.upper():
                        st.error(ai_result, icon="❌")
                    elif "CHO VAY" in ai_result.upper():
                        st.success(ai_result, icon="👍")
                    else:
                        st.info(ai_result)
            else:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("💡 Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
