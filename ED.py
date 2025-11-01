# app.py — Streamlit PD từ 3 sheet CDKT/BCTN/LCTT (Không gọi AI API)

import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình matplotlib
import matplotlib
matplotlib.use('Agg')
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
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Dự báo PD",
    layout="wide"
)

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

def _pick_year_cols(df):
    """Chọn 2 cột năm gần nhất."""
    numeric_years = []
    for c in df.columns[1:]:
        try:
            y = int(float(str(c).strip()))
            if 1990 <= y <= 2100:
                numeric_years.append((y, c))
        except:
            continue
    if numeric_years:
        numeric_years.sort(key=lambda x: x[0])
        return numeric_years[-2][1], numeric_years[-1][1]
    cols = df.columns[-2:]
    return cols[0], cols[1]

def _get_row_vals(df, aliases):
    """Tìm dòng theo alias và trả về (prev, cur)."""
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
        except:
            return np.nan
    
    return to_num(row[prev_col]), to_num(row[cur_col])

@st.cache_data
def compute_ratios_from_three_sheets(xlsx_file):
    """Đọc 3 sheet và tính X1..X14."""
    bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
    is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
    cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")
    
    # BCTN
    DTT_prev, DTT_cur = _get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
    GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
    LNG_prev, LNG_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
    LNTT_prev, LNTT_cur = _get_row_vals(is_, ALIAS_IS["loi_nhuan_truoc_thue"])
    LV_prev, LV_cur = _get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])
    
    # CDKT
    TTS_prev, TTS_cur = _get_row_vals(bs, ALIAS_BS["tong_tai_san"])
    VCSH_prev, VCSH_cur = _get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
    NPT_prev, NPT_cur = _get_row_vals(bs, ALIAS_BS["no_phai_tra"])
    TSNH_prev, TSNH_cur = _get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
    NNH_prev, NNH_cur = _get_row_vals(bs, ALIAS_BS["no_ngan_han"])
    HTK_prev, HTK_cur = _get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
    Tien_prev, Tien_cur = _get_row_vals(bs, ALIAS_BS["tien_tdt"])
    KPT_prev, KPT_cur = _get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
    NDH_prev, NDH_cur = _get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])
    
    # LCTT
    KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])
    
    # Chuẩn hóa
    if pd.notna(GVHB_cur): GVHB_cur = abs(GVHB_cur)
    if pd.notna(LV_cur): LV_cur = abs(LV_cur)
    if pd.notna(KH_cur): KH_cur = abs(KH_cur)
    
    def avg(a, b):
        if pd.isna(a) and pd.isna(b): return np.nan
        if pd.isna(a): return b
        if pd.isna(b): return a
        return (a + b) / 2.0
    
    TTS_avg = avg(TTS_cur, TTS_prev)
    VCSH_avg = avg(VCSH_cur, VCSH_prev)
    HTK_avg = avg(HTK_cur, HTK_prev)
    KPT_avg = avg(KPT_cur, KPT_prev)
    
    EBIT_cur = (LNTT_cur + LV_cur) if (pd.notna(LNTT_cur) and pd.notna(LV_cur)) else np.nan
    NDH_cur = 0.0 if pd.isna(NDH_cur) else NDH_cur
    
    def div(a, b):
        return np.nan if (b is None or pd.isna(b) or b == 0) else a / b
    
    # TÍNH X1..X14
    X1 = div(LNG_cur, DTT_cur)
    X2 = div(LNTT_cur, DTT_cur)
    X3 = div(LNTT_cur, TTS_avg)
    X4 = div(LNTT_cur, VCSH_avg)
    X5 = div(NPT_cur, TTS_cur)
    X6 = div(NPT_cur, VCSH_cur)
    X7 = div(TSNH_cur, NNH_cur)
    X8 = div((TSNH_cur - HTK_cur) if pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)
    X9 = div(EBIT_cur, LV_cur)
    X10 = div((EBIT_cur + (KH_cur if pd.notna(KH_cur) else 0.0)),
              (LV_cur + NDH_cur) if pd.notna(LV_cur) else np.nan)
    X11 = div(Tien_cur, VCSH_cur)
    X12 = div(GVHB_cur, HTK_avg)
    turnover = div(DTT_cur, KPT_avg)
    X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else np.nan
    X14 = div(DTT_cur, TTS_avg)
    
    ratios = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14]],
                          columns=[f"X_{i}" for i in range(1, 15)])
    return ratios

# =========================
# GIAO DIỆN CHÍNH
# =========================
np.random.seed(0)
st.title("DỰ BÁO THAM SỐ PD")
st.write("## Dự báo xác suất vỡ nợ của khách hàng")

# Load CSV huấn luyện
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except:
    df = None

uploaded_file = st.file_uploader("Tải CSV dữ liệu huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

if df is None:
    st.info("Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

# Kiểm tra cột
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

# Đánh giá
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
menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':
    st.subheader("Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD)** của khách hàng doanh nghiệp dựa trên X1–X14.")
    
    st.markdown("""
    ### Các chỉ số X1-X14:
    - **X1**: Biên lợi nhuận gộp
    - **X2**: Biên lợi nhuận trước thuế
    - **X3**: ROA (trước thuế)
    - **X4**: ROE (trước thuế)
    - **X5**: Nợ/Tài sản
    - **X6**: Nợ/Vốn chủ sở hữu
    - **X7**: Thanh toán hiện hành
    - **X8**: Thanh toán nhanh
    - **X9**: Khả năng trả lãi
    - **X10**: Khả năng trả nợ gốc
    - **X11**: Tiền/VCSH
    - **X12**: Vòng quay hàng tồn kho
    - **X13**: Kỳ thu tiền bình quân
    - **X14**: Hiệu suất sử dụng tài sản
    """)
    
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except:
            pass

elif choice == 'Xây dựng mô hình':
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
            st.error(f"Lỗi vẽ biểu đồ: {e}")
    else:
        st.warning("Biến không tồn tại.")
    
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
    st.subheader("Sử dụng mô hình để dự báo")
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")
    
    up_xlsx = st.file_uploader("Tải ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")
    if up_xlsx is not None:
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.stop()
        
        st.markdown("### Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))
        
        # Dự báo PD
        if set(X.columns) == set(ratios_df.columns):
            st.markdown("### Xác suất vỡ nợ dự báo")
            try:
                probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Xác suất vỡ nợ (PD)",
                        value=f"{probs[0]:.2%}",
                        delta="Cao" if probs[0] > 0.5 else "Thấp",
                        delta_color="inverse"
                    )
                with col2:
                    st.metric(
                        label="Dự báo",
                        value="VỠ NỢ" if preds[0] == 1 else "KHÔNG VỠ NỢ",
                        delta=None
                    )
                
                # Phân tích ngưỡng rủi ro
                st.markdown("### Đánh giá rủi ro")
                if probs[0] < 0.3:
                    st.success("✅ **RỦI RO THẤP**: Xác suất vỡ nợ dưới 30%, khả năng cho vay tốt.")
                elif probs[0] < 0.5:
                    st.info("⚠️ **RỦI RO TRUNG BÌNH**: Xác suất vỡ nợ 30-50%, cần xem xét thêm điều kiện.")
                else:
                    st.error("🚫 **RỦI RO CAO**: Xác suất vỡ nợ trên 50%, khuyến nghị không cho vay hoặc yêu cầu thế chấp cao.")
                
                # Bảng chi tiết
                show = ratios_df.copy()
                show["Xác suất vỡ nợ (%)"] = probs * 100
                show["Dự báo"] = ["VỠ NỢ" if p == 1 else "KHÔNG VỠ NỢ" for p in preds]
                st.dataframe(show.style.format({
                    **{f"X_{i}": "{:.4f}" for i in range(1, 15)},
                    "Xác suất vỡ nợ (%)": "{:.2f}%"
                }))
                
            except Exception as e:
                st.warning(f"Không dự báo được: {e}")
        else:
            st.error("Cấu trúc X_1..X_14 không khớp với mô hình huấn luyện.")
    else:
        st.info("📁 Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet: CDKT, BCTN, LCTT)")
        
        with st.expander("📖 Hướng dẫn sử dụng"):
            st.markdown("""
            **Cấu trúc file Excel yêu cầu:**
            
            1. **Sheet CDKT** (Cân đối kế toán):
               - Các chỉ tiêu: Tổng tài sản, Vốn chủ sở hữu, Nợ phải trả, Tài sản ngắn hạn, Nợ ngắn hạn, v.v.
            
            2. **Sheet BCTN** (Báo cáo thu nhập):
               - Các chỉ tiêu: Doanh thu thuần, Giá vốn, Lợi nhuận gộp, Chi phí lãi vay, v.v.
            
            3. **Sheet LCTT** (Lưu chuyển tiền tệ):
               - Các chỉ tiêu: Khấu hao TSCĐ
            
            **Lưu ý:** File CSV huấn luyện phải có cột 'default' và X_1 đến X_14
            """)
