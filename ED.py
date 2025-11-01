# app.py — Streamlit PD + Phân tích Gemini (NÂNG CẤP GIAO DIỆN)

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

MODEL_NAME = "gemini-2.0-flash-exp"

# =========================
# HÀM GỌI GEMINI API
# =========================
def get_ai_analysis(data_payload: dict, api_key: str) -> str:
    """Sử dụng Gemini API để phân tích chỉ số tài chính."""
    if not _GEMINI_OK:
        return "Lỗi: Thiếu thư viện google-genai (cần cài đặt: pip install google-genai)."

    client = genai.Client(api_key=api_key)

    sys_prompt = (
        "Bạn là chuyên gia phân tích tín dụng doanh nghiệp tại ngân hàng với 15 năm kinh nghiệm. "
        "Phân tích toàn diện dựa trên 14 chỉ số tài chính (X1..X14) và xác suất vỡ nợ (PD). "
        "Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4) Hiệu quả hoạt động. "
        "Kết thúc bằng khuyến nghị in hoa: **CHO VAY** hoặc **KHÔNG CHO VAY**, kèm 2–3 điều kiện cụ thể nếu CHO VAY. "
        "Viết bằng tiếng Việt súc tích, chuyên nghiệp, sử dụng markdown để format đẹp."
    )
    
    user_prompt = f"""
Phân tích hồ sơ tín dụng với các thông tin sau:

**BỘ CHỈ SỐ TÀI CHÍNH X1-X14:**
{str(data_payload)}

Hãy phân tích chi tiết và đưa ra khuyến nghị cho vay dựa trên:
- Điểm mạnh/yếu của doanh nghiệp
- Mức độ rủi ro (THẤP/TRUNG BÌNH/CAO)
- Khuyến nghị cuối cùng: CHO VAY hoặc KHÔNG CHO VAY
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config={
                "system_instruction": sys_prompt,
                "temperature": 0.3,
                "max_output_tokens": 2048,
            }
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

def compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
    bs = pd.read_excel(xlsx_file, sheet_name="CDKT", engine="openpyxl")
    is_ = pd.read_excel(xlsx_file, sheet_name="BCTN", engine="openpyxl")
    cf = pd.read_excel(xlsx_file, sheet_name="LCTT", engine="openpyxl")

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
    if pd.notna(LV_cur):    LV_cur    = abs(LV_cur)
    if pd.notna(KH_cur):    KH_cur    = abs(KH_cur)

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

    X1  = div(LNG_cur, DTT_cur)
    X2  = div(LNTT_cur, DTT_cur)
    X3  = div(LNTT_cur, TTS_avg)
    X4  = div(LNTT_cur, VCSH_avg)
    X5  = div(NPT_cur,  TTS_cur)
    X6  = div(NPT_cur,  VCSH_cur)
    X7  = div(TSNH_cur, NNH_cur)
    X8  = div((TSNH_cur - HTK_cur) if pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)
    X9  = div(EBIT_cur, LV_cur)
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

# ẨN TIÊU ĐỀ CHÍNH (chỉ hiện trong sidebar)
st.sidebar.title("🏦 DỰ BÁO PD")
st.sidebar.caption("Dự báo xác suất vỡ nợ khách hàng")

st.caption("🔎 Trạng thái Gemini: " + ("✅ sẵn sàng" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai"))

# Load CSV huấn luyện
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df = None

uploaded_file = st.file_uploader("📊 Tải CSV dữ liệu huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

if df is None:
    st.info("Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

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
choice = st.sidebar.selectbox('📋 Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':
    st.title("🎯 Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD)** của khách hàng doanh nghiệp dựa trên X1–X14.")
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except Exception:
            st.warning(f"Không tìm thấy {img}")

elif choice == 'Xây dựng mô hình':
    st.title("🔧 Xây dựng mô hình")
    
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
    # =====================================
    # GIAO DIỆN NÂNG CẤP - ẨN TIÊU ĐỀ CHÍNH
    # =====================================
    
    st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Upload file - Giao diện đẹp hơn
    st.markdown("### 📂 Tải hồ sơ tài chính")
    st.caption("File Excel phải có đủ 3 sheet: **CDKT** | **BCTN** | **LCTT**")
    
    up_xlsx = st.file_uploader("", type=["xlsx"], key="ho_so_dn", label_visibility="collapsed")
    
    if up_xlsx is not None:
        # Tính X1..X14
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"❌ Lỗi tính X1…X14: {e}")
            st.stop()

        # Hiển thị kết quả trong tabs
        tab1, tab2, tab3 = st.tabs(["📊 Chỉ số X1-X14", "🎯 Dự báo PD", "🤖 Phân tích AI"])
        
        with tab1:
            st.markdown("#### Bộ chỉ số tài chính")
            st.dataframe(ratios_df.style.format("{:.4f}"), use_container_width=True)
            
            # Giải thích chỉ số
            with st.expander("ℹ️ Ý nghĩa các chỉ số"):
                st.markdown("""
                - **X1**: Biên lợi nhuận gộp | **X2**: Biên LNTT | **X3**: ROA | **X4**: ROE
                - **X5**: Nợ/Tài sản | **X6**: Nợ/VCSH | **X7**: Thanh toán hiện hành | **X8**: Thanh toán nhanh
                - **X9**: Khả năng trả lãi | **X10**: Khả năng trả nợ gốc | **X11**: Tiền/VCSH
                - **X12**: Vòng quay HTK | **X13**: Kỳ thu tiền BQ | **X14**: Hiệu suất tài sản
                """)
        
        with tab2:
            st.markdown("#### Kết quả dự báo xác suất vỡ nợ")
            
            if set(X.columns) == set(ratios_df.columns):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    # Hiển thị metrics đẹp
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Xác suất vỡ nợ (PD)",
                            value=f"{probs[0]:.1%}",
                            delta=f"{probs[0]*100:.1f}% điểm" if probs[0] > 0.5 else None
                        )
                    
                    with col2:
                        pred_text = "🔴 VỠ NỢ" if preds[0] == 1 else "🟢 AN TOÀN"
                        st.metric(label="Kết luận", value=pred_text)
                    
                    with col3:
                        if probs[0] < 0.3:
                            risk = "🟢 THẤP"
                        elif probs[0] < 0.5:
                            risk = "🟡 TRUNG BÌNH"
                        else:
                            risk = "🔴 CAO"
                        st.metric(label="Mức độ rủi ro", value=risk)
                    
                    # Progress bar
                    st.markdown("##### Thang đánh giá rủi ro")
                    st.progress(probs[0])
                    
                    # Chi tiết phân tích
                    if probs[0] < 0.3:
                        st.success("✅ **ĐÁNH GIÁ**: Rủi ro thấp, khả năng cho vay tốt.")
                    elif probs[0] < 0.5:
                        st.warning("⚠️ **ĐÁNH GIÁ**: Rủi ro trung bình, cần xem xét thêm điều kiện đảm bảo.")
                    else:
                        st.error("🚫 **ĐÁNH GIÁ**: Rủi ro cao, khuyến nghị không cho vay hoặc yêu cầu tài sản thế chấp cao.")
                    
                    # Bảng chi tiết
                    show = ratios_df.copy()
                    show["PD (%)"] = probs * 100
                    show["Dự báo"] = ["VỠ NỢ" if p == 1 else "AN TOÀN" for p in preds]
                    st.dataframe(show.style.format({
                        **{f"X_{i}": "{:.4f}" for i in range(1, 15)},
                        "PD (%)": "{:.2f}%"
                    }), use_container_width=True)
                    
                    # Lưu data cho AI
                    data_for_ai = ratios_df.iloc[0].to_dict()
                    data_for_ai['PD_Probability'] = probs[0]
                    data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
                    
                except Exception as e:
                    st.warning(f"Không dự báo được PD: {e}")
                    data_for_ai = ratios_df.iloc[0].to_dict()
            else:
                st.error("⚠️ Cấu trúc X_1..X_14 không khớp với mô hình.")
                data_for_ai = ratios_df.iloc[0].to_dict()
        
        with tab3:
            st.markdown("#### Phân tích chuyên sâu bằng Gemini AI")
            
            if st.button("🚀 Yêu cầu Gemini AI Phân tích", type="primary", use_container_width=True):
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if api_key:
                    with st.spinner('⏳ Gemini AI đang phân tích hồ sơ tín dụng...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("---")
                        st.markdown("##### 📋 Báo cáo phân tích từ Gemini AI")
                        st.markdown(ai_result)
                else:
                    st.error("❌ Lỗi: Không tìm thấy **GEMINI_API_KEY** trong Streamlit Secrets.")
                    st.info("Vui lòng cấu hình API key tại: Settings → Secrets")
    else:
        # Hướng dẫn khi chưa upload
        st.info("📁 Vui lòng tải file **ho_so_dn.xlsx** để bắt đầu phân tích")
        
        with st.expander("📖 Hướng dẫn sử dụng"):
            st.markdown("""
            ### Cấu trúc file Excel yêu cầu:
            
            **1. Sheet CDKT** (Cân đối kế toán):
            - Tổng tài sản, Vốn chủ sở hữu, Nợ phải trả
            - Tài sản ngắn hạn, Nợ ngắn hạn, Hàng tồn kho
            - Tiền và tương đương tiền, Phải thu khách hàng
            
            **2. Sheet BCTN** (Báo cáo thu nhập):
            - Doanh thu thuần, Giá vốn hàng bán, Lợi nhuận gộp
            - Chi phí lãi vay, Lợi nhuận trước thuế
            
            **3. Sheet LCTT** (Lưu chuyển tiền tệ):
            - Khấu hao TSCĐ
            
            ### Cấu hình Gemini API:
            1. Lấy API key tại: https://aistudio.google.com/apikey
            2. Thêm vào Secrets: `GEMINI_API_KEY = "your-key"`
            """)
