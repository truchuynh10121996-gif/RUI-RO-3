# app.py — Streamlit PD + phân tích GPT (ĐÃ BỎ GPT/OPENAI)
from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# KHÔNG CẦN DÙNG NỮA VÌ ĐÃ BỎ OPENAI
# MODEL_NAME = "gpt-4o-mini" 

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
# OPENAI SAFE LOADER - ĐÃ BỊ BỎ HOÀN TOÀN
# =========================
# try:
#     from openai import OpenAI  # yêu cầu openai>=1.30
#     _OPENAI_OK = True
# except Exception:
#     OpenAI = None
#     _OPENAI_OK = False

# def get_openai_client():
#     """
#     Ưu tiên: st.secrets["OPENAI_API_KEY"] -> os.getenv -> nhập tạm ở sidebar (không lưu).
#     Trả về (client, err). Nếu client=None => dùng err để hiển thị cảnh báo.
#     """
#     key = None
#     try:
#         key = st.secrets.get("OPENAI_API_KEY", None)
#     except Exception:
#         pass
#     if not key:
#         key = os.getenv("OPENAI_API_KEY")
#     if not key:
#         # cho nhập tạm (không lưu/commit)
#         tmp = st.sidebar.text_input("🔐 OpenAI API Key (không lưu)", type="password")
#         if tmp:
#             key = tmp

#     if not _OPENAI_OK:
#         return None, "Thiếu thư viện openai (cần openai>=1.30)."
#     if not key:
#         return None, "Thiếu OPENAI_API_KEY (đặt trong Secrets/ENV hoặc nhập tạm ở sidebar)."
#     return OpenAI(api_key=key), None

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
# UI & TRAIN MODEL
# =========================
np.random.seed(0)
st.title("DỰ BÁO THAM SỐ PD")
st.write("## Dự báo xác suất vỡ nợ của khách hàng_PD")

# Đã loại bỏ phần hiển thị trạng thái OpenAI
# _client_probe, _err_probe = get_openai_client()
# st.caption("🔎 Trạng thái OpenAI: " + ("✅ sẵn sàng" if _client_probe else f"⚠️ {_err_probe}"))
st.caption("🔎 Trạng thái phân tích: **Chức năng phân tích GPT đã bị loại bỏ theo yêu cầu.**")

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

menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số X1–X14.")
    # ảnh minh họa (có thể không tồn tại)
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except Exception:
            st.warning(f"Không tìm thấy {img}")

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
            # Vẽ đường logistic regression theo 1 biến
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

        st.markdown("### Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))

        # (Tuỳ chọn) dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        if set(X.columns) == set(ratios_df.columns):
            with st.expander("Xác suất vỡ nợ dự báo (nếu đã huấn luyện ở trên)"):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    show = ratios_df.copy()
                    show["pd"] = probs
                    show["pred_default"] = preds
                    st.dataframe(show.style.format({"pd": "{:.3f}"}))
                except Exception as e:
                    st.warning(f"Không dự báo được PD: {e}")

        # GPT phân tích & khuyến nghị - ĐÃ BỊ BỎ HOÀN TOÀN
        st.markdown("### Phân tích GPT & đề xuất CHO VAY/KHÔNG CHO VAY")
        st.warning("Chức năng phân tích GPT đã bị loại bỏ theo yêu cầu. Vui lòng tự phân tích dựa trên kết quả PD.")
        # client, err = get_openai_client()
        # if client is None:
        #     st.warning(err + " — bỏ qua phân tích GPT.")
        # else:
        #     payload = ratios_df.iloc[0].to_dict()
        #     # gợi ý rule-of-thumb
        #     flags = []
        #     if pd.notna(payload.get("X_5")) and payload["X_5"] > 0.8:
        #         flags.append("Đòn bẩy cao (X5>0.8)")
        #     if pd.notna(payload.get("X_7")) and payload["X_7"] < 1.0:
        #         flags.append("Thanh khoản yếu (X7<1)")
        #     if flags:
        #         payload["ghi_chu"] = " ; ".join(flags)

        #     sys_prompt = (
        #         "Bạn là chuyên gia phân tích tín dụng doanh nghiệp tại ngân hàng. "
        #         "Phân tích toàn diện dựa trên X1..X14. "
        #         "Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4) Hiệu quả hoạt động. "
        #         "Kết thúc bằng khuyến nghị in hoa: CHO VAY hoặc KHÔNG CHO VAY, kèm 2–3 điều kiện nếu CHO VAY."
        #     )
        #     user_prompt = "Bộ chỉ số:\n" + str(payload) + "\n\nViết súc tích, tiếng Việt, dùng gạch đầu dòng khi hợp lý."

        #     with st.spinner("GPT đang phân tích..."):
        #         try:
        #             resp = client.chat.completions.create(
        #                 model=MODEL_NAME,
        #                 messages=[
        #                     {"role": "system", "content": sys_prompt},
        #                     {"role": "user", "content": user_prompt},
        #                 ],
        #             )
        #             st.write(resp.choices[0].message.content)
        #         except Exception as e:
        #             st.error(f"Lỗi gọi GPT: {e}")
    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14 và nhận xác suất PD.")
