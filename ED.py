Skip to content
Navigation Menu
truchuynh10121996-gif
RUI-RO-3

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Upgrade risk assessment web application #7
✨ 
 Merged
truchuynh10121996-gif merged 1 commit into main from claude/upgrade-risk-assessment-app-011CUhERemeT16sxZ15fdTG2  22 minutes ago
+568 −100 
 Conversation 0
 Commits 1
 Checks 0
 Files changed 1
 Merged
Upgrade risk assessment web application
#7
File filter 
 
0 / 1 files viewed
  668 changes: 568 additions & 100 deletions668  
ED.py
Viewed
Original file line number	Diff line number	Diff line change
@@ -1,8 +1,7 @@
# app.py — Streamlit PD + Phân tích Gemini (CẬP NHẬT THƯ VIỆN)
# app.py — Streamlit PD + Phân tích Gemini (CẬP NHẬT GIAO DIỆN HIỆN ĐẠI)

# =========================
# THƯ VIỆN BẮT BUỘC VÀ BỔ SUNG
# (Cần đảm bảo các gói này được cài đặt, ví dụ trong requirements.txt)
# =========================
from datetime import datetime
import os
@@ -11,6 +10,9 @@
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Thư viện Machine Learning và Mô hình
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
@@ -23,13 +25,9 @@
    roc_auc_score,
    ConfusionMatrixDisplay,
)
# Các thư viện BỔ SUNG theo yêu cầu (nếu được sử dụng trong code sau này)
# import xgboost as xgb
# import graphviz
# import statsmodels.api as sm

# =========================
# THÊM THƯ VIỆN GOOGLE GEMINI VÀ OPENAI (CHO TƯƠNG THÍCH VỚI REQ CŨ)
# THÊM THƯ VIỆN GOOGLE GEMINI
# =========================
try:
    from google import genai
@@ -48,7 +46,188 @@
    _OPENAI_OK = False


MODEL_NAME = "gemini-2.5-flash" # Model mạnh mẽ và hiệu quả cho phân tích văn bản
MODEL_NAME = "gemini-2.5-flash"

# =========================
# CẤU HÌNH TRANG VÀ CSS
# =========================
st.set_page_config(
    page_title="Đánh giá Rủi ro Tín dụng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS theme
def load_css():
    css_file = "ui/theme.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Thêm CSS bổ sung cho hiệu ứng động
    st.markdown("""
    <style>
    @keyframes slideInFromTop {
        0% {
            opacity: 0;
            transform: translateY(-30px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeInScale {
        0% {
            opacity: 0;
            transform: scale(0.95);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    .main-header {
        animation: slideInFromTop 0.6s ease-out;
    }
    .content-card {
        animation: fadeInScale 0.5s ease-out;
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .content-card:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #800000;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(128, 0, 0, 0.2);
    }
    .indicator-name {
        font-weight: 600;
        color: #800000;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .indicator-desc {
        color: #555;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    /* Gradient text cho tiêu đề */
    .gradient-title {
        background: linear-gradient(135deg, #800000 0%, #D4AF37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0;
        animation: slideInFromTop 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# =========================
# ĐỊNH NGHĨA CHỈ SỐ X1-X14
# =========================
INDICATOR_DEFINITIONS = {
    "X_1": {
        "name": "X1 - Biên lợi nhuận gộp",
        "formula": "Lợi nhuận gộp / Doanh thu thuần",
        "desc": "Đo lường khả năng sinh lời từ hoạt động kinh doanh cốt lõi, thể hiện hiệu quả quản lý giá vốn"
    },
    "X_2": {
        "name": "X2 - Biên lợi nhuận trước thuế",
        "formula": "Lợi nhuận trước thuế / Doanh thu thuần",
        "desc": "Đánh giá khả năng sinh lời tổng thể sau khi trừ mọi chi phí (trước thuế)"
    },
    "X_3": {
        "name": "X3 - ROA (Tỷ suất sinh lời trên tài sản)",
        "formula": "Lợi nhuận trước thuế / Tổng tài sản bình quân",
        "desc": "Hiệu quả sử dụng tài sản để tạo ra lợi nhuận, chỉ số quan trọng đánh giá hiệu suất doanh nghiệp"
    },
    "X_4": {
        "name": "X4 - ROE (Tỷ suất sinh lời trên vốn chủ sở hữu)",
        "formula": "Lợi nhuận trước thuế / Vốn chủ sở hữu bình quân",
        "desc": "Đo lường lợi nhuận tạo ra từ mỗi đồng vốn của chủ sở hữu, quan trọng với nhà đầu tư"
    },
    "X_5": {
        "name": "X5 - Tỷ lệ nợ trên tài sản",
        "formula": "Nợ phải trả / Tổng tài sản",
        "desc": "Đánh giá mức độ sử dụng đòn bẩy tài chính và rủi ro tài chính của doanh nghiệp"
    },
    "X_6": {
        "name": "X6 - Hệ số nợ trên vốn chủ sở hữu",
        "formula": "Nợ phải trả / Vốn chủ sở hữu",
        "desc": "Đo lường cơ cấu vốn, tỷ lệ cao cho thấy doanh nghiệp phụ thuộc nhiều vào vay nợ"
    },
    "X_7": {
        "name": "X7 - Khả năng thanh toán hiện hành",
        "formula": "Tài sản ngắn hạn / Nợ ngắn hạn",
        "desc": "Đánh giá khả năng thanh toán các khoản nợ ngắn hạn bằng tài sản ngắn hạn"
    },
    "X_8": {
        "name": "X8 - Khả năng thanh toán nhanh",
        "formula": "(Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn",
        "desc": "Đo lường khả năng thanh toán nợ ngắn hạn mà không cần bán hàng tồn kho"
    },
    "X_9": {
        "name": "X9 - Khả năng trả lãi vay",
        "formula": "EBIT / Chi phí lãi vay",
        "desc": "Đánh giá năng lực trang trải chi phí lãi vay từ lợi nhuận hoạt động"
    },
    "X_10": {
        "name": "X10 - Khả năng trả nợ gốc và lãi",
        "formula": "(EBIT + Khấu hao) / (Chi phí lãi vay + Nợ dài hạn đến hạn)",
        "desc": "Đo lường khả năng trả cả gốc và lãi từ dòng tiền hoạt động"
    },
    "X_11": {
        "name": "X11 - Tỷ lệ tiền mặt trên vốn CSH",
        "formula": "Tiền và tương đương tiền / Vốn chủ sở hữu",
        "desc": "Đánh giá tính thanh khoản cao và khả năng đáp ứng nhu cầu tài chính đột xuất"
    },
    "X_12": {
        "name": "X12 - Vòng quay hàng tồn kho",
        "formula": "Giá vốn hàng bán / Hàng tồn kho bình quân",
        "desc": "Đo lường hiệu quả quản lý hàng tồn kho, tỷ lệ cao cho thấy bán hàng nhanh"
    },
    "X_13": {
        "name": "X13 - Kỳ thu tiền bình quân (ngày)",
        "formula": "365 / (Doanh thu thuần / Phải thu khách hàng bình quân)",
        "desc": "Số ngày trung bình để thu hồi công nợ từ khách hàng"
    },
    "X_14": {
        "name": "X14 - Hiệu suất sử dụng tài sản",
        "formula": "Doanh thu thuần / Tổng tài sản bình quân",
        "desc": "Đánh giá hiệu quả tạo ra doanh thu từ tài sản, tỷ lệ cao là tốt"
    }
}

# =========================
# HÀM GỌI GEMINI API
@@ -70,7 +249,7 @@ def get_ai_analysis(data_payload: dict, api_key: str) -> str:
        "Kết thúc bằng khuyến nghị in hoa: CHO VAY hoặc KHÔNG CHO VAY, kèm 2–3 điều kiện nếu CHO VAY. "
        "Viết bằng tiếng Việt súc tích, chuyên nghiệp."
    )
    

    user_prompt = "Bộ chỉ số X1..X14 cần phân tích:\n" + str(data_payload) + "\n\nHãy phân tích và đưa ra khuyến nghị."

    try:
@@ -230,35 +409,41 @@ def div(a, b):
# UI & TRAIN MODEL
# =========================
np.random.seed(0)
st.title("DỰ BÁO THAM SỐ PD")
st.write("## Dự báo xác suất vỡ nợ của khách hàng_PD")

# Header với gradient title
st.markdown('<h1 class="gradient-title main-header">🏦 ĐÁNH GIÁ RỦI RO TÍN DỤNG CỦA KHÁCH HÀNG DOANH NGHIỆP</h1>', unsafe_allow_html=True)
st.markdown("---")

# Hiển thị trạng thái thư viện AI
st.caption("🔎 Trạng thái Gemini: " + ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai."))
col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    st.caption("🔎 **Trạng thái Gemini AI:** " + ("✅ Sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai"))
with col_status2:
    st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y')}")

# Load dữ liệu huấn luyện (CSV có default, X_1..X_14)
try:
    df = pd.read_csv('DATASET.csv', encoding='latin-1')
except Exception:
    df = None

uploaded_file = st.file_uploader("Tải CSV dữ liệu huấn luyện", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
with st.expander("📁 Tải CSV dữ liệu huấn luyện (tùy chọn)", expanded=False):
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'], key="train_data")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.success("✅ Đã tải dữ liệu huấn luyện mới!")

if df is None:
    st.info("Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.warning("⚠️ Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14).")
    st.stop()

# Kiểm tra cột cần thiết
required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Thiếu cột: {missing}")
    st.error(f"❌ Thiếu cột: {missing}")
    st.stop()

st.write(df[[f"X_{i}" for i in range(1, 15)]].describe())

# Train model
X = df.drop(columns=['default'])
y = df['default'].astype(int)
@@ -290,112 +475,395 @@ def div(a, b):
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
# =========================
# TABS NGANG (THAY THẾ SIDEBAR)
# =========================
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Sử dụng mô hình dự báo",
    "🏗️ Xây dựng mô hình",
    "📊 Biểu đồ phân tích",
    "📋 Mục tiêu của mô hình"
])

    st.write("##### 4) Ma trận nhầm lẫn (test)")
    cm = confusion_matrix(y_test, y_pred_out)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig2, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig2)
    plt.close()
# =========================
# TAB 1: SỬ DỤNG MÔ HÌNH DỰ BÁO (TRANG CHỦ)
# =========================
with tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🎯 Sử dụng mô hình để dự báo & phân tích AI")
    st.caption("📄 File Excel phải có đủ 3 sheet: **CDKT**, **BCTN**, **LCTT**")

elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo & phân tích AI (3 sheet)")
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")
    up_xlsx = st.file_uploader("📤 Tải file ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")

    up_xlsx = st.file_uploader("Tải ho_so_dn.xlsx", type=["xlsx"], key="ho_so_dn")
    if up_xlsx is not None:
        # Tính X1..X14 từ 3 sheet
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.error(f"❌ Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown("### Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))

        st.markdown("### 📊 Kết quả tính toán chỉ số tài chính X1…X14")

        # Hiển thị bảng với giá trị
        st.dataframe(
            ratios_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1),
            use_container_width=True
        )

        # Hiển thị định nghĩa từng chỉ số
        st.markdown("### 📖 Giải thích chi tiết các chỉ số")

        # Chia thành 2 cột để hiển thị đẹp hơn
        col_left, col_right = st.columns(2)

        for idx, (col_name, definition) in enumerate(INDICATOR_DEFINITIONS.items()):
            target_col = col_left if idx % 2 == 0 else col_right

            with target_col:
                value = ratios_df[col_name].values[0]
                st.markdown(f"""
                <div class="metric-box">
                    <div class="indicator-name">{definition['name']}</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c5aa0; margin: 0.5rem 0;">
                        {value:.4f if pd.notna(value) else 'N/A'}
                    </div>
                    <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">
                        📐 Công thức: <code>{definition['formula']}</code>
                    </div>
                    <div class="indicator-desc">
                        💡 {definition['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Tạo payload data cho AI
        data_for_ai = ratios_df.iloc[0].to_dict()

        # (Tuỳ chọn) dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        # Dự báo PD nếu mô hình đã huấn luyện đúng cấu trúc X_1..X_14
        if set(X.columns) == set(ratios_df.columns):
            with st.expander("Xác suất vỡ nợ dự báo (nếu đã huấn luyện ở trên)"):
            with st.expander("🔮 Xác suất vỡ nợ dự báo", expanded=True):
                try:
                    probs = model.predict_proba(ratios_df[X.columns])[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    # Hiển thị kết quả dự báo nổi bật
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Xác suất vỡ nợ (PD)", f"{probs[0]:.1%}")
                    with col2:
                        status = "⚠️ Default" if preds[0] == 1 else "✅ Non-Default"
                        st.metric("🎯 Dự báo", status)
                    with col3:
                        risk_level = "Cao" if probs[0] > 0.7 else ("Trung bình" if probs[0] > 0.3 else "Thấp")
                        st.metric("⚡ Mức rủi ro", risk_level)

                    show = ratios_df.copy()
                    show["pd"] = probs
                    show["pred_default"] = preds
                    st.dataframe(show.style.format({"pd": "{:.3f}"}))
                    show["PD"] = probs
                    show["Dự báo"] = ["Default" if p == 1 else "Non-Default" for p in preds]
                    st.dataframe(show.style.format({"PD": "{:.3%}"}), use_container_width=True)

                    # Thêm vào payload cho AI
                    data_for_ai['PD_Probability'] = probs[0]
                    data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"

                except Exception as e:
                    st.warning(f"Không dự báo được PD: {e}")
                    st.warning(f"⚠️ Không dự báo được PD: {e}")

        # Gemini Phân tích & khuyến nghị - ĐOẠN CODE BẠN YÊU CẦU THÊM VÀO ĐÂY
        st.markdown("### Phân tích AI & đề xuất CHO VAY/KHÔNG CHO VAY")

        # Thêm các chỉ số PD nếu đã tính được vào payload
        if 'probs' in locals():
            data_for_ai['PD_Probability'] = probs[0]
            data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
        # Gemini Phân tích & khuyến nghị
        st.markdown("---")
        st.markdown("### 🤖 Phân tích AI & đề xuất tín dụng")

        if st.button("Yêu cầu AI Phân tích"):
        if st.button("🚀 Yêu cầu AI Phân tích", use_container_width=True):
            api_key = st.secrets.get("GEMINI_API_KEY")
            

            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                with st.spinner('⏳ Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.markdown("**📝 Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")
                st.error("❌ Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("💡 Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet: CDKT, BCTN, LCTT) để bắt đầu phân tích.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 2: XÂY DỰNG MÔ HÌNH
# =========================
with tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🏗️ Xây dựng mô hình dự báo")

    st.markdown("#### 1️⃣ Hiển thị dữ liệu huấn luyện")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**📊 Dữ liệu đầu:**")
        st.dataframe(df.head(3), use_container_width=True)
    with col2:
        st.write("**📊 Dữ liệu cuối:**")
        st.dataframe(df.tail(3), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2️⃣ Trực quan hóa dữ liệu")
    col_input = st.text_input('🔍 Nhập tên biến X muốn vẽ (ví dụ: X_1)', value='X_1')

    if col_input in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=col_input, y='default', alpha=0.5, s=100, ax=ax)

            # Vẽ đường logistic regression theo 1 biến
            x_range = np.linspace(df[col_input].min(), df[col_input].max(), 100)
            X_temp = df[[col_input]].copy()
            y_temp = df['default']
            lr_temp = LogisticRegression(max_iter=1000)
            lr_temp.fit(X_temp, y_temp)
            x_test = pd.DataFrame({col_input: x_range})
            y_curve = lr_temp.predict_proba(x_test)[:, 1]

            ax.plot(x_range, y_curve, color='#800000', linewidth=3, label='Logistic Curve')
            ax.set_ylabel('Xác suất default', fontsize=12, fontweight='bold')
            ax.set_xlabel(col_input, fontsize=12, fontweight='bold')
            ax.set_title(f'Mối quan hệ giữa {col_input} và Default', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"❌ Lỗi khi vẽ biểu đồ: {e}")
    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
        st.warning("⚠️ Biến không tồn tại trong dữ liệu.")

    st.markdown("---")
    st.markdown("#### 3️⃣ Kết quả đánh giá mô hình")

    metrics_df = pd.DataFrame([metrics_in | metrics_out])

    # Hiển thị metrics dạng card
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 In-sample Performance**")
        st.dataframe(
            metrics_df[['accuracy_in', 'precision_in', 'recall_in', 'f1_in', 'auc_in']].T.rename(columns={0: 'Value'}).style.format("{:.4f}").background_gradient(cmap='Greens'),
            use_container_width=True
        )
    with col2:
        st.markdown("**📊 Out-of-sample Performance**")
        st.dataframe(
            metrics_df[['accuracy_out', 'precision_out', 'recall_out', 'f1_out', 'auc_out']].T.rename(columns={0: 'Value'}).style.format("{:.4f}").background_gradient(cmap='Blues'),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 4️⃣ Ma trận nhầm lẫn (Test set)")
    cm = confusion_matrix(y_test, y_pred_out)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Default', 'Default'])
    fig2, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='RdYlGn_r', values_format='d')
    ax.set_title('Ma trận nhầm lẫn - Test Set', fontsize=14, fontweight='bold')
    st.pyplot(fig2)
    plt.close()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 3: BIỂU ĐỒ PHÂN TÍCH (MỚI)
# =========================
with tab3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📊 Biểu đồ phân tích dữ liệu")

    # Biểu đồ 1: Phân bố Default
    st.markdown("#### 📈 1. Phân bố tỷ lệ Default/Non-Default")
    col1, col2 = st.columns([2, 1])

    with col1:
        default_counts = df['default'].value_counts()
        fig1 = go.Figure(data=[
            go.Pie(
                labels=['Non-Default', 'Default'],
                values=default_counts.values,
                hole=0.4,
                marker=dict(colors=['#2ecc71', '#e74c3c']),
                textinfo='label+percent',
                textfont=dict(size=14)
            )
        ])
        fig1.update_layout(
            title="Phân bố Default trong dữ liệu huấn luyện",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.metric("Tổng số mẫu", len(df))
        st.metric("Non-Default", default_counts[0])
        st.metric("Default", default_counts[1])
        st.metric("Tỷ lệ Default", f"{default_counts[1]/len(df):.1%}")

    st.markdown("---")

    # Biểu đồ 2: Correlation Heatmap
    st.markdown("#### 🔥 2. Ma trận tương quan giữa các chỉ số")

    corr_matrix = df[[f"X_{i}" for i in range(1, 15)]].corr()

    fig2 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig2.update_layout(
        title="Ma trận tương quan giữa các chỉ số X1-X14",
        height=600,
        xaxis_title="Chỉ số",
        yaxis_title="Chỉ số"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Biểu đồ 3: Box plot cho một số chỉ số quan trọng
    st.markdown("#### 📦 3. Phân bố các chỉ số quan trọng theo Default")

    selected_indicators = st.multiselect(
        "Chọn chỉ số muốn xem:",
        options=[f"X_{i}" for i in range(1, 15)],
        default=["X_1", "X_3", "X_4", "X_7"]
    )

    if selected_indicators:
        fig3 = go.Figure()

        for indicator in selected_indicators:
            # Non-default
            fig3.add_trace(go.Box(
                y=df[df['default'] == 0][indicator],
                name=f'{indicator} (Non-Default)',
                marker_color='#2ecc71'
            ))
            # Default
            fig3.add_trace(go.Box(
                y=df[df['default'] == 1][indicator],
                name=f'{indicator} (Default)',
                marker_color='#e74c3c'
            ))

        fig3.update_layout(
            title="So sánh phân bố chỉ số giữa Default và Non-Default",
            yaxis_title="Giá trị",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Biểu đồ 4: Feature Importance (dựa trên coefficients)
    st.markdown("#### 🎯 4. Mức độ quan trọng của các chỉ số")

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': np.abs(model.coef_[0])
    }).sort_values('Coefficient', ascending=True)

    fig4 = go.Figure(go.Bar(
        x=feature_importance['Coefficient'],
        y=feature_importance['Feature'],
        orientation='h',
        marker=dict(
            color=feature_importance['Coefficient'],
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig4.update_layout(
        title="Mức độ ảnh hưởng của các chỉ số trong mô hình",
        xaxis_title="Absolute Coefficient",
        yaxis_title="Chỉ số",
        height=500
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Biểu đồ 5: Thống kê mô tả
    st.markdown("#### 📋 5. Thống kê mô tả các chỉ số")
    st.dataframe(
        df[[f"X_{i}" for i in range(1, 15)]].describe().T.style.format("{:.4f}").background_gradient(cmap='coolwarm', axis=1),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 4: MỤC TIÊU CỦA MÔ HÌNH
# =========================
with tab4:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📋 Mục tiêu của mô hình")

    st.markdown("""
    ### 🎯 Mục tiêu chính
    **Dự báo xác suất vỡ nợ (Probability of Default - PD)** của khách hàng doanh nghiệp
    dựa trên bộ 14 chỉ số tài chính quan trọng (X1–X14).
    ### 🔍 Phạm vi ứng dụng
    - ✅ Đánh giá rủi ro tín dụng trước khi cho vay
    - ✅ Phân loại khách hàng theo mức độ rủi ro
    - ✅ Hỗ trợ quyết định tín dụng dựa trên dữ liệu
    - ✅ Tối ưu hóa danh mục cho vay
    ### 💡 Lợi ích
    1. **Giảm thiểu rủi ro**: Phát hiện sớm khách hàng tiềm ẩn rủi ro cao
    2. **Tăng hiệu quả**: Tự động hóa quy trình đánh giá tín dụng
    3. **Minh bạch**: Dựa trên các chỉ số tài chính rõ ràng, có thể giải thích
    4. **Hỗ trợ AI**: Tích hợp phân tích Gemini AI cho góc nhìn chuyên sâu
    ### 📊 Phương pháp
    - **Mô hình**: Logistic Regression
    - **Input**: 14 chỉ số tài chính từ 3 báo cáo: CDKT, BCTN, LCTT
    - **Output**: Xác suất vỡ nợ (0-100%) và phân loại Default/Non-Default
    """)

    st.markdown("---")
    st.markdown("### 🖼️ Minh họa")

    # Hiển thị ảnh minh họa
    image_cols = st.columns(3)
    images = ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]

    for idx, img in enumerate(images):
        try:
            with image_cols[idx]:
                st.image(img, use_column_width=True, caption=f"Hình minh họa {idx+1}")
        except Exception:
            pass

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏦 <strong>ĐÁNH GIÁ RỦI RO TÍN DỤNG</strong> | Phát triển bởi Streamlit + Gemini AI</p>
    <p style='font-size: 0.85rem;'>© 2025 - Hệ thống hỗ trợ quyết định tín dụng thông minh</p>
</div>
""", unsafe_allow_html=True)
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
