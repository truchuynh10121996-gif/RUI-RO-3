# (GIỮ NGUYÊN TOÀN BỘ CÁC PHẦN KHAI BÁO THƯ VIỆN, HÀM get_ai_analysis VÀ HÀM compute_ratios_from_three_sheets BÊN TRÊN)

# =========================
# UI & TRAIN MODEL (ĐÃ NÂNG CẤP)
# =========================

# 1. Cấu hình Trang và CSS Tùy chỉnh (Hiện đại hóa giao diện)
st.set_page_config(
    page_title="Hệ thống Phân tích & Dự báo PD Doanh nghiệp",
    page_icon="🏦",
    layout="wide", # Sử dụng toàn bộ chiều rộng màn hình (rất quan trọng)
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
    
    # (Có thể thêm ảnh minh họa như cũ nếu có file)


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
