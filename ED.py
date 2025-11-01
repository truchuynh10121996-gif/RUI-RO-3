# app.py — Streamlit PD + Phân tích Gemini (FIX LỖI KHÔNG KHỚP CẤU TRÚC DỮ LIỆU)

# =========================
# (PHẦN THƯ VIỆN, HÀM GEMINI, HÀM COMPUTE_RATIOS GIỮ NGUYÊN)
# ... (Phần code này được lược bỏ để tập trung vào sửa lỗi chính) ...
# =========================

# --- LOGIC CSS VÀ LOGO (GIỮ NGUYÊN) ---
BRIGHT_BORDEAUX = "#A50000" 
AGRIBANK_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Agribank_logo.svg/1024px-Agribank_logo.svg.png" 

st.markdown(
    f"""
    <style>
        /* ... (CSS cho Logo và màu sắc giữ nguyên) ... */
        [data-testid="stSidebar"] {{ padding-top: 50px; }}
        .logo-img {{
            position: fixed; top: 10px; left: 20px;
            width: 100px; height: auto; z-index: 1000;
        }}
        .st-emotion-cache-1wivap2 {{ color: {BRIGHT_BORDEAUX} !important; }}
        h1, h2, h3, h4, h5, h6 {{ color: {BRIGHT_BORDEAUX} !important; }}
        div.stButton > button:first-child {{
            background-color: {BRIGHT_BORDEAUX}; color: white; border-color: {BRIGHT_BORDEAUX};
        }}
        div.stButton > button:hover {{
            background-color: #7A0000; color: white; border-color: #7A0000;
        }}
        .st-emotion-cache-13l3763 {{
            background-color: #FFF0F0; border-left: 5px solid {BRIGHT_BORDEAUX};
        }}
    </style>
    <img src="{AGRIBANK_LOGO_URL}" class="logo-img">
    """,
    unsafe_allow_html=True
)
# --- END LOGIC CSS ---

# --- KHỞI TẠO STATE (GIỮ NGUYÊN) ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_cols' not in st.session_state:
    st.session_state.X_cols = None

np.random.seed(0)
st.title("HỆ THỐNG PHÂN TÍCH TÍN DỤNG DOANH NGHIỆP")
st.caption("🔎 Trạng thái Gemini: " + ("✅ sẵn sàng (cần 'GEMINI_API_KEY' trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện google-genai."))

menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

# =======================================================
# KHỐI 1: MỤC TIÊU CỦA MÔ HÌNH (GIỮ NGUYÊN)
# =======================================================
if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.markdown("**Dự báo xác suất vỡ nợ (PD) của khách hàng doanh nghiệp** dựa trên bộ chỉ số X1–X14.")
    for img in ["hinh2.jpg", "LogReg_1.png", "hinh3.png"]:
        try:
            st.image(img)
        except Exception:
            st.warning(f"Không tìm thấy {img}")

# =======================================================
# KHỐI 2: XÂY DỰNG MÔ HÌNH (Đã thêm logic lưu X_cols)
# =======================================================
elif choice == 'Xây dựng mô hình':
    st.subheader("1. Huấn luyện Mô hình PD và Phân tích Dữ liệu")
    st.markdown("**(Dự báo xác suất vỡ nợ của khách hàng_PD)**")

    st.write("##### A. Tải dữ liệu huấn luyện")
    
    df_default = None
    try:
        df_default = pd.read_csv('DATASET.csv', encoding='latin-1')
    except Exception:
        pass 
        
    uploaded_file = st.file_uploader("Tải CSV dữ liệu huấn luyện", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding='latin-1')
    elif st.session_state.df is None and df_default is not None:
        st.session_state.df = df_default 
        
    df = st.session_state.df 

    if df is None:
        st.info("Hãy tải file CSV huấn luyện (có cột 'default' và X_1...X_14) để tiếp tục.")
        st.stop()

    required_cols = ['default'] + [f"X_{i}" for i in range(1, 15)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Thiếu cột: {missing}")
        st.stop()
    
    st.write("##### B. Huấn luyện Mô hình (Logistic Regression)")
    
    X = df.drop(columns=['default'])
    y = df['default'].astype(int)
    
    # *** ĐIỂM SỬA CHỮA QUAN TRỌNG: LƯU TRỮ CHÍNH XÁC TÊN VÀ THỨ TỰ CỘT ĐÃ TRAIN ***
    st.session_state.X_cols = X.columns.tolist() 
    
    with st.spinner('Đang huấn luyện mô hình...'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(X_train, y_train)
        st.session_state.model = model
    
    st.success("Huấn luyện mô hình thành công! Mô hình đã sẵn sàng cho mục 'Sử dụng mô hình để dự báo'.")

    # (Phần hiển thị metrics, visualization giữ nguyên)
    y_pred_in = model.predict(X_train)
    y_proba_in = model.predict_proba(X_train)[:, 1]
    y_pred_out = model.predict(X_test)
    y_proba_out = model.predict_proba(X_test)[:, 1]

    metrics_in = { "accuracy_in": accuracy_score(y_train, y_pred_in), "precision_in": precision_score(y_train, y_pred_in, zero_division=0), "recall_in": recall_score(y_train, y_pred_in, zero_division=0), "f1_in": f1_score(y_train, y_pred_in, zero_division=0), "auc_in": roc_auc_score(y_train, y_proba_in), }
    metrics_out = { "accuracy_out": accuracy_score(y_test, y_pred_out), "precision_out": precision_score(y_test, y_pred_out, zero_division=0), "recall_out": recall_score(y_test, y_pred_out, zero_division=0), "f1_out": f1_score(y_test, y_pred_out, zero_division=0), "auc_out": roc_auc_score(y_test, y_proba_out), }

    st.write("##### C. Phân tích Dữ liệu")
    st.dataframe(df.head(3))
    st.write(df[[f"X_{i}" for i in range(1, 15)]].describe())

    st.write("##### D. Trực quan hóa dữ liệu")
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
            ax.plot(x_range, y_curve, color=BRIGHT_BORDEAUX, linewidth=2)
            ax.set_ylabel('Xác suất default')
            ax.set_xlabel(col)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")
    else:
        st.warning("Biến không tồn tại trong dữ liệu.")

    st.write("##### E. Kết quả đánh giá mô hình")
    dt = pd.DataFrame([metrics_in | metrics_out])
    st.dataframe(dt)

    st.write("##### F. Ma trận nhầm lẫn (Test set)")
    cm = confusion_matrix(y_test, y_pred_out)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Default', 'Default'])
    fig2, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Reds')
    st.pyplot(fig2)
    plt.close()

# =======================================================
# KHỐI 3: SỬ DỤNG MÔ HÌNH ĐỂ DỰ BÁO (Đã thêm logic sắp xếp cột dự báo)
# =======================================================
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("2. Phân tích Hồ sơ Khách hàng (Sử dụng Model & AI)")
    st.caption("File phải có đủ 3 sheet: **CDKT ; BCTN ; LCTT**")
    
    model = st.session_state.model
    X_cols = st.session_state.X_cols # Lấy thứ tự cột đã train

    if model is None or X_cols is None:
        st.error("⚠️ Vui lòng huấn luyện mô hình ở mục **'Xây dựng mô hình'** trước khi thực hiện dự báo.")
        st.stop()

    up_xlsx = st.file_uploader("Tải **ho_so_dn.xlsx** (3 sheet: CDKT, BCTN, LCTT)", type=["xlsx"], key="ho_so_dn")
    
    if up_xlsx is not None:
        try:
            ratios_df = compute_ratios_from_three_sheets(up_xlsx)
        except Exception as e:
            st.error(f"Lỗi tính X1…X14: {e}")
            st.stop()

        st.markdown("### 2.1. Kết quả tính X1…X14")
        st.dataframe(ratios_df.style.format("{:.4f}"))
        
        # --- ĐIỂM SỬA CHỮA QUAN TRỌNG: XỬ LÝ KHỚP CẤU TRÚC DỮ LIỆU ---
        
        # 1. Kiểm tra tập hợp cột
        if set(X_cols) != set(ratios_df.columns):
            st.error("❌ LỖI: Tập hợp các chỉ số tài chính (X1-X14) của file mới KHÔNG KHỚP với mô hình đã huấn luyện.")
            st.error(f"Cột trong Mô hình: {sorted(X_cols)}")
            st.error(f"Cột trong File mới: {sorted(ratios_df.columns.tolist())}")
            st.warning("Vui lòng kiểm tra lại cấu trúc file XLSX hoặc file CSV huấn luyện.")
            st.stop()

        # 2. Sắp xếp lại thứ tự cột của DataFrame dự báo cho khớp với Model
        ratios_df_aligned = ratios_df[X_cols]
        # -----------------------------------------------------------------
        
        # Tạo payload data cho AI
        data_for_ai = ratios_df.iloc[0].to_dict()

        # Dự báo PD
        with st.expander("2.2. Xác suất vỡ nợ dự báo (PD)"):
            try:
                # Sử dụng ratios_df_aligned đã được sắp xếp
                probs = model.predict_proba(ratios_df_aligned)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                show = ratios_df.copy()
                show["PD"] = probs
                show["Dự báo"] = np.where(preds == 1, "Vỡ nợ (Default)", "Không vỡ nợ (Non-Default)")
                
                st.dataframe(show.style.format({"PD": "{:.3f}"}))
                
                # Thêm PD vào payload cho AI
                data_for_ai['PD_Probability'] = probs[0]
                data_for_ai['PD_Prediction'] = "Default (Vỡ nợ)" if preds[0] == 1 else "Non-Default (Không vỡ nợ)"
            except Exception as e:
                st.warning(f"Không dự báo được PD: {e}. Lỗi xảy ra khi tính toán dự báo.")
            
        # Gemini Phân tích & khuyến nghị
        st.markdown("### 2.3. Phân tích AI & Đề xuất Tín dụng")

        if st.button("Yêu cầu AI Phân tích"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa **'GEMINI_API_KEY'** trong Streamlit Secrets.")

    else:
        st.info("Hãy tải **ho_so_dn.xlsx** (đủ 3 sheet) để tính X1…X14, dự báo PD và phân tích AI.")
