import streamlit as st
import pandas as pd
import numpy as np

# --- Cấu hình Trang (Page Configuration) ---
# Thiết lập cấu hình cho toàn bộ ứng dụng, bao gồm tiêu đề trang (hiển thị trên tab trình duyệt) 
# và layout "wide" để mở rộng giao diện
st.set_page_config(
    page_title="Hệ Thống Đánh Giá Rủi Ro Tín Dụng",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tiêu Đề Chính (Header) ---
# Tiêu đề chính hiển thị ở đầu trang, đáp ứng yêu cầu của người dùng
st.title("HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG")
st.markdown("""
    <style>
    /* Điều chỉnh font và căn giữa cho tiêu đề chính */
    .stApp {
        background-color: #f0f2f6; /* Màu nền nhẹ */
    }
    .stTitle {
        font-family: 'Arial Black', Gadget, sans-serif;
        color: #1f77b4; /* Màu xanh đậm */
        text-align: center;
        padding-top: 10px;
        padding-bottom: 20px;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 0px !important; 
    }
    /* Điều chỉnh style cho các tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px; /* Khoảng cách giữa các tab */
        justify-content: center; /* Căn giữa các tab */
        margin-top: 20px; /* Thêm khoảng cách phía trên tab */
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 250px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #333333;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4; /* Màu nền khi tab được chọn */
        color: #ffffff; /* Màu chữ khi tab được chọn */
        border-bottom: 3px solid #ff7f0e; /* Đường viền nổi bật */
    }
    </style>
""", unsafe_allow_html=True)


# --- Hàm Mockup (Placeholder Functions) ---
# Các hàm này mô phỏng chức năng dự báo và xử lý dữ liệu thực tế

def predict_credit_risk(data):
    """Mô phỏng chức năng dự báo rủi ro tín dụng."""
    # Giả lập logic dự báo: Nếu tổng thu nhập và tài sản cao, rủi ro thấp
    score = data['monthly_income'] * 0.4 + data['assets_value'] * 0.6 - data['loan_amount'] * 0.5
    
    if score > 5000:
        return "RỦI RO THẤP (Low Risk)", "#2ca02c" # Xanh lá
    elif score > 2000:
        return "RỦI RO TRUNG BÌNH (Medium Risk)", "#ff7f0e" # Cam
    else:
        return "RỦI RO CAO (High Risk)", "#d62728" # Đỏ

def display_model_objective():
    """Hiển thị mục tiêu và lợi ích của mô hình."""
    st.header("Mục Tiêu Chính Của Mô Hình")
    st.markdown("""
    Mục tiêu cốt lõi của **Mô hình Đánh giá Rủi ro Tín dụng** này là tối ưu hóa quá trình ra quyết định cho vay, đảm bảo sự cân bằng giữa tăng trưởng kinh doanh và kiểm soát rủi ro.

    * **Tối đa hóa Lợi nhuận:** Phân loại chính xác khách hàng rủi ro thấp để phê duyệt khoản vay nhanh chóng và hiệu quả.
    * **Giảm thiểu Thiệt hại:** Xác định khách hàng rủi ro cao để áp dụng các biện pháp phòng ngừa hoặc từ chối khoản vay.
    * **Tuân thủ Quy định:** Đảm bảo quá trình đánh giá công bằng, minh bạch và tuân thủ các quy định tài chính hiện hành.
    """)
    st.subheader("Lợi ích mang lại")
    st.markdown("""
    1.  **Quyết định Tự động hóa:** Giảm thời gian xử lý hồ sơ từ vài ngày xuống còn vài phút.
    2.  **Tính nhất quán:** Đảm bảo mọi hồ sơ đều được đánh giá theo cùng một tiêu chuẩn khách quan.
    3.  **Hỗ trợ Chiến lược:** Cung cấp thông tin chi tiết về các yếu tố rủi ro chính để cải tiến chính sách cho vay.
    """)

def display_model_construction():
    """Hiển thị thông tin về việc xây dựng mô hình."""
    st.header("Quy Trình Xây Dựng và Huấn Luyện Mô Hình")
    
    st.subheader("1. Chuẩn bị Dữ liệu (Data Preparation)")
    st.markdown("""
    * **Nguồn Dữ liệu:** Sử dụng dữ liệu lịch sử về các khoản vay (đã thanh toán/quá hạn), thông tin nhân khẩu học và tài chính của khách hàng.
    * **Làm sạch và Kỹ thuật Đặc trưng (Feature Engineering):** Xử lý các giá trị thiếu, chuẩn hóa dữ liệu và tạo ra các biến mới có ý nghĩa (ví dụ: Tỷ lệ nợ trên thu nhập).
    """)
    
    st.subheader("2. Lựa chọn Mô hình (Model Selection)")
    st.markdown("""
    * **Thuật toán:** Thường sử dụng các mô hình học máy như **Logistic Regression**, **Random Forest**, hoặc **Gradient Boosting (XGBoost/LightGBM)** vì khả năng giải thích và hiệu suất cao.
    * **Phân chia Dữ liệu:** Dữ liệu được chia thành tập huấn luyện (Training Set), tập kiểm định (Validation Set), và tập kiểm tra (Test Set).
    """)
    
    st.subheader("3. Đánh giá và Triển khai (Evaluation and Deployment)")
    st.markdown("""
    * **Chỉ số Đánh giá:** Các chỉ số chính bao gồm AUC-ROC, F1-Score, và Accuracy. Đặc biệt chú trọng vào khả năng phân loại Rủi ro Cao (Recall).
    * **Triển khai:** Mô hình được đóng gói (ví dụ: sử dụng Pickle hoặc ONNX) và tích hợp vào ứng dụng web (Streamlit) để sử dụng trong thực tế.
    """)
    
    # Mô phỏng biểu đồ hiệu suất mô hình
    chart_data = pd.DataFrame(
        np.random.rand(20, 3),
        columns=['Độ chính xác', 'Độ nhạy', 'Độ đặc hiệu']
    )
    st.line_chart(chart_data)
    st.caption("Biểu đồ giả lập các chỉ số hiệu suất mô hình qua các phiên bản.")


# --- Tạo Tabs (Horizontal Tabs) ---
# Tạo 3 tab nằm ngang theo yêu cầu của người dùng
tab_predict, tab_objective, tab_construction = st.tabs([
    "SỬ DỤNG MÔ HÌNH ĐỂ DỰ BÁO", 
    "MỤC TIÊU CỦA MÔ HÌNH", 
    "XÂY DỰNG MÔ HÌNH"
])

# --- Tab 1: SỬ DỤNG MÔ HÌNH ĐỂ DỰ BÁO (Model Prediction) ---
with tab_predict:
    st.header("Công Cụ Dự Báo Rủi Ro Tín Dụng")
    st.write("Vui lòng nhập các thông tin sau để nhận kết quả đánh giá rủi ro:")

    # Sử dụng st.columns để tạo bố cục nhập liệu hai cột đẹp mắt
    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input("Số tiền vay (VNĐ)", min_value=1000000, max_value=5000000000, value=50000000, step=5000000)
        age = st.slider("Tuổi", min_value=18, max_value=65, value=30)
        num_dependents = st.selectbox("Số người phụ thuộc", options=[0, 1, 2, 3, 4, 5])
        
    with col2:
        monthly_income = st.number_input("Thu nhập hàng tháng (VNĐ)", min_value=1000000, max_value=500000000, value=15000000, step=1000000)
        assets_value = st.number_input("Tổng giá trị tài sản (VNĐ)", min_value=0, max_value=10000000000, value=500000000, step=50000000)
        credit_history = st.selectbox("Lịch sử tín dụng", options=["Tốt (Đã thanh toán đầy đủ)", "Trung bình (Có nợ quá hạn nhỏ)", "Kém (Từng vỡ nợ)"])

    input_data = {
        'loan_amount': loan_amount,
        'age': age,
        'monthly_income': monthly_income,
        'assets_value': assets_value,
        # Các trường khác được sử dụng trong hàm predict_credit_risk sẽ cần được map/chuyển đổi nếu cần
    }

    # Nút thực hiện dự báo
    if st.button("ĐÁNH GIÁ RỦI RO", key='predict_button', type='primary'):
        # Gọi hàm dự báo mockup
        risk_level, color = predict_credit_risk(input_data)
        
        st.subheader("KẾT QUẢ ĐÁNH GIÁ")
        
        # Hiển thị kết quả bằng Markdown với CSS nội tuyến để làm nổi bật
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;">
            MỨC ĐỘ RỦI RO: {risk_level}
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Lưu ý: Đây là đánh giá rủi ro dựa trên mô hình. Cần thêm kiểm tra và xác minh hồ sơ.")


# --- Tab 2: MỤC TIÊU CỦA MÔ HÌNH (Model Objective) ---
with tab_objective:
    display_model_objective()

# --- Tab 3: XÂY DỰNG MÔ HÌNH (Model Construction) ---
with tab_construction:
    display_model_construction()

# --- Thanh Bên (Sidebar) cho các tùy chọn phụ ---
st.sidebar.title("THÔNG TIN BỔ SUNG")
st.sidebar.info("Ứng dụng được xây dựng bằng Streamlit và mô hình học máy giả định để đánh giá rủi ro tín dụng.")
st.sidebar.caption("Phiên bản v1.0 - 2025")

# Kết thúc file Streamlit
