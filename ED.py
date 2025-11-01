# app.py — Streamlit PD + Phân tích Gemini
(CẬP NHẬT THƯ VIỆN)

# =========================
# THƯ VIỆN BẮT BUỘC VÀ BỔ SUNG
# (Cần đảm bảo các gói này được cài đặt, ví
# dụ trong requirements.txt)
# =========================
from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# Thư viện Machine Learning và Mô hình
from sklearn.model_selection import
train_test_split
from sklearn.linear_model import
LogisticRegression
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	accuracy_score,
	recall_score,
	precision_score,
	roc_auc_score,
	ConfusionMatrixDisplay,
)
# Các thư viện BỔ SUNG theo yêu cầu (nếu được
# sử dụng trong code sau này)
# import xgboost as xgb
# import graphviz
# import statsmodels.api as sm

# =========================
# THÊM THƯ VIỆN GOOGLE GEMINI VÀ OPENAI
# (CHO TƯƠNG THÍCH VỚI REQ CŨ)
# =========================
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


MODEL_NAME = "gemini-2.5-flash" #
# Model mạnh mẽ và hiệu quả cho phân tích văn bản

# =========================
# HÀM GỌI GEMINI API
# =========================

def get_ai_analysis(data_payload: dict,
api_key: str) -> str:
	"""
	 Sử
	 Sử dụng Gemini API để phân tích chỉ số tài chính.
	"""
	if not _GEMINI_OK:
		# Đã sửa lỗi: Dùng nháy đơn bên ngoài để bao chuỗi có nháy đơn bên trong
		return 'Lỗi: Thiếu thư viện google-genai (cần cài đặt: pip install google-genai).'

	client = genai.Client(api_key=api_key)

	sys_prompt = (
		"Bạn là chuyên gia phân tích tín dụng doanh nghiệp tại ngân hàng.
"
		"Phân tích toàn diện dựa trên 14 chỉ số tài chính (X1..X14). "
		"Nêu rõ: (1) Khả năng sinh lời, (2) Thanh khoản, (3) Cơ cấu nợ, (4)
Hiệu quả hoạt động. "
		"Kết thúc bằng khuyến nghị in hoa: CHO VAY hoặc KHÔNG CHO VAY, kèm
2–3 điều kiện nếu CHO VAY. "
		"Viết bằng tiếng Việt súc tích, chuyên nghiệp."
	 )
	 
	user_prompt = "Bộ chỉ số X1..X14 cần phân tích:\n" +
str(data_payload) + "\n\nHãy phân tích và đưa ra khuyến nghị."

	try:
		response = client.models.generate_content(
			model=MODEL_NAME,
			contents=[
				 {"role":
"user", "parts": [{"text": sys_prompt +
"\n\n" + user_prompt}]}
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

# Alias các dòng quan trọng trong từng
# sheet
ALIAS_IS = {
	"doanh_thu_thuan": ["Doanh thu thuần", "Doanh
thu bán hàng", "Doanh thu thuần về bán hàng và cung cấp dịch vụ"],
	"gia_von": ["Giá vốn hàng bán"],
	"loi_nhuan_gop": ["Lợi nhuận gộp"],
	"chi_phi_lai_vay": ["Chi phí lãi vay", "Chi phí
tài chính (trong đó: chi phí lãi vay)"],
	"loi_nhuan_truoc_thue": ["Tổng lợi nhuận kế toán trước
thuế", "Lợi nhuận trước thuế", "Lợi nhuận trước thuế thu nhập
DN"],
}
ALIAS_BS = {
	"tong_tai_san": ["Tổng tài sản"],
	"von_chu_so_huu": ["Vốn chủ sở hữu", "Vốn
CSH"],
	"no_phai_tra": ["Nợ phải trả"],
	"tai_san_ngan_han": ["Tài sản ngắn hạn"],
	"no_ngan_han": ["Nợ ngắn hạn"],
	"hang_ton_kho": ["Hàng tồn kho"],
	"tien_tdt": ["Tiền và các khoản tương đương tiền",
"Tiền và tương đương tiền"],
	"phai_thu_kh": ["Phải thu ngắn hạn của khách hàng",
"Phải thu khách hàng"],
	"no_dai_han_den_han": ["Nợ dài hạn đến hạn trả",
"Nợ dài hạn đến hạn"],
}
ALIAS_CF = {
	"khau_hao": ["Khấu hao TSCĐ", "Khấu hao",
"Chi phí khấu hao"],
}

def _pick_year_cols(df: pd.DataFrame):
	"""Chọn 2 cột năm gần nhất từ sheet (ưu tiên cột có nhãn
là năm)."""
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

def _get_row_vals(df: pd.DataFrame,
aliases: list[str]):
	"""Tìm dòng theo alias (contains, không phân biệt hoa/thường).
Trả về (prev, cur) theo 2 cột năm gần nhất."""
	label_col = df.columns[0]
	prev_col, cur_col = _pick_year_cols(df)
	mask = False
	for alias in aliases:
		mask = mask | df[label_col].astype(str).str.contains(alias, case=False,
na=False)
	rows = df[mask]
	if rows.empty:
		return np.nan, np.nan
	row = rows.iloc[0]

	def to_num(x):
		try:
			return float(str(x).replace(",", "").replace("
", ""))
		except Exception:
			return np.nan

	return to_num(row[prev_col]), to_num(row[cur_col])

def
compute_ratios_from_three_sheets(xlsx_file) -> pd.DataFrame:
	"""Đọc 3 sheet CDKT/BCTN/LCTT và tính X1..X14 theo yêu cầu."""
	# Đọc 3 sheet; cần openpyxl trong requirements
	bs = pd.read_excel(xlsx_file, sheet_name="CDKT",
engine="openpyxl")
	is_ = pd.read_excel(xlsx_file, sheet_name="BCTN",
engine="openpyxl")
	cf = pd.read_excel(xlsx_file, sheet_name="LCTT",
engine="openpyxl")

	# ---- KQKD (BCTN)
	DTT_prev, DTT_cur	=
_get_row_vals(is_, ALIAS_IS["doanh_thu_thuan"])
	GVHB_prev, GVHB_cur = _get_row_vals(is_, ALIAS_IS["gia_von"])
	LNG_prev, LNG_cur	=
_get_row_vals(is_, ALIAS_IS["loi_nhuan_gop"])
	LNTT_prev, LNTT_cur = _get_row_vals(is_,
ALIAS_IS["loi_nhuan_truoc_thue"])
	LV_prev, LV_cur		=
_get_row_vals(is_, ALIAS_IS["chi_phi_lai_vay"])

	# ---- CĐKT (CDKT)
	TTS_prev, TTS_cur		=
_get_row_vals(bs, ALIAS_BS["tong_tai_san"])
	VCSH_prev, VCSH_cur	=
_get_row_vals(bs, ALIAS_BS["von_chu_so_huu"])
	NPT_prev, NPT_cur		=
_get_row_vals(bs, ALIAS_BS["no_phai_tra"])
	TSNH_prev, TSNH_cur	=
_get_row_vals(bs, ALIAS_BS["tai_san_ngan_han"])
	NNH_prev, NNH_cur		=
_get_row_vals(bs, ALIAS_BS["no_ngan_han"])
	HTK_prev, HTK_cur		=
_get_row_vals(bs, ALIAS_BS["hang_ton_kho"])
	Tien_prev, Tien_cur	=
_get_row_vals(bs, ALIAS_BS["tien_tdt"])
	KPT_prev, KPT_cur		=
_get_row_vals(bs, ALIAS_BS["phai_thu_kh"])
	NDH_prev, NDH_cur		=
_get_row_vals(bs, ALIAS_BS["no_dai_han_den_han"])

	# ---- LCTT (LCTT) – lấy Khấu hao nếu có
	KH_prev, KH_cur = _get_row_vals(cf, ALIAS_CF["khau_hao"])

	# Chuẩn hoá số âm thường thấy ở GVHB, chi phí lãi vay, khấu hao
	if pd.notna(GVHB_cur): GVHB_cur = abs(GVHB_cur)
	if pd.notna(LV_cur):	LV_cur	= abs(LV_cur)
	if pd.notna(KH_cur):	KH_cur	= abs(KH_cur)

	# Trung bình đầu/cuối kỳ
	def avg(a, b):
		if pd.isna(a) and pd.isna(b): return np.nan
		if pd.isna(a): return b
		if pd.isna(b): return a
		return (a + b) / 2.0
	TTS_avg	 = avg(TTS_cur,	 TTS_prev)
	VCSH_avg = avg(VCSH_cur, VCSH_prev)
	HTK_avg	 = avg(HTK_cur,	 HTK_prev)
	KPT_avg	 = avg(KPT_cur,	 KPT_prev)

	# EBIT ~ LNTT + chi phí lãi vay (nếu thiếu EBIT riêng)
	EBIT_cur = (LNTT_cur + LV_cur) if (pd.notna(LNTT_cur) and
pd.notna(LV_cur)) else np.nan
	# Nợ dài hạn đến hạn trả: có file không ghi -> set 0
	NDH_cur = 0.0 if pd.isna(NDH_cur) else NDH_cur

	def div(a, b):
		return np.nan if (b is None or pd.isna(b) or b == 0) else a / b

	# ==== TÍNH X1..X14 ====
	X1	= div(LNG_cur, DTT_cur)					# Biên LN gộp
	X2	= div(LNTT_cur, DTT_cur)				# Biên LNTT
	X3	= div(LNTT_cur, TTS_avg)				# ROA (trước thuế)
	X4	= div(LNTT_cur, VCSH_avg)				# ROE (trước thuế)
	X5	= div(NPT_cur,	TTS_cur)				# Nợ/Tài sản
	X6	= div(NPT_cur,	VCSH_cur)				# Nợ/VCSH
	X7	= div(TSNH_cur, NNH_cur)				# Thanh toán hiện hành
	X8	= div((TSNH_cur - HTK_cur) if
pd.notna(TSNH_cur) and pd.notna(HTK_cur) else np.nan, NNH_cur)	# Nhanh
	X9	= div(EBIT_cur, LV_cur)					# Khả năng trả lãi
	X10 = div((EBIT_cur + (KH_cur if pd.notna(KH_cur) else 0.0)),
				 (LV_cur + NDH_cur) if
pd.notna(LV_cur) else np.nan)	# Khả năng
# trả nợ gốc
	X11 = div(Tien_cur, VCSH_cur)					# Tiền/VCSH
	X12 = div(GVHB_cur, HTK_avg)					# Vòng quay HTK
	turnover = div(DTT_cur, KPT_avg)				# Vòng quay phải thu
	X13 = div(365.0, turnover) if pd.notna(turnover) and turnover != 0 else
np.nan	# Kỳ thu tiền BQ
	X14 = div(DTT_cur, TTS_avg)					# Hiệu suất sử dụng tài sản

	ratios = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11,
X12, X13, X14]],
								 columns=[f"X_{i}" for i in range(1, 15)])
	return ratios

# =========================
# UI & TRAIN MODEL
# =========================
np.random.seed(0)
st.title("DỰ BÁO THAM SỐ PD")
st.write("## Dự báo xác suất vỡ nợ của
khách hàng_PD")

# Hiển thị trạng thái thư viện AI
st.caption("🔎
Trạng thái Gemini: " + ("✅ sẵn sàng (cần 'GEMINI_API_KEY'
trong Secrets)" if _GEMINI_OK else "⚠️ Thiếu thư viện
google-genai."))

# Load dữ liệu huấn luyện (CSV
