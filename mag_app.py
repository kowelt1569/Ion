import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MagnetModel Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# SAFE NUMBER PARSING (CRITICAL FIX)
# ══════════════════════════════════════════════════════════
def safe_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "."),
        errors="coerce"
    )

# ══════════════════════════════════════════════════════════
# NORMALIZATION (STABILITY FIX)
# ══════════════════════════════════════════════════════════
def _normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

# ══════════════════════════════════════════════════════════
# DESIGN MATRIX
# ══════════════════════════════════════════════════════════
def _design_matrix(x1, x2):
    x1 = _normalize(x1)
    x2 = _normalize(x2)

    return np.column_stack([
        x1 * (x2**2),    # F1
        x1 * x2,         # F2
        x2**2,           # F3
        x1,              # F4
        x2,              # F5
        np.ones_like(x1) # F6
    ])

# ══════════════════════════════════════════════════════════
# RIDGE REGRESSION
# ══════════════════════════════════════════════════════════
def ridge_regression(A, y, alpha=1.0):
    I = np.eye(A.shape[1])
    return np.linalg.solve(A.T @ A + alpha * I, A.T @ y)

# ══════════════════════════════════════════════════════════
# MODEL EVALUATION
# ══════════════════════════════════════════════════════════
def evaluate_model(x1, x2, y_true, coeffs):
    A = _design_matrix(x1, x2)
    y_pred = A @ coeffs

    residuals = y_true - y_pred
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(np.mean(residuals**2))

    return y_pred, residuals, r2, rmse

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
if "train_df" not in st.session_state:
    st.session_state.train_df = pd.DataFrame(columns=["X1","X2","Y"])

if "pred_df" not in st.session_state:
    st.session_state.pred_df = pd.DataFrame(columns=["X1","X2","Y_predicted"])

if "coeffs" not in st.session_state:
    st.session_state.coeffs = None

if "metrics" not in st.session_state:
    st.session_state.metrics = {"R2":"—","RMSE":"—"}

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:

    st.title("🌿 MagnetModel Pro")

    x1 = st.number_input("X₁", value=0.0)
    x2 = st.number_input("X₂", value=0.0)
    y  = st.number_input("Y", value=0.0)

    if st.button("➕ Додати"):
        st.session_state.train_df = pd.concat([
            st.session_state.train_df,
            pd.DataFrame([[x1,x2,y]], columns=["X1","X2","Y"])
        ])
        st.rerun()

    uploaded = st.file_uploader("Файл CSV/XLSX", type=["csv","xlsx"])

    if uploaded is not None:
        if uploaded.name.endswith("csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        df = df.iloc[:, :3]
        df.columns = ["X1","X2","Y"]

        # 🔥 FIX: safe conversion
        df["X1"] = safe_numeric(df["X1"])
        df["X2"] = safe_numeric(df["X2"])
        df["Y"]  = safe_numeric(df["Y"])

        df = df.dropna()

        st.session_state.train_df = pd.concat([st.session_state.train_df, df])
        st.rerun()

    if st.button("⚡ Навчити модель"):

        df = st.session_state.train_df.copy()

        # 🔥 FIX: safe conversion for manual data too
        df["X1"] = safe_numeric(df["X1"])
        df["X2"] = safe_numeric(df["X2"])
        df["Y"]  = safe_numeric(df["Y"])

        df = df.dropna()

        if len(df) < 6:
            st.warning("Потрібно ≥6 точок")
        elif df["X1"].std() == 0 or df["X2"].std() == 0:
            st.error("❌ X1 або X2 константні — модель неможлива")
        else:
            x1 = df["X1"].to_numpy(float)
            x2 = df["X2"].to_numpy(float)
            y  = df["Y"].to_numpy(float)

            A = _design_matrix(x1, x2)
            c = ridge_regression(A, y, alpha=1.0)

            yp, res, r2, rmse = evaluate_model(x1, x2, y, c)

            st.session_state.coeffs = c
            st.session_state.metrics = {
                "R2": f"{r2:.4f}",
                "RMSE": f"{rmse:.4f}"
            }

            st.success("Модель навчена")

# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

col1.metric("R²", st.session_state.metrics["R2"])
col2.metric("RMSE", st.session_state.metrics["RMSE"])
col3.metric("Train points", len(st.session_state.train_df))
col4.metric("Predictions", len(st.session_state.pred_df))

# ══════════════════════════════════════════════════════════
# COEFFICIENTS
# ══════════════════════════════════════════════════════════
st.subheader("Коефіцієнти")

if st.session_state.coeffs is not None:
    c = st.session_state.coeffs
    st.code(f"F1={c[0]:.4f}  F2={c[1]:.4f}  F3={c[2]:.4f}  F4={c[3]:.4f}  F5={c[4]:.4f}  F6={c[5]:.4f}")
else:
    st.info("Модель не навчена")

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Дані", "Графіки"])

with tab1:
    st.dataframe(st.session_state.train_df)

with tab2:
    if st.session_state.coeffs is None:
        st.info("Навчіть модель")
    else:
        df = st.session_state.train_df.copy()

        df["X1"] = safe_numeric(df["X1"])
        df["X2"] = safe_numeric(df["X2"])
        df["Y"]  = safe_numeric(df["Y"])
        df = df.dropna()

        x1 = df["X1"].to_numpy(float)
        x2 = df["X2"].to_numpy(float)
        y  = df["Y"].to_numpy(float)

        yp, res, r2, rmse = evaluate_model(x1, x2, y, st.session_state.coeffs)

        fig, ax = plt.subplots()
        ax.scatter(y, yp)
        ax.plot([min(y), max(y)], [min(y), max(y)], "--")

        st.pyplot(fig)
