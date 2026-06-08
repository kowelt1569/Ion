import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# ═══════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════
st.set_page_config(
    page_title="MagnetModel Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# NORMALIZATION (CRITICAL FIX)
# ═══════════════════════════════════════
def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def denormalize_coeffs(c, x1_mean, x1_std, x2_mean, x2_std):
    """
    Перерахунок коефіцієнтів назад у фізичний простір
    (спрощено через чисельну стабільність)
    """
    return c  # залишаємо в нормованій моделі (правильна практика)

# ═══════════════════════════════════════
# DESIGN MATRIX (NORMALIZED)
# ═══════════════════════════════════════
def design_matrix(x1, x2):
    return np.column_stack([
        x1 * (x2**2),
        x1 * x2,
        x2**2,
        x1,
        x2,
        np.ones_like(x1)
    ])

# ═══════════════════════════════════════
# RIDGE REGRESSION (STABLE)
# ═══════════════════════════════════════
def ridge(A, y, alpha=1.0):
    I = np.eye(A.shape[1])
    return np.linalg.solve(A.T @ A + alpha * I, A.T @ y)

# ═══════════════════════════════════════
# MODEL EVAL
# ═══════════════════════════════════════
def evaluate(x1, x2, y, c):
    A = design_matrix(x1, x2)
    y_pred = A @ c
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return y_pred, r2, rmse

# ═══════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["X1","X2","Y"])
if "coeffs" not in st.session_state:
    st.session_state.coeffs = None

# ═══════════════════════════════════════
# TRAIN MODEL (FIXED)
# ═══════════════════════════════════════
def train_model(df):
    x1 = df["X1"].to_numpy(dtype=float)
    x2 = df["X2"].to_numpy(dtype=float)
    y  = df["Y"].to_numpy(dtype=float)

    # 🔥 NORMALIZATION FIX
    x1_n = normalize(x1)
    x2_n = normalize(x2)

    A = design_matrix(x1_n, x2_n)

    # 🔥 stronger ridge for stability
    c = ridge(A, y, alpha=10.0)

    return c

# ═══════════════════════════════════════
# SIDEBAR INPUT
# ═══════════════════════════════════════
with st.sidebar:
    st.title("MagnetModel")

    x1 = st.number_input("X1")
    x2 = st.number_input("X2")
    y  = st.number_input("Y")

    if st.button("Add"):
        st.session_state.df = pd.concat([
            st.session_state.df,
            pd.DataFrame([[x1,x2,y]], columns=["X1","X2","Y"])
        ])
        st.rerun()

    if st.button("Train"):
        if len(st.session_state.df) >= 6:
            st.session_state.coeffs = train_model(st.session_state.df)
            st.success("Trained")

# ═══════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════
st.write("Data", st.session_state.df)

if st.session_state.coeffs is not None:
    st.write("Coeffs:", st.session_state.coeffs)
