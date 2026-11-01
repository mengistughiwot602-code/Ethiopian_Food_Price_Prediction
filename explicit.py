import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon="🇪🇹",
    layout="wide",
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background-color: #f4f6f9;
        }

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f766e;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ccfbf1;
            padding-bottom: 0.4rem;
        }

        .stSelectbox label, .stDateInput label, .stSlider label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            border-radius: 8px;
            border-color: #cbd5e1;
        }

        .stButton button {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            border: none;
            width: 100%;
            box-shadow: 0 4px 6px rgba(13, 148, 136, 0.2);
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background: linear-gradient(135deg, #115e59 0%, #0f766e 100%);
            color: #ffffff;
            box-shadow: 0 6px 12px rgba(13, 148, 136, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MODEL FEATURES
# These must remain consistent with the features used during model training.
# =============================================================================

CATEGORICAL_FEATURES = [
    "admin1", "admin2", "market", "category",
    "commodity", "unit", "priceflag", "pricetype",
]

NUMERIC_FEATURES = [
    "latitude", "longitude", "year", "month", "month_sin", "month_cos",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMNS = ["price", "usdprice"]

REQUIRED_DATA_COLUMNS = [
    "date",
    "admin1",
