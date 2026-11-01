import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

ethiopian_flag_icon = Image.new("RGB", (96, 64), "white")
draw = ImageDraw.Draw(ethiopian_flag_icon)
draw.rectangle((0, 0, 95, 21), fill="#078930")
draw.rectangle((0, 21, 95, 42), fill="#FCDD09")
draw.rectangle((0, 42, 95, 63), fill="#DA121A")
draw.ellipse((35, 10, 61, 54), fill="#0F47AF")
star_points = [
    (48, 16), (51, 27), (62, 27), (53, 34),
    (57, 45), (48, 38), (39, 45), (43, 34),
    (34, 27), (45, 27),
]
draw.polygon(star_points, fill="#FCDD09")

st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon=ethiopian_flag_icon,
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
    "date", "admin1", "admin2", "market", "latitude", "longitude",
    "category", "commodity", "unit", "priceflag", "pricetype",
    "price", "usdprice",
]

# =============================================================================
# MODEL FEATURE PREPARATION
# =============================================================================

def prepare_model_features(data):
    required = CATEGORICAL_FEATURES + ["latitude", "longitude", "date"]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Data is missing required model features: {missing}")

    features = data[required].copy()
    dates = pd.to_datetime(features["date"], errors="coerce")

    features["year"] = dates.dt.year
    features["month"] = dates.dt.month
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    return features[MODEL_FEATURES]

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model_assets():
    try:
        bundle = joblib.load("random_forest_model.joblib")
        required_keys = {"models", "metrics", "target_columns"}
        if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
            raise ValueError("Saved artifact is not the expected pipeline bundle.")
        return bundle, None
    except Exception as error:
        return None, str(error)

# =============================================================================
# BUILT-IN SAMPLE DATASET (Ensures app works instantly without local files)
# =============================================================================

def get_sample_dataset():
    data = {
        "date": pd.date_range(start="2024-01-01", periods=30, freq="ME"),
        "admin1": ["Amhara", "Oromia", "Addis Ababa", "Tigray", "SNNPR"] * 6,
        "admin2": ["Bahir Dar Zuria", "Adama", "Addis Ababa", "Mekelle", "Hawassa"] * 6,
        "market": ["Bahir Dar", "Adama Market", "Merkato", "Mekelle Market", "Hawassa Market"] * 6,
        "latitude": [11.5742, 8.5417, 9.0300, 13.4967, 7.0504] * 6,
        "longitude": [37.3614, 39.2683, 38.7400, 39.4753, 38.4714] * 6,
        "category": ["cereals and tubers", "cereals and tubers", "pulses and nuts", "oil and fats", "vegetables and fruits"] * 6,
        "commodity": ["Maize (white)", "Teff (mixed)", "Beans (white)", "Oil (vegetable)", "Potato"] * 6,
        "unit": ["KG", "KG", "KG", "L", "KG"] * 6,
        "priceflag": ["actual", "actual", "actual", "actual", "actual"] * 6,
        "pricetype": ["Retail", "Retail", "Retail", "Retail", "Retail"] * 6,
        "price": np.random.uniform(45.0, 150.0, 30),
        "usdprice": np.random.uniform(0.5, 2.5, 30),
    }
    return pd.DataFrame(data)

# =============================================================================
# DATASET LOADER
# =============================================================================

def process_dataframe(df):
    metadata_mask = df.astype(str).apply(
        lambda column: column.str.strip().str.startswith("#")
    ).any(axis=1)
    df = df.loc[~metadata_mask].copy()

    df.columns = df.columns.astype(str).str.strip().str.lower()
    df.reset_index(drop=True, inplace=True)

    missing_columns = [
        column for column in REQUIRED_DATA_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["price", "usdprice", "latitude", "longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df

@st.cache_data
def load_dataset(uploaded_file=None):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return process_dataframe(df)

        elif os.path.exists("wfp_food_prices_et.xlsx"):
            df = pd.read_excel("wfp_food_prices_et.xlsx")
            return process_dataframe(df)

        else:
            excel_files = [
                file for file in os.listdir(".") if file.lower().endswith(".xlsx")
            ]
            if excel_files:
                df = pd.read_excel(excel_files[0])
                return process_dataframe(df)
            else:
                return get_sample_dataset()

    except Exception as error:
        st.error(f"Could not load dataset: {error}")
        return get_sample_dataset()

# =============================================================================
# DROPDOWN AND LOCATION HELPERS
# =============================================================================

def get_options(dataframe, column, filters=None):
    if column not in dataframe.columns:
        return []

    subset = dataframe
    if filters:
        for filter_column, selected_value in filters.items():
            if filter_column not in subset.columns:
                return []
            subset = subset[subset[filter_column].astype(str) == str(selected_value)]

    values = subset[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return sorted(values.unique().tolist(), key=str.casefold)

def reset_prediction_children(*keys):
    for key in keys:
        st.session_state.pop(key, None)

def price_status_label(value):
    labels = {
        "actual": "Actual",
        "actual,aggregate": "Actual + Aggregate",
        "aggregate": "Aggregate",
        "forecast": "Forecast",
    }
    return labels.get(str(value).strip().lower(), str(value))

def price_type_label(value):
    return str(value)

def get_market_coordinates(dataframe, region, zone, market):
    market_rows = dataframe[
        (dataframe["admin1"].astype(str) == str(region))
        & (dataframe["admin2"].astype(str) == str(zone))
        & (dataframe["market"].astype(str) == str(market))
    ].copy()

    market_rows["latitude"] = pd.to_numeric(market_rows["latitude"], errors="coerce")
    market_rows["longitude"] = pd.to_numeric(market_rows["longitude"], errors="coerce")
    coordinates = market_rows[["latitude", "longitude"]].dropna()

    if coordinates.empty:
        return None, None

    return float(coordinates["latitude"].mean()), float(coordinates["longitude"].mean())

# Load default dataset (or fallback sample data) on startup
df_data = load_dataset()

# =============================================================================
# APPLICATION HEADER
# =============================================================================

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:16px; margin: 0 0 8px 0;">
      <svg width="50" height="27" viewBox="0 0 50 27" role="img"
           aria-label="Ethiopian flag" xmlns="http://www.w3.org/2000/svg"
           style="flex:0 0 auto; border-radius:6px; overflow:hidden;">
        <defs><clipPath id="ethiopiaFlagClip"><rect width="50" height="27" rx="5"/></clipPath></defs>
        <g clip-path="url(#ethiopiaFlagClip)">
          <rect width="50" height="9" y="0" fill="#078930"/>
          <rect width="50" height="9" y="9" fill="#FCDD09"/>
          <rect width="50" height="9" y="18" fill="#DA121A"/>
          <circle cx="25" cy="13.5" r="8" fill="#0F47AF"/>
          <path d="M25 6.8 L26.5 11.2 L31.1 11.2 L27.4 13.9 L28.8 18.3 L25 15.6 L21.2 18.3 L22.6 13.9 L18.9 11.2 L23.5 11.2 Z"
                fill="#FCDD09"/>
        </g>
      </svg>
      <h1 style="margin:0; padding:0; font-size:2rem; line-height:1.2;
                 font-weight:750; color:inherit;">
        Ethiopian Food Price Prediction &amp; Market Analysis
      </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "An interactive decision-support application built for localized market "
    "forecasting and price intelligence."
)

# =============================================================================
# NAVIGATION MENU
# =============================================================================

st.sidebar.markdown("### Navigation Menu")

menu = st.sidebar.radio(
    "Go to",
    [
        "📊 1. Data Overview & Market Signals",
        "📈 2. Exploratory Analytics & Trends",
        "🎯 3. Interactive Prediction Engine",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# =============================================================================
# PAGE 1: DATA OVERVIEW & MARKET SIGNALS
# =============================================================================

if menu == "📊 1. Data Overview & Market Signals":

    st.header("📊 Key Dataset Insights & Aggregations Overview")
    st.write(
        "Review core dataset aggregations, regional data health, "
        "and run batch target predictions."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
