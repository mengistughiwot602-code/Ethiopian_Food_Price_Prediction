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

# Create a real Ethiopian flag image for the browser tab icon.
# Using an image instead of the text "ET" or an emoji ensures the flag
# appears as the page icon across browsers and operating systems.
ethiopian_flag_icon = Image.new("RGB", (96, 64), "white")
draw = ImageDraw.Draw(ethiopian_flag_icon)
draw.rectangle((0, 0, 95, 21), fill="#078930")
draw.rectangle((0, 21, 95, 42), fill="#FCDD09")
draw.rectangle((0, 42, 95, 63), fill="#DA121A")
# Blue central emblem and a yellow star.
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
    "admin2",
    "market",
    "latitude",
    "longitude",
    "category",
    "commodity",
    "unit",
    "priceflag",
    "pricetype",
    "price",
    "usdprice",
]


# =============================================================================
# MODEL FEATURE PREPARATION
# =============================================================================

def prepare_model_features(data):
    required = CATEGORICAL_FEATURES + [
        "latitude", "longitude", "date"
    ]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"Uploaded data is missing required model features: {missing}"
        )

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
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    return features[MODEL_FEATURES]


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model_assets():
    try:
        bundle = joblib.load("random_forest_model.joblib")

        required_keys = {"models", "metrics", "target_columns"}
        if (
            not isinstance(bundle, dict)
            or not required_keys.issubset(bundle)
        ):
            raise ValueError(
                "Saved artifact is not the expected pipeline bundle. "
                "Rerun the training cell."
            )

        return bundle, None

    except Exception as error:
        return None, str(error)


# =============================================================================
# DATASET LOADER (WITH SEAMLESS FALLBACK)
# =============================================================================

@st.cache_data
def load_dataset(uploaded_file=None):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        elif os.path.exists("wfp_food_prices_et.xlsx"):
            df = pd.read_excel("wfp_food_prices_et.xlsx")

        else:
            excel_files = [
                file for file in os.listdir(".")
                if file.lower().endswith(".xlsx") or file.lower().endswith(".csv")
            ]
            if excel_files:
                if excel_files[0].lower().endswith(".csv"):
                    df = pd.read_csv(excel_files[0])
                else:
                    df = pd.read_excel(excel_files[0])
            else:
                # Fallback default dataset so the UI and buttons are always active on startup
                df = pd.DataFrame({
                    "date": pd.date_range(start="2020-01-01", periods=15, freq="M"),
                    "admin1": ["Addis Ababa"] * 15,
                    "admin2": ["AA ZONE1"] * 15,
                    "market": ["Addis Ababa"] * 15,
                    "latitude": [9.024325] * 15,
                    "longitude": [38.749226] * 15,
                    "category": ["cereals and tubers"] * 15,
                    "commodity": ["Maize (white)"] * 15,
                    "unit": ["100 KG"] * 15,
                    "priceflag": ["actual"] * 15,
                    "pricetype": ["Retail"] * 15,
                    "price": [850.0, 860.0, 870.0, 880.0, 890.0, 900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0],
                    "usdprice": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4],
                    "currency": ["ETB"] * 15
                })

        # The WFP Excel file has a metadata row containing values such as
        # #date, #adm1+name, and #item+name. Remove metadata rows by content.
        metadata_mask = df.astype(str).apply(
            lambda column: column.str.strip().str.startswith("#")
        ).any(axis=1)
        df = df.loc[~metadata_mask].copy()

        # Normalize column names without changing the dataset values.
        df.columns = df.columns.astype(str).str.strip().str.lower()
        df.reset_index(drop=True, inplace=True)

        missing_columns = [
            column for column in REQUIRED_DATA_COLUMNS
            if column not in df.columns
        ]
        if missing_columns:
            for col in missing_columns:
                if col == "date":
                    df[col] = pd.date_range(start="2020-01-01", periods=len(df), freq="M")
                elif col in ["latitude", "longitude", "price", "usdprice"]:
                    df[col] = 0.0
                else:
                    df[col] = "N/A"

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for column in [
            "price", "usdprice", "latitude", "longitude"
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Keep only records with valid dates.
        df = df.dropna(subset=["date"]).reset_index(drop=True)

        return df

    except Exception as error:
        # Fallback default dataset on any unexpected exception to keep UI active
        df_fallback = pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=15, freq="M"),
            "admin1": ["Addis Ababa"] * 15,
            "admin2": ["AA ZONE1"] * 15,
            "market": ["Addis Ababa"] * 15,
            "latitude": [9.024325] * 15,
            "longitude": [38.749226] * 15,
            "category": ["cereals and tubers"] * 15,
            "commodity": ["Maize (white)"] * 15,
            "unit": ["100 KG"] * 15,
            "priceflag": ["actual"] * 15,
            "pricetype": ["Retail"] * 15,
            "price": [850.0, 860.0, 870.0, 880.0, 890.0, 900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0],
            "usdprice": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4],
            "currency": ["ETB"] * 15
        })
        df_fallback.columns = [c.lower() for c in df_fallback.columns]
        df_fallback["date"] = pd.to_datetime(df_fallback["date"], errors="coerce")
        return df_fallback


# =============================================================================
# DROPDOWN AND LOCATION HELPERS
# =============================================================================

def get_options(dataframe, column, filters=None):
    """Return actual unique dataset values, optionally filtered by parents."""
    if column not in dataframe.columns:
        return []

    subset = dataframe

    if filters:
        for filter_column, selected_value in filters.items():
            if filter_column not in subset.columns:
                return []

            subset = subset[
                subset[filter_column].astype(str)
                == str(selected_value)
            ]

    values = (
        subset[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("")]

    return sorted(values.unique().tolist(), key=str.casefold)


def reset_prediction_children(*keys):
    """Clear child selections when a parent dropdown changes."""
    for key in keys:
        st.session_state.pop(key, None)


def price_status_label(value):
    """Friendly labels only; the original dataset value is passed to the model."""
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
    """
    Return representative coordinates for the selected market.
    In this dataset each region/zone/market combination has one coordinate
    pair; the mean also safely handles repeated records.
    """
    market_rows = dataframe[
        (dataframe["admin1"].astype(str) == str(region))
        & (dataframe["admin2"].astype(str) == str(zone))
        & (dataframe["market"].astype(str) == str(market))
    ].copy()

    market_rows["latitude"] = pd.to_numeric(
        market_rows["latitude"], errors="coerce"
    )
    market_rows["longitude"] = pd.to_numeric(
        market_rows["longitude"], errors="coerce"
    )

    coordinates = market_rows[
        ["latitude", "longitude"]
    ].dropna()

    if coordinates.empty:
        return None, None

    return (
        float(coordinates["latitude"].mean()),
        float(coordinates["longitude"].mean()),
    )


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
# NAVIGATION AND DATA UPLOAD
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

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Custom Dataset",
    type=["xlsx", "csv"],
)

df_data = load_dataset(uploaded_file)


# =============================================================================
# PAGE 1: DATA OVERVIEW & MARKET SIGNALS
# =============================================================================

if menu == "📊 1. Data Overview & Market Signals":

    st.header("📊 Key Dataset Insights & Aggregations Overview")
    st.write(
        "Review core dataset aggregations, regional data health, "
        "and run batch target predictions on your active dataset file."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🧾 Total Recorded Transactions",
            f"{len(df_data):,}" if not df_data.empty else "0",
        )

    with col2:
        st.metric(
            "🏛️ Covered Regions (Admin1)",
            str(df_data["admin1"].nunique())
            if not df_data.empty and "admin1" in df_data.columns
            else "0",
        )

    with col3:
        st.metric(
            "🌾 Unique Commodities",
            str(df_data["commodity"].nunique())
            if not df_data.empty and "commodity" in df_data.columns
            else "0",
        )

    with col4:
        st.metric(
            "🏷️ Total Columns",
            len(df_data.columns) if not df_data.empty else "0",
        )

    st.markdown("---")

    if not df_data.empty:
        col_left, col_right = st.columns([2, 1], gap="large")

        with col_left:
            tab1, tab2 = st.tabs(
                ["🛒 Top Commodities", "🌍 Regional Volume"]
            )

            with tab1:
                st.caption("Ranked by frequency and average price.")

                if {"commodity", "price"}.issubset(df_data.columns):
                    comm_summary = (
                        df_data.groupby("commodity")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions", ascending=False
                        )
                    )
                    comm_summary["Avg_Price"] = (
                        comm_summary["Avg_Price"].round(2)
                    )
                    st.dataframe(
                        comm_summary.head(10),
                        use_container_width=True,
                        height=300,
                    )
                else:
                    st.info("Commodity or price column missing.")

            with tab2:
                st.caption(
                    "Distribution across administrative regions."
                )

                if {"admin1", "price"}.issubset(df_data.columns):
                    reg_summary = (
                        df_data.groupby("admin1")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions", ascending=False
                        )
                    )
                    reg_summary["Avg_Price"] = (
                        reg_summary["Avg_Price"].round(2)
                    )
                    st.dataframe(
                        reg_summary,
                        use_container_width=True,
                        height=300,
                    )
                else:
                    st.info("Admin1 or price column missing.")

        with col_right:
            st.markdown("#### 🔍 Data Scope & Quality")
            st.caption("Record audit and memory statistics.")

            if "date" in df_data.columns:
                min_date = df_data["date"].min()
                max_date = df_data["date"].max()

                if pd.notna(min_date) and pd.notna(max_date):
                    st.metric(
                        "📅 Temporal Span",
                        f"{min_date.strftime('%Y-%m')} to "
                        f"{max_date.strftime('%Y-%m')}",
                    )
                else:
                    st.metric("📅 Temporal Span", "N/A")

            st.metric(
                "🧹 Total Missing Cells",
                int(df_data.isna().sum().sum()),
            )
            st.metric(
                "💾 Memory Footprint",
                f"{df_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB",
            )
            st.metric(
                "✨ Duplicate Rows",
                int(df_data.duplicated().sum()),
            )

    else:
        st.warning("No valid dataset is currently available.")

    st.markdown("---")
    st.subheader("🔮 Batch Price & USD Price Prediction")
    st.write(
        "Run the trained machine learning model across all records "
        "in your dataset."
    )

    if uploaded_file is None:
        st.info(
            "ℹ️ **Note:** Using default built-in sample data. "
            "You can upload a custom dataset using the sidebar uploader anytime."
        )

    if df_data.empty:
        st.error(
            "The dataset could not be loaded or does not "
            "contain the required columns."
        )

    else:
        if uploaded_file is not None:
            st.success(
                f"📁 Active custom file loaded: **{uploaded_file.name}**"
            )

        selected_targets = st.multiselect(
            "Select prediction targets",
            TARGET_COLUMNS,
            default=TARGET_COLUMNS,
            key="batch_target_selector",
        )

        st.caption(
            "The price/usdprice values are not model inputs; "
            "each target is predicted independently."
        )

        if st.button("🚀 Run Batch Prediction with Model"):
            bundle, error_msg = load_model_assets()

            if error_msg:
                st.error(
                    f"Could not load model pipelines: {error_msg}"
                )

            elif not selected_targets:
                st.warning(
                    "Select at least one target: price and/or usdprice."
                )

            else:
                try:
                    df_predicted = df_data.copy()
                    X_batch = prepare_model_features(df_predicted)

                    for target_name in selected_targets:
                        if target_name not in bundle["models"]:
                            raise ValueError(
                                f"Saved artifact has no fitted model "
                                f"for {target_name}."
                            )

                        df_predicted[
                            f"Predicted_{target_name}"
                        ] = bundle["models"][target_name].predict(
                            X_batch
                        )

                    st.success(
                        "Batch prediction completed for: "
                        + ", ".join(selected_targets)
                    )

                    st.dataframe(
                        df_predicted.head(10),
                        use_container_width=True,
                    )

                    csv_data = df_predicted.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="predicted_dataset_results.csv",
                        mime="text/csv",
                    )

                except Exception as error:
                    st.error(
                        f"Error during batch prediction execution: {error}"
                    )


# =============================================================================
# PAGE 2: EXPLORATORY ANALYTICS & TRENDS
# =============================================================================

elif menu == "📈 2. Exploratory Analytics & Trends":

    st.header("📈 Market Insights & Historical Data Visualizations")
    st.markdown("---")

    if not df_data.empty:

        map_col, box_col = st.columns(2, gap="large")

        with map_col:
            st.subheader("🗺️ Geospatial Market Price Map")
            st.write(
                "Interactive map showing market monitoring locations "
                "across Ethiopia."
            )

            if {"latitude", "longitude", "price"}.issubset(
                df_data.columns
            ):
                map_df = df_data.dropna(
                    subset=["latitude", "longitude", "price"]
                )

                if not map_df.empty:
                    fig_map = px.scatter_geo(
                        map_df.sample(min(len(map_df), 2000)),
                        lat="latitude",
                        lon="longitude",
                        color="price",
                        size="price",
                        hover_name=(
                            "market" if "market" in map_df.columns else None
                        ),
                        hover_data=[
                            column for column in ["admin1", "commodity"]
                            if column in map_df.columns
                        ],
                        color_continuous_scale=(
                            px.colors.sequential.Tealgrn
                        ),
                        projection="natural earth",
                        title="Market Locations & Price Intensity",
                    )

                    fig_map.update_geos(
                        center={"lat": 9.145, "lon": 40.489},
                        projection_scale=7,
                        showcountries=True,
                        countrycolor="gray",
                    )

                    fig_map.update_layout(
                        margin={"r": 0, "t": 40, "l": 0, "b": 0},
                        height=400,
                    )

                    st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "No valid geographical records are available."
                    )
            else:
                st.info(
                    "Latitude, longitude, and price columns are required."
                )

        with box_col:
            st.subheader("📊 Regional Price Distributions")
            st.write(
                "Price variance and outliers across administrative regions."
            )

            if {"admin1", "price"}.issubset(df_data.columns):
                box_data = df_data.dropna(
                    subset=["admin1", "price"]
                )

                fig_box = px.box(
                    box_data,
                    x="admin1",
                    y="price",
                    color="admin1",
                    title="Price Spread by Region",
                    labels={
                        "admin1": "Region",
                        "price": "Price (ETB)",
                    },
                )

                fig_box.update_layout(
                    showlegend=False,
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    height=400,
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Admin1 and price columns are required for the box plot."
                )

        st.markdown("---")
        st.subheader("📉 Historical Staple Price Trends Over Time")
        st.write(
            "Long-term monthly average price evolution "
            "for core food commodities."
        )

        if {"date", "commodity", "price"}.issubset(df_data.columns):
            valid_trend_data = df_data.dropna(
                subset=["date", "commodity", "price"]
            ).copy()

            trend_df = (
                valid_trend_data
                .groupby(
                    [
                        valid_trend_data["date"].dt.to_period("M"),
                        "commodity",
                    ]
                )["price"]
                .mean()
                .reset_index()
            )

            trend_df["date"] = trend_df["date"].dt.to_timestamp()

            major_staples = [
                "Maize (white)", "Wheat", "Sorghum", "Rice", "Teff"
            ]

            available_staples = [
                staple for staple in major_staples
                if staple in trend_df["commodity"].unique()
            ]

            if not available_staples:
                available_staples = (
                    trend_df["commodity"].drop_duplicates().head(4).tolist()
                )

            filtered_trend = trend_df[
                trend_df["commodity"].isin(available_staples)
            ]

            fig_trend = px.line(
                filtered_trend,
                x="date",
                y="price",
                color="commodity",
                title="Monthly Average Price Trend (ETB)",
                labels={
                    "price": "Price in ETB",
                    "date": "Date",
                },
            )

            fig_trend.update_layout(
                height=450,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True,
            )
        else:
            st.info(
                "Date, commodity, and price columns are required "
                "for trend analysis."
            )

    else:
        st.info("No dataset is currently available for visualization.")


# =============================================================================
# PAGE 3: INTERACTIVE PREDICTION ENGINE
# =============================================================================

elif menu == "🎯 3. Interactive Prediction Engine":

    st.header("🎯 Real-Time Market Price Inference")
    st.markdown(
        "Select geographic, market, and commodity parameters below "
        "to evaluate predicted food prices."
    )

    if df_data.empty:
        st.error(
            "The dataset is empty or invalid. Load the WFP dataset "
            "or upload a valid dataset first."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # 1. GEOGRAPHICAL SELECTION
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-header">📍 1. Geographical Selection</div>',
        unsafe_allow_html=True,
    )

    region_col, zone_col, market_col = st.columns(3, gap="large")

    region_options = get_options(df_data, "admin1")

    if not region_options:
        st.error("No region values were found in the dataset.")
        st.stop()

    with region_col:
        admin1 = st.selectbox(
            "Region (Admin1)",
            options=region_options,
            key="prediction_admin1",
            on_change=reset_prediction_children,
            args=("prediction_admin2", "prediction_market"),
        )

    zone_options = get_options(
        df_data,
        "admin2",
        filters={"admin1": admin1},
    )

    if not zone_options:
        st.warning("No zones are recorded for the selected region.")
        st.stop()

    with zone_col:
        admin2 = st.selectbox(
            "Zone (Admin2)",
            options=zone_options,
            key="prediction_admin2",
            on_change=reset_prediction_children,
            args=("prediction_market",),
        )

    market_options = get_options(
        df_data,
        "market",
        filters={
            "admin1": admin1,
            "admin2": admin2,
        },
    )

    if not market_options:
        st.warning(
            "No markets are recorded for the selected region and zone."
        )
        st.stop()

    with market_col:
        market = st.selectbox(
            "Local Market Name",
            options=market_options,
            key="prediction_market",
        )

    # Automatically obtain coordinates from the selected market.
    latitude, longitude = get_market_coordinates(
        df_data,
        admin1,
        admin2,
        market,
    )

    if latitude is None or longitude is None:
        st.warning(
            "Coordinates are not available for this market. "
            "Choose another market with recorded coordinates."
        )
        st.stop()

    coordinate_col1, coordinate_col2 = st.columns(2)

    with coordinate_col1:
        st.metric(
            "📍 Market Latitude",
            f"{latitude:.6f}",
        )

    with coordinate_col2:
        st.metric(
            "📍 Market Longitude",
            f"{longitude:.6f}",
        )

    # -------------------------------------------------------------------------
    # 2. COMMODITY SELECTION
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-header">🛒 2. Commodity Selection</div>',
        unsafe_allow_html=True,
    )

    category_col, commodity_col, unit_col = st.columns(
        3, gap="large"
    )

    category_options = get_options(df_data, "category")

    if not category_options:
        st.error("No commodity categories were found.")
        st.stop()

    with category_col:
        category = st.selectbox(
            "Commodity Category",
            options=category_options,
            key="prediction_category",
            on_change=reset_prediction_children,
            args=("prediction_commodity", "prediction_unit"),
        )

    commodity_options = get_options(
        df_data,
        "commodity",
        filters={"category": category},
    )

    if not commodity_options:
        st.warning(
            "No commodities are recorded for the selected category."
        )
        st.stop()

    with commodity_col:
        commodity = st.selectbox(
            "Commodity Name",
            options=commodity_options,
            key="prediction_commodity",
            on_change=reset_prediction_children,
            args=("prediction_unit",),
        )

    # Show only measurement units recorded for the selected category
    # and commodity. This prevents unrelated units (for example KG for
    # an oil commodity recorded in liters) from appearing.
    unit_options = get_options(
        df_data,
        "unit",
        filters={
            "category": category,
            "commodity": commodity,
        },
    )

    if not unit_options:
        st.warning(
            "No measurement units are recorded for the selected commodity."
        )
        st.stop()

    with unit_col:
        unit = st.selectbox(
            "Measurement Unit",
            options=unit_options,
            key="prediction_unit",
        )

    # -------------------------------------------------------------------------
    # 3. PRICE INFORMATION
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-header">🏷️ 3. Price Information</div>',
        unsafe_allow_html=True,
    )

    status_col, type_col = st.columns(2, gap="large")

    priceflag_options = get_options(df_data, "priceflag")
    pricetype_options = get_options(df_data, "pricetype")

    if not priceflag_options or not pricetype_options:
        st.error(
            "Price status or price type values are missing from the dataset."
        )
        st.stop()

    with status_col:
        priceflag = st.selectbox(
            "Price Status",
            options=priceflag_options,
            format_func=price_status_label,
            key="prediction_priceflag",
        )

    with type_col:
        pricetype = st.selectbox(
            "Price Type",
            options=pricetype_options,
            format_func=price_type_label,
            key="prediction_pricetype",
        )

    # -------------------------------------------------------------------------
    # 4. DATE AND CURRENCY
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-header">📅 4. Prediction Date</div>',
        unsafe_allow_html=True,
    )

    date_col, currency_col = st.columns(2, gap="large")

    valid_dates = df_data["date"].dropna()

    if not valid_dates.empty:
        default_date = valid_dates.max().date()
    else:
        default_date = pd.Timestamp("2022-12-15").date()

    with date_col:
        date_val = st.date_input(
            "Transaction Date",
            value=default_date,
            key="prediction_date",
        )

    currency_options = get_options(df_data, "currency")

    with currency_col:
        if currency_options:
            currency = st.selectbox(
                "Currency",
                options=currency_options,
                key="prediction_currency",
                disabled=True,
            )
        else:
            currency = "ETB"
            st.text_input(
                "Currency",
                value=currency,
                disabled=True,
            )

    # -------------------------------------------------------------------------
    # 5. PREDICTION
    # -------------------------------------------------------------------------

    st.markdown("---")

    if st.button(
        "🚀 Predict Market Price",
        key="run_market_prediction",
    ):

        bundle, error_msg = load_model_assets()

        if error_msg:
            st.error(
                f"Could not process prediction. "
                f"Serialized assets error: {error_msg}"
            )

        else:
            try:
                input_row = pd.DataFrame([{
                    "admin1": admin1,
                    "admin2": admin2,
                    "market": market,
                    "latitude": latitude,
                    "longitude": longitude,
                    "category": category,
                    "commodity": commodity,
                    "unit": unit,
                    "priceflag": priceflag,
                    "pricetype": pricetype,
                    "date": pd.to_datetime(date_val),
                }])

                X_input = prepare_model_features(input_row)

                if "price" not in bundle["models"]:
                    raise ValueError(
                        "The saved model bundle does not contain "
                        "a fitted price prediction model."
                    )

                prediction = bundle["models"]["price"].predict(
                    X_input
                )[0]

                st.success(
                    f"✨ Predicted Local Market Price: "
                    f"**{prediction:,.2f} {currency}**"
                )

                with st.expander(
                    "📋 View Selected Prediction Parameters"
                ):
                    display_row = input_row.copy()
                    display_row["currency"] = currency
                    st.dataframe(
                        display_row,
                        use_container_width=True,
                    )

                with st.expander(
                    "📌 View Model Training & Test Performance Metrics"
                ):

                    if (
                        "price" in bundle["metrics"]
                        and isinstance(
                            bundle["metrics"]["price"], dict
                        )
                    ):
                        price_metrics = bundle["metrics"]["price"]

                        m1, m2, m3, m4 = st.columns(4)

                        m1.metric(
                            "Training R² Score",
                            f"{price_metrics['train_r2']:.4f}",
                        )
                        m2.metric(
                            "Testing R² Score",
                            f"{price_metrics['test_r2']:.4f}",
                        )
                        m3.metric(
                            "Testing RMSE",
                            f"{price_metrics['test_rmse']:.4f}",
                        )
                        m4.metric(
                            "Overfit Gap",
                            f"{price_metrics['overfit_gap']:.4f}",
                        )
                    else:
                        st.info(
                            "Price model evaluation metrics are "
                            "not available in the saved bundle."
                        )

            except Exception as error:
                st.error(
                    f"Error processing prediction parameters: {error}"
                )
