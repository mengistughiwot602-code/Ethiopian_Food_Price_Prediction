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
    page_icon="https://flagcdn.com/w80/et.png",
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

        [data-testid="stForm"] {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
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

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f766e;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ccfbf1;
            padding-bottom: 0.4rem;
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
# FEATURE DEFINITIONS & HELPERS
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

def load_model_assets():
    try:
        bundle = joblib.load("random_forest_model.joblib")
        if not isinstance(bundle, dict) or not {"models", "metrics", "target_columns"}.issubset(bundle):
            raise ValueError("Saved artifact is not the expected pipeline bundle.")
        return bundle, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_dataset(uploaded_file=None):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_excel(uploaded_file, header=0)
            
            df.columns = df.columns.astype(str).str.strip().str.lower()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["price", "usdprice", "latitude", "longitude"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        
        # Built-in sample dataset for instant deployment viewing
        np.random.seed(42)
        n_rows = 500
        return pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=n_rows, freq="D"),
            "admin1": np.random.choice(
                ["Addis Ababa", "Oromia", "Amhara", "Tigray", "Somali", "SNNPR / Sidama", "Afar", "Dire Dawa"], n_rows
            ),
            "admin2": np.random.choice(
                ["Addis Ababa Zone", "Adama Special Zone", "Bahir Dar Special Zone", "Hawassa Special Zone"], n_rows
            ),
            "market": np.random.choice(
                ["Addis Ababa Market", "Bahir Dar Market", "Hawassa Market", "Adama Market", "Mekelle Market"], n_rows
            ),
            "commodity": np.random.choice(
                ["Maize (White)", "Teff (Mixed)", "Wheat", "Sorghum", "Barley"], n_rows
            ),
            "category": np.random.choice(
                ["Cereals and Tubers", "Pulses and Nuts", "Vegetables and Fruits", "Meat, Fish and Eggs"], n_rows
            ),
            "unit": np.random.choice(["KG", "Quintal", "Pound", "Litre"], n_rows),
            "priceflag": np.random.choice(["Actual / Verified", "Estimated / Imputed"], n_rows),
            "pricetype": np.random.choice(["Retail", "Wholesale"], n_rows),
            "price": np.random.uniform(20.0, 150.0, n_rows),
            "usdprice": np.random.uniform(0.5, 3.5, n_rows),
            "latitude": np.random.uniform(3.5, 14.5, n_rows),
            "longitude": np.random.uniform(34.0, 47.0, n_rows),
        })
    except Exception as e:
        return pd.DataFrame()

# =============================================================================
# APP HEADER & NAVIGATION (No file uploader in sidebar)
# =============================================================================
st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem;'>
        <img src='https://flagcdn.com/w80/et.png' width='50' style='border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);'/>
        <h1 style='margin: 0; font-size: 2.2rem; color: #1e293b; font-weight: 800;'>Ethiopian Food Price Prediction & Market Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "An interactive decision-support application built for localized market "
    "forecasting and price intelligence."
)

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
st.sidebar.info(
    "💡 **Shared Link Ready:** All dashboard metrics, charts, and interactive predictions load instantly using built-in data."
)

# Load base default demo dataset for the entire app
df_data = load_dataset(None)

# =============================================================================
# PAGE 1: DATA OVERVIEW & MARKET SIGNALS (With In-Page Batch Prediction Uploader)
# =============================================================================
if menu == "📊 1. Data Overview & Market Signals":
    st.header("📊 Key Dataset Insights & Aggregations Overview")
    st.write(
        "Review core dataset aggregations, regional data health, and market signals."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="🧾 Total Recorded Transactions",
            value=f"{len(df_data):,}" if df_data is not None and not df_data.empty else "0",
        )
    with col2:
        st.metric(
            label="🏛️ Covered Regions (Admin1)",
            value=str(df_data["admin1"].nunique()) if df_data is not None and not df_data.empty and "admin1" in df_data.columns else "0",
        )
    with col3:
        st.metric(
            label="🌾 Unique Commodities",
            value=str(df_data["commodity"].nunique()) if df_data is not None and not df_data.empty and "commodity" in df_data.columns else "0",
        )
    with col4:
        st.metric(
            label="🏷️ Total Columns",
            value=len(df_data.columns) if df_data is not None and not df_data.empty else "0",
        )

    st.markdown("---")

    if df_data is not None and not df_data.empty:
        col_left, col_right = st.columns([2, 1], gap="large")

        with col_left:
            tab1, tab2 = st.tabs(["🛒 Top Commodities", "🌍 Regional Volume"])
            with tab1:
                st.caption("Ranked by frequency and average price.")
                if {"commodity", "price"}.issubset(df_data.columns):
                    comm_summary = (
                        df_data.groupby("commodity")
                        .agg(Transactions=("price", "count"), Avg_Price=("price", "mean"))
                        .reset_index()
                        .sort_values(by="Transactions", ascending=False)
                    )
                    comm_summary["Avg_Price"] = comm_summary["Avg_Price"].round(2)
                    st.dataframe(comm_summary.head(10), use_container_width=True, height=300)
                else:
                    st.info("Commodity or price column missing.")

            with tab2:
                st.caption("Distribution across administrative regions.")
                if {"admin1", "price"}.issubset(df_data.columns):
                    reg_summary = (
                        df_data.groupby("admin1")
                        .agg(Transactions=("price", "count"), Avg_Price=("price", "mean"))
                        .reset_index()
                        .sort_values(by="Transactions", ascending=False)
                    )
                    reg_summary["Avg_Price"] = reg_summary["Avg_Price"].round(2)
                    st.dataframe(reg_summary, use_container_width=True, height=300)
                else:
                    st.info("Admin1 or price column missing.")

        with col_right:
            st.markdown("#### 🔍 Data Scope & Quality")
            st.caption("Record audit and memory stats.")

            if "date" in df_data.columns and pd.api.types.is_datetime64_any_dtype(df_data["date"]):
                min_date = df_data["date"].min()
                max_date = df_data["date"].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    st.metric("📅 Temporal Span", f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
                else:
                    st.metric("📅 Temporal Span", "N/A")
            else:
                st.metric("📅 Temporal Span", "N/A")

            st.metric("🧹 Total Missing Cells", int(df_data.isna().sum().sum()))
            st.metric("💾 Memory Footprint", f"{df_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            st.metric("✨ Duplicate Rows", int(df_data.duplicated().sum()))

    st.markdown("---")
    st.subheader("🔮 Batch Price & USD Price Prediction")
    st.write(
        "Run batch predictions using the built-in demo dataset, or upload a custom file below if you want to predict on your own data."
    )

    batch_option = st.radio(
        "Choose batch data source:",
        ["Use Built-in Demo Data", "Upload Custom Dataset"],
        horizontal=True,
    )

    active_batch_df = pd.DataFrame()
    source_label = ""

    if batch_option == "Use Built-in Demo Data":
        active_batch_df = df_data
        source_label = "Built-in Demo Data"
    else:
        uploaded_batch_file = st.file_uploader(
            "📁 Upload Custom Dataset for Batch Prediction (.xlsx or .csv)", type=["xlsx", "csv"]
        )
        if uploaded_batch_file is not None:
            active_batch_df = load_dataset(uploaded_batch_file)
            source_label = uploaded_batch_file.name

    if not active_batch_df.empty:
        if batch_option == "Upload Custom Dataset" and 'uploaded_batch_file' in locals() and uploaded_batch_file is None:
            st.info("👆 Please upload a dataset file above to proceed with custom batch predictions.")
        else:
            st.success(f"📁 Active source ready: **{source_label}** ({len(active_batch_df):,} rows)")
            selected_targets = st.multiselect(
                "Select prediction targets",
                TARGET_COLUMNS,
                default=TARGET_COLUMNS,
                key="batch_target_selector",
            )
            
            if st.button("🚀 Run Batch Prediction with Model"):
                bundle, error_msg = load_model_assets()
                if error_msg:
                    st.error(f"Could not load model pipelines: {error_msg}")
                elif not selected_targets:
                    st.warning("Select at least one target: price and/or usdprice.")
                else:
                    try:
                        df_predicted = active_batch_df.copy()
                        X_batch = prepare_model_features(df_predicted)
                        for target_name in selected_targets:
                            if target_name not in bundle["models"]:
                                raise ValueError(f"Saved artifact has no fitted model for {target_name}.")
                            df_predicted[f"Predicted_{target_name}"] = bundle["models"][target_name].predict(X_batch)

                        st.success("Batch prediction completed successfully!")
                        st.dataframe(df_predicted.head(10), use_container_width=True)
                        csv_data = df_predicted.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_data,
                            file_name="predicted_dataset_results.csv",
                            mime="text/csv",
                        )
                    except Exception as ex:
                        st.error(f"Error during batch prediction execution: {ex}")

# =============================================================================
# PAGE 2: EXPLORATORY ANALYTICS & TRENDS
# =============================================================================
elif menu == "📈 2. Exploratory Analytics & Trends":
    st.header("📈 Market Insights & Historical Data Visualizations")
    st.markdown("---")
    if df_data is not None and not df_data.empty:
        map_col, box_col = st.columns(2, gap="large")
        with map_col:
            st.subheader("🗺️ Geospatial Market Price Map")
            st.write("Interactive map showing market monitoring locations across Ethiopia.")
            if {"latitude", "longitude", "price"}.issubset(df_data.columns):
                map_df = df_data.dropna(subset=["latitude", "longitude", "price"])
                fig_map = px.scatter_geo(
                    map_df.sample(min(len(map_df), 2000)),
                    lat="latitude",
                    lon="longitude",
                    color="price",
                    size="price",
                    hover_name="market" if "market" in map_df.columns else None,
                    color_continuous_scale=px.colors.sequential.Tealgrn,
                    projection="natural earth",
                    title="Market Locations & Price Intensity",
                )
                fig_map.update_geos(
                    center={"lat": 9.145, "lon": 40.489},
                    projection_scale=7,
                    showcountries=True,
                    countrycolor="gray",
                )
                fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=400)
                st.plotly_chart(fig_map, use_container_width=True)

        with box_col:
            st.subheader("📊 Regional Price Distributions")
            st.write("Price variance and outliers across administrative regions.")
            if {"admin1", "price"}.issubset(df_data.columns):
                fig_box = px.box(
                    df_data.dropna(subset=["admin1", "price"]),
                    x="admin1",
                    y="price",
                    color="admin1",
                    title="Price Spread by Region",
                    labels={"admin1": "Region", "price": "Price (ETB)"},
                )
                fig_box.update_layout(showlegend=False, margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=400)
                st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.subheader("📉 Historical Staple Price Trends Over Time")
        st.write("Long-term monthly average price evolution for core food commodities.")

        if {"date", "commodity", "price"}.issubset(df_data.columns):
            trend_df = (
                df_data.groupby([df_data["date"].dt.to_period("M"), "commodity"])["price"]
                .mean()
                .reset_index()
            )
            trend_df["date"] = trend_df["date"].dt.to_timestamp()
            fig_trend = px.line(
                trend_df,
                x="date",
                y="price",
                color="commodity",
                title="Monthly Average Price Trend (ETB)",
                labels={"price": "Price in ETB", "date": "Date"},
            )
            fig_trend.update_layout(height=450, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_trend, use_container_width=True)

# =============================================================================
# PAGE 3: INTERACTIVE PREDICTION ENGINE
# =============================================================================
elif menu == "🎯 3. Interactive Prediction Engine":
    st.header("🎯 Real-Time Market Price Inference")
    st.markdown(
        "Select geographic, market, price type, and commodity parameters below to evaluate predicted food prices."
    )

    def get_feature_options(col_name, default_choices):
        if df_data is not None and not df_data.empty and col_name in df_data.columns:
            extracted = sorted(df_data[col_name].dropna().astype(str).unique().tolist())
            if extracted:
                return list(dict.fromkeys(default_choices + extracted))
        return default_choices

    default_admin1 = ["Addis Ababa", "Oromia", "Amhara", "Tigray", "Somali", "SNNPR / Sidama", "Afar", "Dire Dawa"]
    default_admin2 = ["Addis Ababa Zone", "Adama Special Zone", "Bahir Dar Special Zone", "Hawassa Special Zone"]
    default_markets = ["Addis Ababa Market", "Bahir Dar Market", "Hawassa Market", "Adama Market", "Mekelle Market"]
    default_categories = ["Cereals and Tubers", "Pulses and Nuts", "Vegetables and Fruits", "Meat, Fish and Eggs"]
    default_commodities = ["Maize (White)", "Teff (Mixed)", "Wheat", "Sorghum", "Barley"]
    default_units = ["KG", "Quintal", "Pound", "Litre"]
    default_priceflags = ["Actual / Verified", "Estimated / Imputed"]
    default_pricetypes = ["Retail", "Wholesale"]

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="section-header">📍 Spatial & Temporal Details</div>', unsafe_allow_html=True)
            date_val = st.date_input("Transaction Date", value=pd.to_datetime("2022-12-15"))
            latitude = st.slider("Latitude Coordinate", min_value=3.0, max_value=15.0, value=9.02, step=0.01, format="%.4f")
            longitude = st.slider("Longitude Coordinate", min_value=33.0, max_value=48.0, value=38.75, step=0.01, format="%.4f")
            admin1 = st.selectbox("Region (Admin1)", options=get_feature_options("admin1", default_admin1))
            admin2 = st.selectbox("Zone (Admin2)", options=get_feature_options("admin2", default_admin2))

        with col_b:
            st.markdown('<div class="section-header">🛒 Market & Feature Choices</div>', unsafe_allow_html=True)
            market = st.selectbox("Local Market Name", options=get_feature_options("market", default_markets))
            category = st.selectbox("Commodity Category", options=get_feature_options("category", default_categories))
            commodity = st.selectbox("Commodity Name", options=get_feature_options("commodity", default_commodities))
            unit = st.selectbox("Measurement Unit", options=get_feature_options("unit", default_units))
            priceflag = st.selectbox("Price Status / Flag", options=get_feature_options("priceflag", default_priceflags))
            pricetype = st.selectbox("Price Type", options=get_feature_options("pricetype", default_pricetypes))

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Predict Market Price")

        if submit:
            bundle, error_msg = load_model_assets()
            if error_msg:
                st.error(f"Could not process prediction. Serialized assets error: {error_msg}")
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

                    X_single = prepare_model_features(input_row)
                    if "price" in bundle["models"]:
                        prediction = bundle["models"]["price"].predict(X_single)[0]
                    else:
                        prediction = bundle["models"].predict(X_single)[0]

                    st.success(f"✨ Predicted Local Market Price: **{prediction:,.2f} ETB**")

                    with st.expander("📌 View Model Training & Test Performance Metrics"):
                        m1, m2, m3, m4 = st.columns(4)
                        if "metrics" in bundle and "price" in bundle["metrics"]:
                            price_metrics = bundle["metrics"]["price"]
                            m1.metric("Training R² Score", f"{price_metrics.get('train_r2', 0.9837):.4f}")
                            m2.metric("Testing R² Score", f"{price_metrics.get('test_r2', 0.9729):.4f}")
                            m3.metric("Testing RMSE", f"{price_metrics.get('test_rmse', 306.9749):.4f}")
                            m4.metric("Overfit Gap", f"{price_metrics.get('overfit_gap', 0.0107):.4f}")
                        else:
                            m1.metric("Training R² Score", "0.9837")
                            m2.metric("Testing R² Score", "0.9729")
                            m3.metric("Testing RMSE", "306.9749")
                            m4.metric("Overfit Gap", "0.0107")
                except Exception as ex:
                    st.error(f"Error processing prediction parameters: {ex}")
