
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

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f766e;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ccfbf1;
            padding-bottom: 0.4rem;
        }

        .stSelectbox label,
        .stDateInput label,
        .stSlider label {
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
            background: linear-gradient(
                135deg, #0f766e 0%, #0d9488 100%
            );
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
            background: linear-gradient(
                135deg, #115e59 0%, #0f766e 100%
            );
            color: #ffffff;
            box-shadow: 0 6px 12px rgba(13, 148, 136, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MODEL FEATURE CONFIGURATION
# =============================================================================

CATEGORICAL_FEATURES = [
    "admin1",
    "admin2",
    "market",
    "category",
    "commodity",
    "unit",
    "priceflag",
    "pricetype",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "year",
    "month",
    "month_sin",
    "month_cos",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMNS = ["price", "usdprice"]


# =============================================================================
# MODEL FEATURE PREPARATION
# =============================================================================

def prepare_model_features(data):

    required = CATEGORICAL_FEATURES + [
        "latitude",
        "longitude",
        "date",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Uploaded data is missing required model features: {missing}"
        )

    features = data[required].copy()

    dates = pd.to_datetime(
        features["date"],
        errors="coerce"
    )

    features["year"] = dates.dt.year
    features["month"] = dates.dt.month

    features["month_sin"] = np.sin(
        2 * np.pi * features["month"] / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * features["month"] / 12
    )

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(
            lambda value: (
                str(value) if pd.notna(value) else np.nan
            )
        )

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    return features[MODEL_FEATURES]


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model_assets():

    try:

        bundle = joblib.load(
            "random_forest_model.joblib"
        )

        required_keys = {
            "models",
            "metrics",
            "target_columns",
        }

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
# DATASET LOADER
# =============================================================================

@st.cache_data
def load_dataset(uploaded_file=None):

    try:

        # Load uploaded dataset
        if uploaded_file is not None:

            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

        # Load the original local dataset
        elif os.path.exists("wfp_food_prices_et.xlsx"):

            df = pd.read_excel(
                "wfp_food_prices_et.xlsx"
            )

        else:

            excel_files = [
                file
                for file in os.listdir(".")
                if file.lower().endswith(".xlsx")
            ]

            if excel_files:

                df = pd.read_excel(excel_files[0])

            else:

                raise FileNotFoundError(
                    "No dataset file found locally."
                )

        # ---------------------------------------------------------
        # Remove metadata rows such as:
        # #date, #adm1+name, #item+name, etc.
        # ---------------------------------------------------------

        metadata_mask = df.astype(str).apply(
            lambda column: column.str.strip().str.startswith("#")
        ).any(axis=1)

        df = df.loc[~metadata_mask].copy()

        df.reset_index(drop=True, inplace=True)

        # ---------------------------------------------------------
        # Validate dataset structure
        # ---------------------------------------------------------

        required_columns = [
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

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            if uploaded_file is not None:

                return pd.DataFrame()

            raise ValueError(
                f"Dataset is missing columns: {missing_columns}"
            )

        # ---------------------------------------------------------
        # Convert date and numeric columns
        # ---------------------------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        numeric_columns = [
            "price",
            "usdprice",
            "latitude",
            "longitude",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Remove rows without usable dates
        df = df.dropna(
            subset=["date"]
        ).reset_index(drop=True)

        return df

    except Exception as error:

        if uploaded_file is not None:

            st.error(
                f"Could not load uploaded dataset: {error}"
            )

            return pd.DataFrame()

        # Synthetic fallback if no local dataset exists
        np.random.seed(42)

        n_rows = 500

        return pd.DataFrame({

            "date": pd.date_range(
                start="2020-01-01",
                periods=n_rows,
                freq="D"
            ),

            "admin1": np.random.choice(
                ["Addis Ababa", "Oromia", "Amhara", "Tigray"],
                n_rows
            ),

            "admin2": np.random.choice(
                ["AA ZONE1", "E. SHEWA"],
                n_rows
            ),

            "market": np.random.choice(
                ["Addis Ababa", "Adama"],
                n_rows
            ),

            "commodity": np.random.choice(
                ["Maize (white)", "Wheat", "Sorghum"],
                n_rows
            ),

            "category": [
                "cereals and tubers"
            ] * n_rows,

            "unit": [
                "KG"
            ] * n_rows,

            "priceflag": [
                "actual"
            ] * n_rows,

            "pricetype": [
                "Retail"
            ] * n_rows,

            "price": np.random.uniform(
                20.0, 150.0, n_rows
            ),

            "usdprice": np.random.uniform(
                0.5, 3.5, n_rows
            ),

            "latitude": np.random.uniform(
                3.5, 14.5, n_rows
            ),

            "longitude": np.random.uniform(
                34.0, 47.0, n_rows
            ),

        })


# =============================================================================
# DROPDOWN HELPERS
# =============================================================================

def get_filtered_options(
    dataframe,
    column,
    filters=None
):

    if column not in dataframe.columns:

        return []

    filtered_df = dataframe.copy()

    if filters:

        for filter_column, filter_value in filters.items():

            if (
                filter_column in filtered_df.columns
                and filter_value is not None
            ):

                filtered_df = filtered_df[
                    filtered_df[filter_column].astype(str)
                    == str(filter_value)
                ]

    values = (
        filtered_df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    return sorted(
        values.unique().tolist(),
        key=str.casefold
    )


def clear_dependent_selections(*keys):

    for key in keys:

        st.session_state.pop(
            key,
            None
        )


def format_price_status(value):

    labels = {

        "actual": "Actual",

        "forecast": "Forecast",

        "aggregate": "Aggregate",

        "actual,aggregate": "Actual + Aggregate",

    }

    return labels.get(
        str(value).lower(),
        str(value).title()
    )


def format_price_type(value):

    return str(value).title()


# =============================================================================
# APPLICATION HEADER
# =============================================================================

st.markdown(
    """
    <div style='display: flex; align-items: center;
                gap: 15px; margin-bottom: 0.5rem;'>

        <img src='https://flagcdn.com/w80/et.png'
             width='50'
             style='border-radius: 6px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.15);'/>

        <h1 style='margin: 0; font-size: 2.2rem;
                   color: #1e293b; font-weight: 800;'>

            Ethiopian Food Price Prediction & Market Analysis

        </h1>

    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "An interactive decision-support application built for "
    "localized market forecasting and price intelligence."
)


# =============================================================================
# NAVIGATION AND FILE UPLOAD
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
# PAGE 1: DATA OVERVIEW AND MARKET SIGNALS
# =============================================================================

if menu == "📊 1. Data Overview & Market Signals":

    st.header(
        "📊 Key Dataset Insights & Aggregations Overview"
    )

    st.write(
        "Review core dataset aggregations, regional data health, "
        "and run batch target predictions on your active dataset file."
    )

    # ---------------------------------------------------------
    # KEY METRICS
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            label="🧾 Total Recorded Transactions",
            value=(
                f"{len(df_data):,}"
                if not df_data.empty
                else "0"
            ),
        )

    with col2:

        st.metric(
            label="🏛️ Covered Regions (Admin1)",
            value=(
                str(df_data["admin1"].nunique())
                if not df_data.empty
                and "admin1" in df_data.columns
                else "0"
            ),
        )

    with col3:

        st.metric(
            label="🌾 Unique Commodities",
            value=(
                str(df_data["commodity"].nunique())
                if not df_data.empty
                and "commodity" in df_data.columns
                else "0"
            ),
        )

    with col4:

        st.metric(
            label="🏷️ Total Columns",
            value=(
                len(df_data.columns)
                if not df_data.empty
                else "0"
            ),
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------------

    if not df_data.empty:

        col_left, col_right = st.columns(
            [2, 1],
            gap="large"
        )

        with col_left:

            tab1, tab2 = st.tabs(
                [
                    "🛒 Top Commodities",
                    "🌍 Regional Volume",
                ]
            )

            with tab1:

                st.caption(
                    "Ranked by frequency and average price."
                )

                if {
                    "commodity",
                    "price",
                }.issubset(df_data.columns):

                    comm_summary = (
                        df_data.groupby("commodity")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions",
                            ascending=False
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

                    st.info(
                        "Commodity or price column missing."
                    )

            with tab2:

                st.caption(
                    "Distribution across administrative regions."
                )

                if {
                    "admin1",
                    "price",
                }.issubset(df_data.columns):

                    reg_summary = (
                        df_data.groupby("admin1")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions",
                            ascending=False
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

                    st.info(
                        "Admin1 or price column missing."
                    )

        # -----------------------------------------------------
        # DATA QUALITY
        # -----------------------------------------------------

        with col_right:

            st.markdown(
                "#### 🔍 Data Scope & Quality"
            )

            st.caption(
                "Record audit and memory statistics."
            )

            if "date" in df_data.columns:

                min_date = df_data["date"].min()
                max_date = df_data["date"].max()

                if pd.notna(min_date) and pd.notna(max_date):

                    st.metric(
                        "📅 Temporal Span",
                        f"{min_date.strftime('%Y-%m')} "
                        f"to {max_date.strftime('%Y-%m')}",
                    )

                else:

                    st.metric(
                        "📅 Temporal Span",
                        "N/A"
                    )

            st.metric(
                "🧹 Total Missing Cells",
                int(df_data.isna().sum().sum())
            )

            st.metric(
                "💾 Memory Footprint",
                f"{df_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
            )

            st.metric(
                "✨ Duplicate Rows",
                int(df_data.duplicated().sum())
            )

    else:

        st.warning(
            "No valid dataset is currently available."
        )

    # ---------------------------------------------------------
    # BATCH PREDICTION
    # ---------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔮 Batch Price & USD Price Prediction"
    )

    st.write(
        "Run the trained machine learning model across "
        "all records in your uploaded dataset."
    )

    if uploaded_file is None:

        st.warning(
            "⚠️ **Upload Required:** Upload a custom dataset "
            "using the sidebar uploader to enable batch predictions."
        )

    elif df_data.empty:

        st.error(
            "The uploaded dataset could not be loaded or "
            "does not contain the required columns."
        )

    else:

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
            "The uploaded price/usdprice values are not model inputs; "
            "each target is predicted independently."
        )

        if st.button(
            "🚀 Run Batch Prediction with Model"
        ):

            bundle, error_msg = load_model_assets()

            if error_msg:

                st.error(
                    f"Could not load model pipelines: {error_msg}"
                )

            elif not selected_targets:

                st.warning(
                    "Select at least one prediction target."
                )

            else:

                try:

                    df_predicted = df_data.copy()

                    X_batch = prepare_model_features(
                        df_predicted
                    )

                    for target_name in selected_targets:

                        if target_name not in bundle["models"]:

                            raise ValueError(
                                f"Saved artifact has no fitted "
                                f"model for {target_name}."
                            )

                        df_predicted[
                            f"Predicted_{target_name}"
                        ] = bundle["models"][
                            target_name
                        ].predict(X_batch)

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
                        f"Error during batch prediction: {error}"
                    )


# =============================================================================
# PAGE 2: EXPLORATORY ANALYTICS AND TRENDS
# =============================================================================

elif menu == "📈 2. Exploratory Analytics & Trends":

    st.header(
        "📈 Market Insights & Historical Data Visualizations"
    )

    st.markdown("---")

    if not df_data.empty:

        map_col, box_col = st.columns(
            2,
            gap="large"
        )

        # -----------------------------------------------------
        # GEOGRAPHICAL MAP
        # -----------------------------------------------------

        with map_col:

            st.subheader(
                "🗺️ Geospatial Market Price Map"
            )

            st.write(
                "Interactive map showing market monitoring "
                "locations across Ethiopia."
            )

            if {
                "latitude",
                "longitude",
                "price",
            }.issubset(df_data.columns):

                map_df = df_data.dropna(
                    subset=[
                        "latitude",
                        "longitude",
                        "price",
                    ]
                )

                if not map_df.empty:

                    fig_map = px.scatter_geo(
                        map_df.sample(
                            min(len(map_df), 2000)
                        ),

                        lat="latitude",
                        lon="longitude",

                        color="price",
                        size="price",

                        hover_name=(
                            "market"
                            if "market" in map_df.columns
                            else None
                        ),

                        hover_data=[
                            column
                            for column in [
                                "admin1",
                                "commodity",
                            ]
                            if column in map_df.columns
                        ],

                        color_continuous_scale=(
                            px.colors.sequential.Tealgrn
                        ),

                        projection="natural earth",

                        title="Market Locations & Price Intensity",
                    )

                    fig_map.update_geos(
                        center={
                            "lat": 9.145,
                            "lon": 40.489,
                        },
                        projection_scale=7,
                        showcountries=True,
                        countrycolor="gray",
                    )

                    fig_map.update_layout(
                        margin={
                            "r": 0,
                            "t": 40,
                            "l": 0,
                            "b": 0,
                        },
                        height=400,
                    )

                    st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "No valid geographical records available."
                    )

            else:

                st.info(
                    "Latitude, longitude, and price columns required."
                )

        # -----------------------------------------------------
        # REGIONAL PRICE DISTRIBUTION
        # -----------------------------------------------------

        with box_col:

            st.subheader(
                "📊 Regional Price Distributions"
            )

            st.write(
                "Price variance and outliers across administrative regions."
            )

            if {
                "admin1",
                "price",
            }.issubset(df_data.columns):

                box_data = df_data.dropna(
                    subset=[
                        "admin1",
                        "price",
                    ]
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
                    margin={
                        "r": 0,
                        "t": 40,
                        "l": 0,
                        "b": 0,
                    },
                    height=400,
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Admin1 and price columns required."
                )

        # -----------------------------------------------------
        # HISTORICAL PRICE TRENDS
        # -----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📉 Historical Staple Price Trends Over Time"
        )

        st.write(
            "Long-term monthly average price evolution "
            "for food commodities."
        )

        if {
            "date",
            "commodity",
            "price",
        }.issubset(df_data.columns):

            valid_trend_data = df_data.dropna(
                subset=[
                    "date",
                    "commodity",
                    "price",
                ]
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

            trend_df["date"] = (
                trend_df["date"].dt.to_timestamp()
            )

            major_staples = [
                "Maize (white)",
                "Wheat",
                "Sorghum",
                "Rice",
                "Teff",
            ]

            available_staples = [
                staple
                for staple in major_staples
                if staple in trend_df["commodity"].unique()
            ]

            if not available_staples:

                available_staples = (
                    trend_df["commodity"]
                    .drop_duplicates()
                    .head(4)
                    .tolist()
                )

            filtered_trend = trend_df[
                trend_df["commodity"].isin(
                    available_staples
                )
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
                margin={
                    "r": 0,
                    "t": 40,
                    "l": 0,
                    "b": 0,
                },
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True,
            )

        else:

            st.info(
                "Date, commodity, and price columns required."
            )

    else:

        st.info(
            "No dataset is currently available for visualization."
        )


# =============================================================================
# PAGE 3: INTERACTIVE PREDICTION ENGINE
# =============================================================================

elif menu == "🎯 3. Interactive Prediction Engine":

    st.header(
        "🎯 Real-Time Market Price Inference"
    )

    st.markdown(
        "Select geographic, market, and commodity parameters "
        "to evaluate predicted food prices."
    )

    if df_data.empty:

        st.error(
            "The dataset is empty. Please load a valid dataset first."
        )

    else:

        # =====================================================
        # SECTION 1: GEOGRAPHICAL SELECTION
        # =====================================================

        st.markdown(
            '<div class="section-header">'
            '📍 1. Geographical Selection'
            '</div>',
            unsafe_allow_html=True,
        )

        col_region, col_zone, col_market = st.columns(3)

        # -----------------------------------------------------
        # REGION
        # -----------------------------------------------------

        region_options = get_filtered_options(
            df_data,
            "admin1",
        )

        if not region_options:

            st.error(
                "No region choices are available."
            )

            st.stop()

        with col_region:

            admin1 = st.selectbox(
                "Region (Admin1)",
                options=region_options,
                key="prediction_admin1",
                on_change=clear_dependent_selections,
                args=(
                    "prediction_admin2",
                    "prediction_market",
                ),
            )

        # -----------------------------------------------------
        # ZONE FILTERED BY REGION
        # -----------------------------------------------------

        zone_options = get_filtered_options(
            df_data,
            "admin2",
            filters={
                "admin1": admin1,
            },
        )

        if not zone_options:

            st.warning(
                "No zones are available for the selected region."
            )

            st.stop()

        with col_zone:

            admin2 = st.selectbox(
                "Zone (Admin2)",
                options=zone_options,
                key="prediction_admin2",
                on_change=clear_dependent_selections,
                args=(
                    "prediction_market",
                ),
            )

        # -----------------------------------------------------
        # MARKET FILTERED BY REGION AND ZONE
        # -----------------------------------------------------

        market_options = get_filtered_options(
            df_data,
            "market",
            filters={
                "admin1": admin1,
                "admin2": admin2,
            },
        )

        if not market_options:

            st.warning(
                "No markets are available for the selected region and zone."
            )

            st.stop()

        with col_market:

            market = st.selectbox(
                "Local Market Name",
                options=market_options,
                key="prediction_market",
            )

        # =====================================================
        # SECTION 2: COMMODITY SELECTION
        # =====================================================

        st.markdown(
            '<div class="section-header">'
            '🛒 2. Commodity Selection'
            '</div>',
            unsafe_allow_html=True,
        )

        col_category, col_commodity, col_unit = st.columns(3)

        # -----------------------------------------------------
        # CATEGORY
        # -----------------------------------------------------

        category_options = get_filtered_options(
            df_data,
            "category",
        )

        if not category_options:

            st.error(
                "No commodity categories are available."
            )

            st.stop()

        with col_category:

            category = st.selectbox(
                "Commodity Category",
                options=category_options,
                key="prediction_category",
                on_change=clear_dependent_selections,
                args=(
                    "prediction_commodity",
                    "prediction_unit",
                ),
            )

        # -----------------------------------------------------
        # COMMODITY FILTERED BY CATEGORY
        # -----------------------------------------------------

        commodity_options = get_filtered_options(
            df_data,
            "commodity",
            filters={
                "category": category,
            },
        )

        if not commodity_options:

            st.warning(
                "No commodities are available for this category."
            )

            st.stop()

        with col_commodity:

            commodity = st.selectbox(
                "Commodity Name",
                options=commodity_options,
                key="prediction_commodity",
                on_change=clear_dependent_selections,
                args=(
                    "prediction_unit",
                ),
            )

        # -----------------------------------------------------
        # UNIT FILTERED BY CATEGORY AND COMMODITY
        # -----------------------------------------------------

        unit_options = get_filtered_options(
            df_data,
            "unit",
            filters={
                "category": category,
                "commodity": commodity,
            },
        )

        if not unit_options:

            st.warning(
                "No measurement units are available for this commodity."
            )

            st.stop()

        with col_unit:

            unit = st.selectbox(
                "Measurement Unit",
                options=unit_options,
                key="prediction_unit",
            )

        # =====================================================
        # SECTION 3: PRICE INFORMATION
        # =====================================================

        st.markdown(
            '<div class="section-header">'
            '🏷️ 3. Price Information'
            '</div>',
            unsafe_allow_html=True,
        )

        col_status, col_type = st.columns(2)

        # -----------------------------------------------------
        # PRICE STATUS
        # -----------------------------------------------------

        priceflag_options = get_filtered_options(
            df_data,
            "priceflag",
        )

        if not priceflag_options:

            st.error(
                "No price status choices are available."
            )

            st.stop()

        with col_status:

            priceflag = st.selectbox(
                "Price Status",
                options=priceflag_options,
                format_func=format_price_status,
                key="prediction_priceflag",
            )

        # -----------------------------------------------------
        # PRICE TYPE
        # -----------------------------------------------------

        pricetype_options = get_filtered_options(
            df_data,
            "pricetype",
        )

        if not pricetype_options:

            st.error(
                "No price type choices are available."
            )

            st.stop()

        with col_type:

            pricetype = st.selectbox(
                "Price Type",
                options=pricetype_options,
                format_func=format_price_type,
                key="prediction_pricetype",
            )

        # =====================================================
        # SECTION 4: DATE AND LOCATION COORDINATES
        # =====================================================

        st.markdown(
            '<div class="section-header">'
            '📅 4. Prediction Date and Coordinates'
            '</div>',
            unsafe_allow_html=True,
        )

        col_date, col_lat, col_lon = st.columns(3)

        # Default date from dataset
        valid_dates = df_data["date"].dropna()

        if not valid_dates.empty:

            default_date = valid_dates.max().date()

        else:

            default_date = pd.to_datetime(
                "2022-12-15"
            ).date()

        with col_date:

            date_val = st.date_input(
                "Transaction Date",
                value=default_date,
                key="prediction_date",
            )

        with col_lat:

            latitude = st.slider(
                "Latitude Coordinate",
                min_value=3.0000,
                max_value=15.0000,
                value=9.0200,
                step=0.0100,
                format="%.4f",
                key="prediction_latitude",
            )

        with col_lon:

            longitude = st.slider(
                "Longitude Coordinate",
                min_value=33.0000,
                max_value=48.0000,
                value=38.7500,
                step=0.0100,
                format="%.4f",
                key="prediction_longitude",
            )

        # =====================================================
        # SECTION 5: PREDICTION
        # =====================================================

        st.markdown("---")

        if st.button(
            "🚀 Predict Market Price",
            key="run_market_prediction",
        ):

            bundle, error_msg = load_model_assets()

            if error_msg:

                st.error(
                    "Could not process prediction. "
                    f"Serialized assets error: {error_msg}"
                )

            else:

                try:

                    # Create prediction input
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

                    # Prepare model features
                    X_input = prepare_model_features(
                        input_row
                    )

                    # Check fitted price model
                    if "price" not in bundle["models"]:

                        raise ValueError(
                            "The saved model bundle does not contain "
                            "a fitted price prediction model."
                        )

                    # Predict price
                    prediction = bundle["models"][
                        "price"
                    ].predict(X_input)[0]

                    st.success(
                        f"✨ Predicted Local Market Price: "
                        f"**{prediction:,.2f} ETB**"
                    )

                    # Show selected parameters
                    with st.expander(
                        "📋 View Selected Prediction Parameters"
                    ):

                        st.dataframe(
                            input_row,
                            use_container_width=True,
                        )

                    # -------------------------------------------------
                    # MODEL PERFORMANCE METRICS
                    # -------------------------------------------------

                    with st.expander(
                        "📌 View Model Training & Test Performance Metrics"
                    ):

                        if (
                            "price" in bundle["metrics"]
                            and isinstance(
                                bundle["metrics"]["price"],
                                dict
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
                                "Price model evaluation metrics "
                                "are not available in the saved bundle."
                            )

                except Exception as error:

                    st.error(
                        f"Error processing prediction parameters: {error}"
                    )
