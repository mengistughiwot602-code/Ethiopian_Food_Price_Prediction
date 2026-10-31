
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon="https://flagcdn.com/w80/et.png",
    layout="wide",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

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

# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_model_features(data):

    required = CATEGORICAL_FEATURES + [
        "latitude",
        "longitude",
        "date",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Uploaded data is missing required model features: {missing}"
        )

    features = data[required].copy()

    dates = pd.to_datetime(
        features["date"],
        errors="coerce",
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
            errors="coerce",
        )

    return features[MODEL_FEATURES]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model_assets():

    try:
        bundle = joblib.load(
            "random_forest_model.joblib"
        )

        if (
            not isinstance(bundle, dict)
            or not {
                "models",
                "metrics",
                "target_columns",
            }.issubset(bundle)
        ):
            raise ValueError(
                "Saved artifact is not the expected pipeline bundle. "
                "Rerun the training cell."
            )

        return bundle, None

    except Exception as e:
        return None, str(e)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset(uploaded_file=None):

    try:

        if uploaded_file is not None:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(
                    uploaded_file,
                    header=0,
                )

            else:
                df = pd.read_excel(
                    uploaded_file,
                    header=0,
                )

            core_features = [
                "admin1",
                "commodity",
                "latitude",
                "longitude",
                "price",
            ]

            matched = sum(
                1 for feature in core_features
                if feature in df.columns
            )

            if matched < 2:
                return pd.DataFrame()

        elif os.path.exists("wfp_food_prices_et.xlsx"):

            df = pd.read_excel(
                "wfp_food_prices_et.xlsx",
                header=0,
            )

        else:

            excel_files = [
                file for file in os.listdir(".")
                if file.endswith(".xlsx")
            ]

            if excel_files:
                df = pd.read_excel(
                    excel_files[0],
                    header=0,
                )

            else:
                raise FileNotFoundError(
                    "No dataset file found locally."
                )

        if len(df) > 0:
            df = df.drop(0, errors="ignore").reset_index(drop=True)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

        for column in [
            "price",
            "usdprice",
            "latitude",
            "longitude",
        ]:

            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        return df

    except Exception:

        if uploaded_file is not None:
            return pd.DataFrame()

        np.random.seed(42)

        n_rows = 500

        return pd.DataFrame({
            "date": pd.date_range(
                start="2020-01-01",
                periods=n_rows,
                freq="D",
            ),

            "admin1": np.random.choice(
                ["Addis Ababa", "Oromia", "Amhara", "Tigray"],
                n_rows,
            ),

            "admin2": np.random.choice(
                ["Addis Ababa Zone", "Adama Zone"],
                n_rows,
            ),

            "market": np.random.choice(
                ["Addis Ababa Market", "Adama Market"],
                n_rows,
            ),

            "commodity": np.random.choice(
                ["Maize (white)", "Wheat", "Sorghum"],
                n_rows,
            ),

            "category": ["Cereals and Tubers"] * n_rows,
            "unit": ["KG"] * n_rows,
            "priceflag": [0] * n_rows,
            "pricetype": [0] * n_rows,

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


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem;'>
        <img src='https://flagcdn.com/w80/et.png' width='50'
        style='border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);'/>
        <h1 style='margin: 0; font-size: 2.2rem; color: #1e293b; font-weight: 800;'>
            Ethiopian Food Price Prediction & Market Analysis
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "An interactive decision-support application built for localized market "
    "forecasting and price intelligence."
)


# ============================================================
# NAVIGATION MENU
# ============================================================

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
    "📁 Upload Custom Dataset (Required for Batch Prediction)",
    type=["xlsx", "csv"],
)

df_data = load_dataset(uploaded_file)


# ============================================================
# PAGE 1: DATA OVERVIEW & MARKET SIGNALS
# ============================================================

if menu == "📊 1. Data Overview & Market Signals":

    st.header(
        "📊 Key Dataset Insights & Aggregations Overview"
    )

    st.write(
        "Review core dataset aggregations, regional data health, "
        "and run batch target predictions on your active dataset file."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🧾 Total Recorded Transactions",
            value=(
                f"{len(df_data):,}"
                if df_data is not None and not df_data.empty
                else "0"
            ),
        )

    with col2:
        st.metric(
            label="🏛️ Covered Regions (Admin1)",
            value=(
                str(df_data["admin1"].nunique())
                if (
                    df_data is not None
                    and not df_data.empty
                    and "admin1" in df_data.columns
                )
                else "0"
            ),
        )

    with col3:
        st.metric(
            label="🌾 Unique Commodities",
            value=(
                str(df_data["commodity"].nunique())
                if (
                    df_data is not None
                    and not df_data.empty
                    and "commodity" in df_data.columns
                )
                else "0"
            ),
        )

    with col4:
        st.metric(
            label="🏷️ Total Columns",
            value=(
                len(df_data.columns)
                if df_data is not None and not df_data.empty
                else "0"
            ),
        )

    st.markdown("---")

    if df_data is not None and not df_data.empty:

        col_left, col_right = st.columns(
            [2, 1],
            gap="large",
        )

        with col_left:

            tab1, tab2 = st.tabs(
                ["🛒 Top Commodities", "🌍 Regional Volume"]
            )

            with tab1:

                st.caption(
                    "Ranked by frequency and average price."
                )

                if {"commodity", "price"}.issubset(df_data.columns):

                    comm_summary = (
                        df_data.groupby("commodity")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions",
                            ascending=False,
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

                if {"admin1", "price"}.issubset(df_data.columns):

                    reg_summary = (
                        df_data.groupby("admin1")
                        .agg(
                            Transactions=("price", "count"),
                            Avg_Price=("price", "mean"),
                        )
                        .reset_index()
                        .sort_values(
                            by="Transactions",
                            ascending=False,
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

        with col_right:

            st.markdown(
                "#### 🔍 Data Scope & Quality"
            )

            st.caption(
                "Record audit and memory stats."
            )

            if (
                "date" in df_data.columns
                and pd.api.types.is_datetime64_any_dtype(
                    df_data["date"]
                )
            ):

                min_date = df_data["date"].min()
                max_date = df_data["date"].max()

                if (
                    pd.notna(min_date)
                    and pd.notna(max_date)
                ):

                    st.metric(
                        "📅 Temporal Span",
                        f"{min_date.strftime('%Y-%m')} to "
                        f"{max_date.strftime('%Y-%m')}",
                    )

                else:
                    st.metric(
                        "📅 Temporal Span",
                        "N/A",
                    )

            else:
                st.metric(
                    "📅 Temporal Span",
                    "N/A",
                )

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

    st.markdown("---")

    st.subheader(
        "🔮 Batch Price & USD Price Prediction"
    )

    st.write(
        "Run the trained machine learning model across all records "
        "in your uploaded dataset."
    )

    if uploaded_file is None:

        st.warning(
            "⚠️ **Upload Required:** Please upload a custom dataset "
            "file (CSV or Excel) using the sidebar uploader above "
            "to enable batch predictions."
        )

    elif df_data is None or df_data.empty:

        st.error(
            "⚠️ **Dataset Feature Mismatch:** The uploaded dataset's "
            "features do not match the model's required training columns. "
            "Please ensure your file contains the correct regional, "
            "commodity, or spatial columns."
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
                    "Select at least one target: price and/or usdprice."
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
                                f"Saved artifact has no fitted model "
                                f"for {target_name}."
                            )

                        df_predicted[
                            f"Predicted_{target_name}"
                        ] = bundle["models"][target_name].predict(
                            X_batch
                        )

                    st.success(
                        "Batch prediction completed with the saved "
                        "OneHot pipelines for: "
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

                except Exception as ex:

                    st.error(
                        f"Error during batch prediction execution: {ex}"
                    )


# ============================================================
# PAGE 2: EXPLORATORY ANALYTICS & TRENDS
# ============================================================

elif menu == "📈 2. Exploratory Analytics & Trends":

    st.header(
        "📈 Market Insights & Historical Data Visualizations"
    )

    st.markdown("---")

    if df_data is not None and not df_data.empty:

        map_col, box_col = st.columns(
            2,
            gap="large",
        )

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
                        hover_data=(
                            ["admin1", "commodity"]
                            if {
                                "admin1",
                                "commodity",
                            }.issubset(map_df.columns)
                            else None
                        ),
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
                        "No valid geographic records are available."
                    )

            else:
                st.info(
                    "Latitude, longitude, or price columns "
                    "required for map view."
                )

        with box_col:

            st.subheader(
                "📊 Regional Price Distributions"
            )

            st.write(
                "Price variance and outliers across "
                "administrative regions."
            )

            if {
                "admin1",
                "price",
            }.issubset(df_data.columns):

                box_data = df_data.dropna(
                    subset=["admin1", "price"]
                )

                if not box_data.empty:

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
                        "No valid regional price records are available."
                    )

            else:
                st.info(
                    "Admin1 and price columns required for box plot view."
                )

        st.markdown("---")

        st.subheader(
            "📉 Historical Staple Price Trends Over Time"
        )

        st.write(
            "Long-term monthly average price evolution "
            "for core food commodities."
        )

        if {
            "date",
            "commodity",
            "price",
        }.issubset(df_data.columns):

            trend_data = df_data.dropna(
                subset=["date", "commodity", "price"]
            )

            if not trend_data.empty:

                trend_df = (
                    trend_data.groupby(
                        [
                            trend_data["date"].dt.to_period("M"),
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
                        trend_df["commodity"].unique()[:4]
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
                    "No valid date, commodity, and price records "
                    "are available for trend analysis."
                )

        else:
            st.info(
                "Date, commodity, and price columns "
                "required for trend analysis."
            )

    else:

        st.info(
            "No dataset is currently available for visualization."
        )


# ============================================================
# PAGE 3: INTERACTIVE PREDICTION ENGINE
# ============================================================

elif menu == "🎯 3. Interactive Prediction Engine":

    st.header(
        "🎯 Real-Time Market Price Inference"
    )

    st.markdown(
        "Select actual geographic, market, and commodity parameters "
        "below to evaluate predicted food prices."
    )

    # --------------------------------------------------------
    # GET UNIQUE VALUES FROM THE CURRENT FILTERED DATASET
    # --------------------------------------------------------

    def category_options(data, column):

        if column not in data.columns:
            return []

        values = (
            data[column]
            .dropna()
            .unique()
            .tolist()
        )

        return sorted(
            values,
            key=lambda value: str(value).lower(),
        )

    # --------------------------------------------------------
    # FILTER DATA BASED ON A SELECTED VALUE
    # --------------------------------------------------------

    def filter_data(data, column, selected_value):

        if (
            selected_value is None
            or column not in data.columns
        ):
            return data

        return data[
            data[column].astype(str)
            == str(selected_value)
        ].copy()

    # --------------------------------------------------------
    # GET DESCRIPTIVE LABELS WHEN AVAILABLE
    # --------------------------------------------------------

    def get_display_labels(data, column):

        possible_labels = {
            "priceflag": [
                "priceflag_label",
                "price_status",
                "priceflag_description",
                "price_status_description",
            ],

            "pricetype": [
                "pricetype_label",
                "price_type",
                "pricetype_description",
                "price_type_description",
            ],
        }

        label_column = None

        for candidate in possible_labels.get(column, []):

            if candidate in data.columns:
                label_column = candidate
                break

        labels = {}

        if label_column is not None:

            mapping_data = data[
                [column, label_column]
            ].dropna().drop_duplicates()

            for _, row in mapping_data.iterrows():

                labels[row[column]] = str(
                    row[label_column]
                )

        return labels

    # --------------------------------------------------------
    # SAFE SELECTBOX WITH AUTOMATIC OPTION UPDATES
    # --------------------------------------------------------

    def dynamic_selectbox(
        data,
        column,
        label,
        key,
        display_labels=None,
    ):

        options = category_options(
            data,
            column,
        )

        if not options:

            st.warning(
                f"No available choices for {label} "
                "based on the current selections."
            )

            return None

        current_value = st.session_state.get(key)

        if current_value not in options:

            st.session_state[key] = options[0]

        if display_labels is None:
            display_labels = {}

        return st.selectbox(
            label,
            options=options,
            key=key,
            format_func=lambda value: display_labels.get(
                value,
                str(value),
            ),
        )

    # --------------------------------------------------------
    # PREDICTION INTERFACE
    # --------------------------------------------------------

    if df_data is None or df_data.empty:

        st.error(
            "The dataset is empty or unavailable. "
            "Please check your dataset file."
        )

    else:

        try:

            required_dropdown_columns = [
                "admin1",
                "admin2",
                "market",
                "category",
                "commodity",
                "unit",
                "priceflag",
                "pricetype",
            ]

            missing_columns = [
                column
                for column in required_dropdown_columns
                if column not in df_data.columns
            ]

            if missing_columns:

                st.error(
                    "The dataset is missing the following "
                    f"required prediction columns: {missing_columns}"
                )

            else:

                # Start with the complete dataset.
                filtered_data = df_data.copy()

                col_a, col_b = st.columns(
                    2,
                    gap="large",
                )

                # ------------------------------------------------
                # LEFT COLUMN: LOCATION AND DATE
                # ------------------------------------------------

                with col_a:

                    st.markdown(
                        '<div class="section-header">📍 Spatial & Temporal Details</div>',
                        unsafe_allow_html=True,
                    )

                    date_val = st.date_input(
                        "Transaction Date",
                        value=pd.to_datetime("2022-12-15"),
                        key="prediction_date",
                    )

                    latitude = st.slider(
                        "Latitude Coordinate",
                        min_value=3.0000,
                        max_value=15.0000,
                        value=9.0200,
                        step=0.0100,
                        format="%.4f",
                        key="prediction_latitude",
                    )

                    longitude = st.slider(
                        "Longitude Coordinate",
                        min_value=33.0000,
                        max_value=48.0000,
                        value=38.7500,
                        step=0.0100,
                        format="%.4f",
                        key="prediction_longitude",
                    )

                    # Admin1
                    admin1 = dynamic_selectbox(
                        filtered_data,
                        "admin1",
                        "Region (Admin1)",
                        "prediction_admin1",
                    )

                    if admin1 is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "admin1",
                            admin1,
                        )

                    # Admin2 depends on Admin1.
                    admin2 = dynamic_selectbox(
                        filtered_data,
                        "admin2",
                        "Zone (Admin2)",
                        "prediction_admin2",
                    )

                    if admin2 is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "admin2",
                            admin2,
                        )

                # ------------------------------------------------
                # RIGHT COLUMN: MARKET AND COMMODITY
                # ------------------------------------------------

                with col_b:

                    st.markdown(
                        '<div class="section-header">🛒 Market & Commodity Attributes</div>',
                        unsafe_allow_html=True,
                    )

                    # Market depends on selected location.
                    market = dynamic_selectbox(
                        filtered_data,
                        "market",
                        "Local Market Name",
                        "prediction_market",
                    )

                    if market is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "market",
                            market,
                        )

                    # Category depends on selected market.
                    category = dynamic_selectbox(
                        filtered_data,
                        "category",
                        "Commodity Category",
                        "prediction_category",
                    )

                    if category is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "category",
                            category,
                        )

                    # Commodity depends on category.
                    commodity = dynamic_selectbox(
                        filtered_data,
                        "commodity",
                        "Commodity Name",
                        "prediction_commodity",
                    )

                    if commodity is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "commodity",
                            commodity,
                        )

                    # Unit depends on selected commodity.
                    unit = dynamic_selectbox(
                        filtered_data,
                        "unit",
                        "Measurement Unit",
                        "prediction_unit",
                    )

                    if unit is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "unit",
                            unit,
                        )

                    # Price Status
                    priceflag_labels = get_display_labels(
                        df_data,
                        "priceflag",
                    )

                    priceflag = dynamic_selectbox(
                        filtered_data,
                        "priceflag",
                        "Price Status",
                        "prediction_priceflag",
                        display_labels=priceflag_labels,
                    )

                    if priceflag is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "priceflag",
                            priceflag,
                        )

                    # Price Type
                    pricetype_labels = get_display_labels(
                        df_data,
                        "pricetype",
                    )

                    pricetype = dynamic_selectbox(
                        filtered_data,
                        "pricetype",
                        "Price Type",
                        "prediction_pricetype",
                        display_labels=pricetype_labels,
                    )

                    if pricetype is not None:

                        filtered_data = filter_data(
                            filtered_data,
                            "pricetype",
                            pricetype,
                        )

                # ------------------------------------------------
                # PREDICTION BUTTON
                # ------------------------------------------------

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True,
                )

                submit = st.button(
                    "🚀 Predict Market Price",
                    key="run_market_prediction",
                    use_container_width=True,
                )

                # ------------------------------------------------
                # CHECK SELECTED FEATURES
                # ------------------------------------------------

                selection_ready = all(
                    value is not None
                    for value in [
                        admin1,
                        admin2,
                        market,
                        category,
                        commodity,
                        unit,
                        priceflag,
                        pricetype,
                    ]
                )

                if submit:

                    if not selection_ready:

                        st.warning(
                            "Please select valid values for all "
                            "geographic and commodity features."
                        )

                    else:

                        bundle, error_msg = load_model_assets()

                        if error_msg:

                            st.error(
                                "Could not process prediction. "
                                f"Serialized assets error: {error_msg}"
                            )

                        else:

                            try:

                                dt_date = pd.to_datetime(
                                    date_val
                                )

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
                                    "date": dt_date,
                                }])

                                prediction = bundle["models"][
                                    "price"
                                ].predict(
                                    prepare_model_features(input_row)
                                )[0]

                                st.success(
                                    f"✨ Predicted Local Market Price: "
                                    f"**{prediction:,.2f} ETB**"
                                )

                                # Model performance metrics
                                with st.expander(
                                    "📌 View Model Training & Test Performance Metrics"
                                ):

                                    m1, m2, m3, m4 = st.columns(4)

                                    price_metrics = bundle[
                                        "metrics"
                                    ]["price"]

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

                            except Exception as ex:

                                st.error(
                                    f"Error processing prediction parameters: {ex}"
                                )

        except Exception as ex:

            st.error(
                f"Unable to create prediction interface: {ex}"
            )
