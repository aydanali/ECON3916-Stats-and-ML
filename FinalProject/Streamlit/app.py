# ============================================================
# app.py — GDP & Emissions Dashboard
# World Bank Environmental Economics Project
# ============================================================
# Tabs:
#   1. Data Overview   — cleaned OWID panel with summary stats & charts
#   2. Model Explorer  — interactive feature selection, model comparison
#   3. Predictor       — sidebar controls → point estimate + PI via MAPIE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="GDP & Emissions | World Bank",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a clean, professional look ─────────────────────────────────
st.markdown("""
<style>
    /* Main background and font */
    .main { background-color: #F8F9FA; }
    h1 { color: #1a3a5c; }
    h2, h3 { color: #1a3a5c; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e4ea;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1a3a5c; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSlider > label { color: #cce0f5 !important; }

    /* Info boxes */
    .info-box {
        background: #eaf4fb;
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #FFC107;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match the notebook exactly
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Feature columns fed into the models (log-transformed where applicable)
MODEL_FEATURES = [
    "co2_log",
    "co2_per_unit_energy",
    "energy_per_capita",
    "ghg_per_capita",
    "methane_log",
    "nitrous_oxide_log",
    "oil_co2_log",
    "primary_energy_consumption_log",
    "total_ghg_log",
]

# Human-readable labels for each model feature (used in UI sliders / charts)
FEATURE_LABELS = {
    "co2_log":                         "CO₂ Emissions (Mt)",
    "co2_per_unit_energy":             "CO₂ Intensity (kg per kWh)",
    "energy_per_capita":               "Energy per Capita (kWh)",
    "ghg_per_capita":                  "GHG per Capita (tCO₂e)",
    "methane_log":                     "Methane Emissions (Mt CO₂e)",
    "nitrous_oxide_log":               "Nitrous Oxide Emissions (Mt CO₂e)",
    "oil_co2_log":                     "Oil CO₂ Emissions (Mt)",
    "primary_energy_consumption_log":  "Primary Energy Consumption (TWh)",
    "total_ghg_log":                   "Total GHG Emissions (Mt CO₂e)",
}

# Raw (un-logged) column names that the user-facing sliders operate on,
# mapped to their log-transformed counterparts used in models
RAW_TO_LOG = {
    "co2":                        "co2_log",
    "methane":                    "methane_log",
    "nitrous_oxide":              "nitrous_oxide_log",
    "oil_co2":                    "oil_co2_log",
    "primary_energy_consumption": "primary_energy_consumption_log",
    "total_ghg":                  "total_ghg_log",
}
# Features that are NOT log-transformed (sliders operate directly on these)
DIRECT_FEATURES = ["co2_per_unit_energy", "energy_per_capita", "ghg_per_capita"]


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading — cached so they only execute once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """
    Load the cleaned OWID dashboard dataset.
    Returns the full cleaned DataFrame (all columns) and the model-ready
    subset (log-transformed features + gdp_log).
    """
    df_full  = pd.read_csv("https://raw.githubusercontent.com/aydanali/ECON3916-Stats-and-ML/refs/heads/main/FinalProject/Streamlit/owid_clean_dash.csv")
    df_model = pd.read_csv("https://raw.githubusercontent.com/aydanali/ECON3916-Stats-and-ML/refs/heads/main/FinalProject/Streamlit/owid_clean_dash.csv")
    return df_full, df_model


@st.cache_resource
def load_models():
    import warnings
    warnings.filterwarnings("ignore")

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    # ── Load raw data ────────────────────────────────────────────
    df_raw = pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/refs/heads/master/owid-co2-data.csv')

    # ── Clean ────────────────────────────────────────────────────
    START_YEAR            = 1990
    END_YEAR              = 2024
    COLUMN_NAN_THRESHOLD  = 0.50
    COUNTRY_NAN_THRESHOLD = 0.50
    FOCUS_COLS = [
        "co2", "co2_per_capita", "primary_energy_consumption",
        "energy_per_capita", "energy_per_gdp", "gdp",
        "methane", "nitrous_oxide", "total_ghg",
        "coal_co2", "gas_co2", "oil_co2",
    ]

    df = df_raw[df_raw["iso_code"].notna()].copy()
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    miss_share = df.isna().mean()
    drop_cols  = [c for c in miss_share[miss_share > COLUMN_NAN_THRESHOLD].index
                  if c not in ["country", "year", "iso_code"]]
    df = df.drop(columns=drop_cols)

    focus_present = [c for c in FOCUS_COLS if c in df.columns]
    country_miss  = (df.groupby("country")[focus_present]
                       .apply(lambda g: g.isna().to_numpy().mean()))
    keep          = country_miss[country_miss <= COUNTRY_NAN_THRESHOLD].index
    df            = df[df["country"].isin(keep)].copy()

    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    id_cols      = ["country", "year", "iso_code"]
    numeric_cols = [c for c in df.columns
                    if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    for _, idx in df.groupby("country").groups.items():
        df.loc[idx, numeric_cols] = (
            df.loc[idx, numeric_cols]
              .interpolate(method="linear", limit_direction="both", limit_area="inside")
              .values
        )

    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df = df[[
        "country", "year", "iso_code", "population", "gdp", "co2",
        "co2_per_unit_energy", "energy_per_capita", "ghg_per_capita",
        "methane", "nitrous_oxide", "oil_co2",
        "primary_energy_consumption", "total_ghg",
    ]]

    interp_cols = [c for c in df.columns if c not in ["country", "year", "iso_code"]]
    for col in interp_cols:
        df[col] = (df.groupby("country")[col]
                     .transform(lambda x: x.interpolate())
                     .ffill().bfill())

    # ── Feature engineering ──────────────────────────────────────
    df["gdp_log"]                        = np.log1p(df["gdp"])
    df["co2_log"]                        = np.log1p(df["co2"])
    df["methane_log"]                    = np.log1p(df["methane"])
    df["nitrous_oxide_log"]              = np.log1p(df["nitrous_oxide"])
    df["oil_co2_log"]                    = np.log1p(df["oil_co2"])
    df["primary_energy_consumption_log"] = np.log1p(df["primary_energy_consumption"])
    df["total_ghg_log"]                  = np.log1p(df["total_ghg"])

    MODEL_FEATURES = [
        "co2_log", "co2_per_unit_energy", "energy_per_capita",
        "ghg_per_capita", "methane_log", "nitrous_oxide_log",
        "oil_co2_log", "primary_energy_consumption_log", "total_ghg_log",
    ]
    TARGET = "gdp_log"

    df_model = df[["country", "year"] + MODEL_FEATURES + [TARGET]].dropna()

    # ── Train models ─────────────────────────────────────────────
    X = df_model[MODEL_FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train)
    X_test_sc     = scaler.transform(X_test)
    X_train_sc_df = pd.DataFrame(X_train_sc, columns=MODEL_FEATURES)
    X_test_sc_df  = pd.DataFrame(X_test_sc,  columns=MODEL_FEATURES)

    model_ols = LinearRegression()
    model_ols.fit(X_train_sc_df, y_train)

    ridge_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    ridge_model.fit(X_train_sc_df, y_train)

    model_rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)
    model_rf.fit(X_train, y_train)

    # ── Feature stats ────────────────────────────────────────────
    RAW_LOG_COLS = ["co2", "methane", "nitrous_oxide",
                    "oil_co2", "primary_energy_consumption", "total_ghg"]
    DIRECT_COLS  = ["co2_per_unit_energy", "energy_per_capita", "ghg_per_capita"]

    feat_stats = {}
    for col in RAW_LOG_COLS + DIRECT_COLS:
        feat_stats[col] = {
            "min":    float(df[col].quantile(0.01)),
            "max":    float(df[col].quantile(0.99)),
            "median": float(df[col].median()),
        }

    return model_ols, ridge_model, model_rf, scaler, feat_stats


@st.cache_data
def build_train_test(_df_model):
    """
    Reconstruct the exact train/test split used in the notebook.
    Underscore prefix on df_model tells Streamlit not to hash the DataFrame
    argument (avoids slow hashing on every call).
    """
    X = _df_model[MODEL_FEATURES]
    y = _df_model["gdp_log"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


@st.cache_data
def compute_test_predictions(_df_model, _scaler):
    """Pre-compute test-set predictions for all three models (used in Tab 2)."""
    X_tr, X_te, y_tr, y_te = build_train_test(_df_model)
    X_te_sc = _scaler.transform(X_te)

    m_ols, m_ridge, m_rf, *_ = load_models()

    return {
        "OLS Regression":   (y_te, m_ols.predict(X_te_sc)),
        "Ridge Regression": (y_te, m_ridge.predict(X_te_sc)),
        "Random Forest":    (y_te, m_rf.predict(X_te.values)),
    }


def metrics_dict(y_true, y_pred):
    """Compute RMSE, MAE, R² for a prediction array."""
    return {
        "R²":   r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAPIE prediction intervals helper
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction_interval(model_name, raw_input_df, scaler, feat_stats, df_model, alpha=0.10):
    """
    Fit a MAPIE conformal predictor around the chosen model and return the
    point estimate plus (lower, upper) prediction interval at (1-alpha) coverage.

    MAPIE uses split conformal prediction: it fits the base model on a
    calibration split of the training data and uses residuals to construct
    distribution-free prediction intervals — no normality assumption required.

    Parameters
    ----------
    model_name    : 'OLS Regression' | 'Ridge Regression' | 'Random Forest'
    raw_input_df  : 1-row DataFrame with raw (un-logged) user inputs
    scaler        : fitted StandardScaler
    feat_stats    : dict of raw column stats (for slider bounds)
    df_model      : full model-ready DataFrame for calibration
    alpha         : significance level (0.10 → 90% PI)
    """
    try:
        from mapie.regression import MapieRegressor
    except ImportError:
        # If MAPIE is not installed, fall back to ±1.96*RMSE of the test set
        return None, None, None

    # Build log-transformed feature DataFrame (preserves feature names)
    X_input = _build_feature_row(raw_input_df)

    # Calibration data — full model set keeps enough residuals for MAPIE
    X_all = df_model[MODEL_FEATURES]   # DataFrame, preserves feature names
    y_all = df_model["gdp_log"].values

    # Select base estimator and apply scaling where required
    if model_name == "OLS Regression":
        base  = LinearRegression()
        X_cal = pd.DataFrame(scaler.transform(X_all), columns=MODEL_FEATURES)
        X_new = pd.DataFrame(scaler.transform(X_input), columns=MODEL_FEATURES)
    elif model_name == "Ridge Regression":
        base  = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        X_cal = pd.DataFrame(scaler.transform(X_all), columns=MODEL_FEATURES)
        X_new = pd.DataFrame(scaler.transform(X_input), columns=MODEL_FEATURES)
    else:
        base  = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        X_cal = X_all
        X_new = X_input

    mapie = MapieRegressor(estimator=base, method="base", cv=5)
    mapie.fit(X_cal, y_all)
    y_pred, y_pi = mapie.predict(X_new, alpha=alpha)

    point   = float(y_pred[0])
    lower   = float(y_pi[0, 0, 0])
    upper   = float(y_pi[0, 1, 0])
    return point, lower, upper


def _build_feature_row(raw_df):
    """
    Convert a 1-row DataFrame of raw (user-slider) values into the 9-column
    log-transformed feature DataFrame that the models expect.
    Returns a DataFrame (not a numpy array) so sklearn doesn't warn about
    missing feature names when calling predict().
    """
    row = {}
    for raw_col, log_col in RAW_TO_LOG.items():
        row[log_col] = np.log1p(float(raw_df[raw_col].iloc[0]))
    for col in DIRECT_FEATURES:
        row[col] = float(raw_df[col].iloc[0])
    # Return as single-row DataFrame with columns in the exact model order
    return pd.DataFrame([row])[MODEL_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# Load data & models (runs once, then cached)
# ─────────────────────────────────────────────────────────────────────────────
df_full, df_model = load_data()
model_ols, model_ridge, model_rf, scaler, feat_stats = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Persistent header — always visible regardless of active tab
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌍 GDP, Energy & Emissions Explorer")
st.markdown(
    """
    **World Bank Environmental Economics Project** &nbsp;|&nbsp;
    Data: [Our World in Data CO₂ Dataset](https://github.com/owid/co2-data) &nbsp;|&nbsp;
    Panel: 201 countries · 1990–2024
    """
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Three main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Data Overview",
    "🔬  Model Explorer",
    "🎯  GDP Predictor",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Dataset Overview")
    st.markdown("""
    <div class="info-box">
    The cleaned panel covers <b>201 countries</b> from <b>1990 to 2024</b>, derived from
    the OWID CO₂ dataset. Regional aggregates and micro-territories with &gt;50% missing
    data have been removed. Interior gaps are filled by linear interpolation within each
    country's time series.
    </div>
    """, unsafe_allow_html=True)

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Countries", f"{df_full['country'].nunique():,}")
    k2.metric("Year Range", f"{int(df_full['year'].min())} – {int(df_full['year'].max())}")
    k3.metric("Observations", f"{len(df_full):,}")
    k4.metric("Features", f"{df_full.shape[1]}")

    st.markdown("#### Summary Statistics")
    # Only display key numeric columns to keep the table manageable
    display_cols = ["gdp", "co2", "primary_energy_consumption",
                    "total_ghg", "energy_per_capita", "ghg_per_capita",
                    "methane", "nitrous_oxide", "oil_co2"]
    present = [c for c in display_cols if c in df_full.columns]
    st.dataframe(df_full[present].describe().round(2), width="stretch")

    # ── Full data table with country/year filter ──────────────────────────────
    st.markdown("#### Explore the Data")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        country_options = ["All"] + sorted(df_full["country"].unique().tolist())
        selected_country = st.selectbox("Filter by country", country_options, key="tab1_country")
    with col_f2:
        year_range = st.slider(
            "Year range",
            int(df_full["year"].min()), int(df_full["year"].max()),
            (2000, 2022), key="tab1_years"
        )

    df_view = df_full[
        (df_full["year"] >= year_range[0]) & (df_full["year"] <= year_range[1])
    ]
    if selected_country != "All":
        df_view = df_view[df_view["country"] == selected_country]

    st.dataframe(
        df_view[["country", "year", "gdp", "co2", "primary_energy_consumption",
                 "total_ghg", "energy_per_capita", "ghg_per_capita",
                 "methane", "nitrous_oxide", "oil_co2"]].reset_index(drop=True),
        width="stretch", height=300
    )
    st.caption(f"Showing {len(df_view):,} rows")

    # ── Global emissions trend ────────────────────────────────────────────────
    st.markdown("#### Global Trends Over Time")
    trend_metric = st.selectbox(
        "Select metric to plot",
        options=["co2", "total_ghg", "primary_energy_consumption",
                 "gdp", "methane", "energy_per_capita"],
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        key="tab1_trend"
    )

    # Aggregate: sum for flow variables, mean for per-capita/intensity
    per_capita_cols = ["energy_per_capita", "ghg_per_capita", "co2_per_unit_energy"]
    agg_func = "mean" if trend_metric in per_capita_cols else "sum"
    df_trend = (df_full.groupby("year")[trend_metric]
                .agg(agg_func).reset_index().dropna())

    fig_trend = px.area(
        df_trend, x="year", y=trend_metric,
        labels={"year": "Year", trend_metric: FEATURE_LABELS.get(trend_metric, trend_metric)},
        title=f"Global {FEATURE_LABELS.get(trend_metric, trend_metric)} — {agg_func.capitalize()} across all countries",
        color_discrete_sequence=["#1a6fa8"],
        template="plotly_white",
    )
    fig_trend.update_traces(fillcolor="rgba(26, 111, 168, 0.15)", line_color="#1a6fa8")
    fig_trend.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=30))
    st.plotly_chart(fig_trend, width="stretch")

    # ── GDP vs CO2 scatter ────────────────────────────────────────────────────
    st.markdown("#### GDP vs. CO₂ Emissions by Country")
    scatter_year = st.slider("Select year for scatter", 1990, 2022, 2015, key="tab1_scatter")
    df_scatter = df_full[df_full["year"] == scatter_year].dropna(
        subset=["gdp", "co2", "primary_energy_consumption"]
    )
    fig_scatter = px.scatter(
        df_scatter,
        x="co2", y="gdp",
        size="primary_energy_consumption",
        color="energy_per_capita",
        hover_name="country",
        hover_data={"co2": ":.1f", "gdp": ":.2e", "primary_energy_consumption": ":.1f"},
        labels={"co2": "CO₂ Emissions (Mt)", "gdp": "GDP (USD)",
                "primary_energy_consumption": "Primary Energy (TWh)",
                "energy_per_capita": "Energy/Capita (kWh)"},
        title=f"GDP vs. CO₂ in {scatter_year}  (bubble size = primary energy consumption)",
        log_x=True, log_y=True,
        template="plotly_white",
        color_continuous_scale="Viridis",
    )
    fig_scatter.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=30))
    st.plotly_chart(fig_scatter, width="stretch")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### Feature Correlation Matrix")
    corr_cols = ["gdp", "co2", "total_ghg", "methane", "nitrous_oxide",
                 "oil_co2", "primary_energy_consumption",
                 "energy_per_capita", "ghg_per_capita", "co2_per_unit_energy"]
    corr_present = [c for c in corr_cols if c in df_full.columns]
    corr_mat = df_full[corr_present].corr().round(2)

    fig_corr = px.imshow(
        corr_mat,
        text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels={"color": "Pearson r"},
        title="Pearson Correlations — Key Variables",
        template="plotly_white",
    )
    fig_corr.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_corr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Model Performance Explorer")
    st.markdown("""
    <div class="info-box">
    Three regression models predict <b>log(GDP)</b> from emissions and energy features.
    Use the controls below to toggle features on/off and observe how each model responds.
    All models are retrained live on the filtered feature set.
    </div>
    """, unsafe_allow_html=True)

    # ── Feature toggle ────────────────────────────────────────────────────────
    st.markdown("#### Select Features for Comparison")
    st.caption("Toggle features to see how model performance changes. All 9 features are selected by default.")

    # Lay checkboxes in a 3-column grid
    cols_grid = st.columns(3)
    selected_features = []
    for i, feat in enumerate(MODEL_FEATURES):
        with cols_grid[i % 3]:
            checked = st.checkbox(
                FEATURE_LABELS[feat],
                value=True,
                key=f"feat_{feat}"
            )
            if checked:
                selected_features.append(feat)

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    # ── Retrain on selected features ─────────────────────────────────────────
    @st.cache_data
    def retrain_models(feature_subset: tuple, _df_model):
        """
        Retrain all three models on the given feature subset.
        Tuple input (not list) so Streamlit can hash it for caching.
        """
        feat_list = list(feature_subset)
        X = _df_model[feat_list].values
        y = _df_model["gdp_log"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # OLS
        ols = LinearRegression()
        ols.fit(X_tr_sc, y_tr)
        pred_ols = ols.predict(X_te_sc)

        # Ridge
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        ridge.fit(X_tr_sc, y_tr)
        pred_ridge = ridge.predict(X_te_sc)

        # Random Forest (no scaling needed for trees)
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_te)

        results = {
            "OLS Regression":   {"y_test": y_te, "y_pred": pred_ols,   "color": "#2166AC"},
            "Ridge Regression": {"y_test": y_te, "y_pred": pred_ridge, "color": "#1A7A4A"},
            "Random Forest":    {"y_test": y_te, "y_pred": pred_rf,    "color": "#B2182B"},
        }

        # Feature importances (RF) and coefficients (OLS, Ridge)
        importance_ols   = np.abs(ols.coef_)
        importance_ridge = np.abs(ridge.coef_)
        importance_rf    = rf.feature_importances_

        return results, importance_ols, importance_ridge, importance_rf, feat_list

    results, imp_ols, imp_ridge, imp_rf, feat_list = retrain_models(
        tuple(selected_features), df_model
    )

    # ── Metrics comparison table ──────────────────────────────────────────────
    st.markdown("#### Model Metrics on Test Set")
    metric_rows = []
    for mname, res in results.items():
        m = metrics_dict(res["y_test"], res["y_pred"])
        metric_rows.append({"Model": mname, "R²": round(m["R²"], 4),
                             "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4)})
    df_metrics = pd.DataFrame(metric_rows)

    mc1, mc2, mc3 = st.columns(3)
    for col, row in zip([mc1, mc2, mc3], metric_rows):
        with col:
            st.markdown(f"**{row['Model']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²",   f"{row['R²']:.4f}")
            c2.metric("RMSE", f"{row['RMSE']:.4f}")
            c3.metric("MAE",  f"{row['MAE']:.4f}")

    # ── Actual vs Predicted scatter — all three models side-by-side ──────────
    st.markdown("#### Actual vs. Predicted log(GDP)")

    # Build a combined figure with 3 subplots
    fig_avp = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(results.keys()),
        shared_yaxes=True,
    )

    all_vals = np.concatenate([
        results[m]["y_test"] for m in results
    ] + [results[m]["y_pred"] for m in results])
    ax_min = float(all_vals.min()) - 0.5
    ax_max = float(all_vals.max()) + 0.5

    for col_idx, (mname, res) in enumerate(results.items(), start=1):
        y_t  = res["y_test"]
        y_p  = res["y_pred"]
        resid = y_p - y_t
        color = res["color"]

        # Scatter — coloured by residual
        fig_avp.add_trace(
            go.Scatter(
                x=y_t, y=y_p,
                mode="markers",
                marker=dict(
                    color=resid,
                    colorscale="RdBu_r",
                    size=4, opacity=0.55,
                    colorbar=dict(title="Residual", thickness=10) if col_idx == 3 else None,
                    showscale=(col_idx == 3),
                    cmid=0,
                ),
                hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>Residual: %{marker.color:.2f}",
                name=mname, showlegend=False,
            ),
            row=1, col=col_idx,
        )
        # Perfect-prediction line
        fig_avp.add_trace(
            go.Scatter(
                x=[ax_min, ax_max], y=[ax_min, ax_max],
                mode="lines",
                line=dict(color="#333333", dash="dash", width=1.5),
                name="y = x", showlegend=(col_idx == 1),
            ),
            row=1, col=col_idx,
        )
        # Trend line
        m_slope, b = np.polyfit(y_t, y_p, 1)
        x_line = np.array([ax_min, ax_max])
        fig_avp.add_trace(
            go.Scatter(
                x=x_line, y=m_slope * x_line + b,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Trend (β₁={m_slope:.2f})", showlegend=False,
            ),
            row=1, col=col_idx,
        )
        # Axes labels
        fig_avp.update_xaxes(title_text="Actual log(GDP)", range=[ax_min, ax_max],
                              row=1, col=col_idx)
        if col_idx == 1:
            fig_avp.update_yaxes(title_text="Predicted log(GDP)", row=1, col=col_idx)

    fig_avp.update_layout(
        height=420,
        template="plotly_white",
        title_text="Actual vs. Predicted — Coloured by Residual",
        margin=dict(l=20, r=20, t=60, b=30),
    )
    st.plotly_chart(fig_avp, width="stretch")

    # ── Feature importance comparison ─────────────────────────────────────────
    st.markdown("#### Feature Importance / Coefficient Magnitude")
    st.markdown("""
    <div class="warning-box">
    <b>Caution:</b> These represent <i>predictive importance</i>, not causal effects.
    Strong multicollinearity among emissions variables means individual coefficients
    should not be interpreted as independent causal impacts on GDP.
    </div>
    """, unsafe_allow_html=True)

    feat_labels_selected = [FEATURE_LABELS[f] for f in feat_list]

    fig_imp = go.Figure()
    # OLS absolute coefficients (scaled → comparable)
    imp_ols_norm   = imp_ols   / imp_ols.sum()
    imp_ridge_norm = imp_ridge / imp_ridge.sum()
    imp_rf_norm    = imp_rf    / imp_rf.sum()

    fig_imp.add_trace(go.Bar(
        name="OLS (|coef|, normalised)",
        y=feat_labels_selected, x=imp_ols_norm,
        orientation="h", marker_color="#2166AC", opacity=0.8,
    ))
    fig_imp.add_trace(go.Bar(
        name="Ridge (|coef|, normalised)",
        y=feat_labels_selected, x=imp_ridge_norm,
        orientation="h", marker_color="#1A7A4A", opacity=0.8,
    ))
    fig_imp.add_trace(go.Bar(
        name="Random Forest (Gini importance)",
        y=feat_labels_selected, x=imp_rf_norm,
        orientation="h", marker_color="#B2182B", opacity=0.8,
    ))
    fig_imp.update_layout(
        barmode="group",
        height=420,
        template="plotly_white",
        title="Normalised Feature Importance Across Models",
        xaxis_title="Normalised Importance",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_imp, width="stretch")

    # ── Residual distribution ─────────────────────────────────────────────────
    st.markdown("#### Residual Distribution")
    fig_resid = go.Figure()
    colors = {"OLS Regression": "#2166AC", "Ridge Regression": "#1A7A4A",
              "Random Forest": "#B2182B"}
    for mname, res in results.items():
        resid = res["y_pred"] - res["y_test"]
        fig_resid.add_trace(go.Histogram(
            x=resid, name=mname,
            opacity=0.6, nbinsx=60,
            marker_color=colors[mname],
        ))
    fig_resid.add_vline(x=0, line_dash="dash", line_color="#333333")
    fig_resid.update_layout(
        barmode="overlay",
        height=380,
        template="plotly_white",
        title="Distribution of Residuals (Predicted − Actual)",
        xaxis_title="Residual (log GDP units)",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_resid, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GDP PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.subheader("GDP Predictor")
    st.markdown("""
    <div class="info-box">
    Enter a country's emissions and energy profile using the sliders in the sidebar.
    The selected model will return a <b>point estimate</b> of log(GDP) alongside a
    <b>90% prediction interval</b> computed via MAPIE conformal prediction.
    All slider values are in natural (un-logged) units — the log transformation
    is applied automatically before passing inputs to the model.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar — model selection + feature sliders ───────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Prediction Controls")
        st.markdown("---")

        selected_model = st.selectbox(
            "Select Model",
            ["OLS Regression", "Ridge Regression", "Random Forest"],
            index=2,
            help="Random Forest has the highest predictive accuracy (R² ≈ 0.95).",
        )

        st.markdown("---")
        st.markdown("### 📥 Input Features")
        st.caption("Sliders use natural units. Log transforms are applied automatically.")

        # ── Sliders for log-transformed raw inputs ────────────────────────────
        co2_val = st.slider(
            "CO₂ Emissions (Mt)",
            min_value=0.0,
            max_value=float(feat_stats["co2"]["max"]),
            value=float(feat_stats["co2"]["median"]),
            step=0.5,
            help="Annual CO₂ emissions in million tonnes",
        )
        methane_val = st.slider(
            "Methane Emissions (Mt CO₂e)",
            min_value=0.0,
            max_value=float(feat_stats["methane"]["max"]),
            value=float(feat_stats["methane"]["median"]),
            step=0.1,
        )
        nitrous_val = st.slider(
            "Nitrous Oxide Emissions (Mt CO₂e)",
            min_value=0.0,
            max_value=float(feat_stats["nitrous_oxide"]["max"]),
            value=float(feat_stats["nitrous_oxide"]["median"]),
            step=0.05,
        )
        oil_co2_val = st.slider(
            "Oil CO₂ Emissions (Mt)",
            min_value=0.0,
            max_value=float(feat_stats["oil_co2"]["max"]),
            value=float(feat_stats["oil_co2"]["median"]),
            step=0.1,
        )
        pec_val = st.slider(
            "Primary Energy Consumption (TWh)",
            min_value=0.0,
            max_value=float(feat_stats["primary_energy_consumption"]["max"]),
            value=float(feat_stats["primary_energy_consumption"]["median"]),
            step=1.0,
        )
        ghg_val = st.slider(
            "Total GHG Emissions (Mt CO₂e)",
            min_value=0.0,
            max_value=float(feat_stats["total_ghg"]["max"]),
            value=float(feat_stats["total_ghg"]["median"]),
            step=0.5,
        )

        st.markdown("---")
        st.markdown("### 📥 Direct Features")
        st.caption("These features are not log-transformed.")

        co2_int_val = st.slider(
            "CO₂ Intensity (kg CO₂ per kWh)",
            min_value=float(feat_stats["co2_per_unit_energy"]["min"]),
            max_value=float(feat_stats["co2_per_unit_energy"]["max"]),
            value=float(feat_stats["co2_per_unit_energy"]["median"]),
            step=0.01,
            help="Carbon intensity of the energy supply",
        )
        energy_pc_val = st.slider(
            "Energy per Capita (kWh per person)",
            min_value=float(feat_stats["energy_per_capita"]["min"]),
            max_value=float(feat_stats["energy_per_capita"]["max"]),
            value=float(feat_stats["energy_per_capita"]["median"]),
            step=100.0,
        )
        ghg_pc_val = st.slider(
            "GHG per Capita (tCO₂e per person)",
            min_value=float(feat_stats["ghg_per_capita"]["min"]),
            max_value=float(feat_stats["ghg_per_capita"]["max"]),
            value=float(feat_stats["ghg_per_capita"]["median"]),
            step=0.1,
        )

        st.markdown("---")
        pi_alpha = st.selectbox(
            "Prediction Interval Coverage",
            [0.90, 0.80, 0.95],
            index=0,
            format_func=lambda x: f"{int(x*100)}%",
            help="Confidence level for the MAPIE prediction interval",
        )

    # ── Build the raw input DataFrame from sidebar values ─────────────────────
    raw_input = pd.DataFrame({
        "co2":                        [co2_val],
        "methane":                    [methane_val],
        "nitrous_oxide":              [nitrous_val],
        "oil_co2":                    [oil_co2_val],
        "primary_energy_consumption": [pec_val],
        "total_ghg":                  [ghg_val],
        "co2_per_unit_energy":        [co2_int_val],
        "energy_per_capita":          [energy_pc_val],
        "ghg_per_capita":             [ghg_pc_val],
    })

    # Build log-transformed feature DataFrame
    X_input = _build_feature_row(raw_input)
    # Scaler expects DataFrame with feature names; transform returns numpy array
    X_input_sc = pd.DataFrame(scaler.transform(X_input), columns=MODEL_FEATURES)

    # ── Point estimate from the cached (full-data) model ──────────────────────
    if selected_model == "OLS Regression":
        point_log = float(model_ols.predict(X_input_sc)[0])
    elif selected_model == "Ridge Regression":
        point_log = float(model_ridge.predict(X_input_sc)[0])
    else:
        point_log = float(model_rf.predict(X_input)[0])

    # Convert log(GDP) → GDP in USD
    point_gdp = np.expm1(point_log)

    # ── MAPIE prediction interval ─────────────────────────────────────────────
    with st.spinner("Computing prediction interval via MAPIE conformal prediction…"):
        pi_log_pt, pi_lower_log, pi_upper_log = get_prediction_interval(
            selected_model, raw_input, scaler, feat_stats, df_model,
            alpha=(1 - pi_alpha)
        )

    # Determine whether MAPIE was available
    mapie_available = pi_log_pt is not None

    if not mapie_available:
        # Fallback: ±1.96 × model RMSE on training data
        X_all_sc = scaler.transform(df_model[MODEL_FEATURES].values)
        y_all    = df_model["gdp_log"].values
        if selected_model == "OLS Regression":
            resid_std = np.sqrt(mean_squared_error(y_all, model_ols.predict(X_all_sc)))
        elif selected_model == "Ridge Regression":
            resid_std = np.sqrt(mean_squared_error(y_all, model_ridge.predict(X_all_sc)))
        else:
            resid_std = np.sqrt(mean_squared_error(
                y_all, model_rf.predict(df_model[MODEL_FEATURES].values)))
        z = {0.90: 1.645, 0.80: 1.282, 0.95: 1.960}[pi_alpha]
        pi_lower_log = point_log - z * resid_std
        pi_upper_log = point_log + z * resid_std

    pi_lower_gdp = np.expm1(pi_lower_log)
    pi_upper_gdp = np.expm1(pi_upper_log)

    # ── Output layout ─────────────────────────────────────────────────────────
    st.markdown(f"### Prediction — {selected_model}")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric(
            label="📌 Predicted GDP (USD)",
            value=f"${point_gdp:,.0f}",
            help="Point estimate from the selected model",
        )
    with res_col2:
        st.metric(
            label=f"⬇️ PI Lower ({int(pi_alpha*100)}%)",
            value=f"${pi_lower_gdp:,.0f}",
        )
    with res_col3:
        st.metric(
            label=f"⬆️ PI Upper ({int(pi_alpha*100)}%)",
            value=f"${pi_upper_gdp:,.0f}",
        )

    if mapie_available:
        st.caption(f"Prediction interval computed via MAPIE split conformal prediction "
                   f"at {int(pi_alpha*100)}% coverage level.")
    else:
        st.caption(f"⚠️ MAPIE not installed — showing ±{int(pi_alpha*100)}% normal approximation interval "
                   f"based on training RMSE. Install `mapie` for distribution-free intervals.")

    # ── Log-scale display ─────────────────────────────────────────────────────
    st.markdown("---")
    log_col1, log_col2, log_col3 = st.columns(3)
    log_col1.metric("Predicted log(GDP)", f"{point_log:.4f}")
    log_col2.metric("PI Lower log(GDP)",  f"{pi_lower_log:.4f}")
    log_col3.metric("PI Upper log(GDP)",  f"{pi_upper_log:.4f}")
    st.caption("Models operate in log(GDP) space. Values above are the raw model outputs before back-transformation.")

    # ── Prediction interval gauge chart ──────────────────────────────────────
    st.markdown("#### Prediction Interval Visualisation")

    # Use log scale for the chart (more interpretable range)
    all_gdp_log = df_model["gdp_log"]
    chart_min = max(float(all_gdp_log.quantile(0.01)) - 0.5, 0)
    chart_max = float(all_gdp_log.quantile(0.99)) + 0.5

    fig_pi = go.Figure()

    # Background: distribution of observed log(GDP) values
    fig_pi.add_trace(go.Violin(
        x=all_gdp_log,
        name="Observed log(GDP) distribution",
        line_color="#AAAAAA",
        fillcolor="rgba(180,180,180,0.15)",
        orientation="h",
        side="positive",
        width=1.2,
        showlegend=True,
        y0=0,
        points=False,
    ))

    # PI band
    fig_pi.add_shape(
        type="rect",
        x0=pi_lower_log, x1=pi_upper_log,
        y0=-0.4, y1=0.4,
        fillcolor="rgba(26, 111, 168, 0.20)",
        line=dict(color="#1a6fa8", width=1.5, dash="dot"),
    )
    # Point estimate line
    fig_pi.add_shape(
        type="line",
        x0=point_log, x1=point_log,
        y0=-0.6, y1=0.6,
        line=dict(color="#B2182B", width=3),
    )
    # Labels
    fig_pi.add_annotation(
        x=point_log, y=0.65,
        text=f"<b>Estimate: {point_log:.2f}</b>",
        showarrow=False, font=dict(color="#B2182B", size=12),
    )
    fig_pi.add_annotation(
        x=(pi_lower_log + pi_upper_log) / 2, y=-0.65,
        text=f"{int(pi_alpha*100)}% Prediction Interval",
        showarrow=False, font=dict(color="#1a6fa8", size=11),
    )

    fig_pi.update_layout(
        height=260,
        template="plotly_white",
        xaxis=dict(title="log(GDP)", range=[chart_min, chart_max]),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_pi, width="stretch")

    # ── Feature contribution waterfall (RF only) ──────────────────────────────
    if selected_model == "Random Forest":
        st.markdown("#### Feature Contributions (Random Forest Importances × Input)")
        st.caption("Illustrative decomposition: importance weight × normalised input value. "
                   "Not a SHAP decomposition — treat as directional, not exact.")

        importances  = model_rf.feature_importances_
        input_values = X_input.iloc[0].values  # .values converts DataFrame row → numpy array
        # Normalise inputs to [0,1] for visual scale
        X_all_arr = df_model[MODEL_FEATURES].values
        x_min = X_all_arr.min(axis=0)
        x_max = X_all_arr.max(axis=0)
        x_range = np.where(x_max - x_min > 0, x_max - x_min, 1)
        input_norm = (input_values - x_min) / x_range
        contributions = importances * input_norm

        contrib_df = pd.DataFrame({
            "Feature":      [FEATURE_LABELS[f] for f in MODEL_FEATURES],
            "Contribution": contributions,
        }).sort_values("Contribution", ascending=True)

        fig_contrib = px.bar(
            contrib_df, x="Contribution", y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale="Blues",
            template="plotly_white",
            title="Feature Contribution to Prediction",
            labels={"Contribution": "Importance × Normalised Input"},
        )
        fig_contrib.update_layout(
            height=380, margin=dict(l=20, r=20, t=50, b=30),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_contrib, width="stretch")

    # ── How does this compare to the training data? ────────────────────────────
    st.markdown("#### How Does This Prediction Compare to Historical Data?")

    # Find the 5 closest countries in the training set (Euclidean in feature space)
    X_all_arr = df_model[MODEL_FEATURES].values
    dists     = np.linalg.norm(X_all_arr - X_input.values, axis=1)  # .values → numpy for broadcast
    top5_idx  = np.argsort(dists)[:5]
    top5_df   = df_model.iloc[top5_idx][["country", "year", "gdp_log"]].copy()
    top5_df["gdp_usd"]      = np.expm1(top5_df["gdp_log"]).round(0)
    top5_df["distance"]     = dists[top5_idx].round(3)
    top5_df["gdp_log"]      = top5_df["gdp_log"].round(4)
    top5_df = top5_df.rename(columns={
        "country": "Country", "year": "Year",
        "gdp_log": "log(GDP)", "gdp_usd": "GDP (USD)", "distance": "Feature Distance"
    })

    st.markdown("**5 nearest historical observations to your input** (by Euclidean distance in feature space):")
    st.dataframe(top5_df.reset_index(drop=True), width="stretch")
