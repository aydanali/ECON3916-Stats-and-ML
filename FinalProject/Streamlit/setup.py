# ============================================================
# setup.py — Model Training & Artifact Generation
# GDP & Emissions Dashboard  |  World Bank Environmental Economics
# ============================================================
# Run this ONCE before launching the dashboard.
# It trains all three models on the cleaned OWID data and saves
# the .pkl and .csv files that app.py loads at runtime.
#
# Usage (Windows Command Prompt or PowerShell):
#   python setup.py
#
# Expected output files (all saved to the same folder as this script):
#   model_ols.pkl         OLS regression
#   model_ridge.pkl       Ridge regression (RidgeCV, alpha=100)
#   model_rf.pkl          Random Forest (200 trees)
#   scaler.pkl            StandardScaler fitted on training data
#   feature_stats.pkl     Slider min/median/max bounds for the dashboard
#   owid_clean_dash.csv   Full cleaned panel (Tab 1 data table)
#   owid_model_data.csv   Log-transformed model-ready subset (Tab 2 & 3)
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# Step 1 — Load the raw OWID CSV
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("GDP & Emissions Dashboard — Setup")
print("=" * 55)
print()
print("[1/5] Loading owid-co2-data.csv ...")

df_raw = pd.read_csv('https://raw.githubusercontent.com/owid/co2-data/refs/heads/master/owid-co2-data.csv')

print(f"  Raw data loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# Step 2 — Clean the data
# (Exact pipeline from owid_cleaning.py / notebook)
# ─────────────────────────────────────────────────────────────
print()
print("[2/5] Cleaning data ...")

START_YEAR              = 1990
END_YEAR                = 2024
COLUMN_NAN_THRESHOLD    = 0.50
COUNTRY_NAN_THRESHOLD   = 0.50
FOCUS_COLS = [
    "co2", "co2_per_capita", "primary_energy_consumption",
    "energy_per_capita", "energy_per_gdp", "gdp",
    "methane", "nitrous_oxide", "total_ghg",
    "coal_co2", "gas_co2", "oil_co2",
]

# Stage 1a: remove regional aggregates (no iso_code)
df = df_raw[df_raw["iso_code"].notna()].copy()

# Stage 1b: restrict year window
df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

# Stage 1c: drop columns with >50% missing
miss_share = df.isna().mean()
drop_cols  = [c for c in miss_share[miss_share > COLUMN_NAN_THRESHOLD].index
              if c not in ["country", "year", "iso_code"]]
df = df.drop(columns=drop_cols)

# Stage 2: drop countries with >50% missing on focus columns
focus_present = [c for c in FOCUS_COLS if c in df.columns]
country_miss  = (df.groupby("country")[focus_present]
                   .apply(lambda g: g.isna().to_numpy().mean()))
keep          = country_miss[country_miss <= COUNTRY_NAN_THRESHOLD].index
df            = df[df["country"].isin(keep)].copy()

# Stage 3: linear interpolation within each country's time series
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
print(f"  After cleaning: {len(df):,} rows | "
      f"{df['country'].nunique()} countries | "
      f"{df['year'].min()}–{df['year'].max()}")

# ── Column selection (matches notebook exactly) ───────────────
df = df[[
    "country", "year", "iso_code", "population", "gdp", "co2",
    "co2_per_unit_energy", "energy_per_capita", "ghg_per_capita",
    "methane", "nitrous_oxide", "oil_co2",
    "primary_energy_consumption", "total_ghg",
]]

# ── Second interpolation pass (matches notebook) ─────────────
interp_cols = [c for c in df.columns if c not in ["country", "year", "iso_code"]]
for col in interp_cols:
    df[col] = (df.groupby("country")[col]
                 .transform(lambda x: x.interpolate())
                 .ffill().bfill())

# ─────────────────────────────────────────────────────────────
# Step 3 — Feature engineering (log transforms)
# ─────────────────────────────────────────────────────────────
print()
print("[3/5] Engineering features ...")

df["gdp_log"]                        = np.log1p(df["gdp"])
df["co2_log"]                        = np.log1p(df["co2"])
df["methane_log"]                    = np.log1p(df["methane"])
df["nitrous_oxide_log"]              = np.log1p(df["nitrous_oxide"])
df["oil_co2_log"]                    = np.log1p(df["oil_co2"])
df["primary_energy_consumption_log"] = np.log1p(df["primary_energy_consumption"])
df["total_ghg_log"]                  = np.log1p(df["total_ghg"])

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
TARGET = "gdp_log"

# Drop rows missing any model feature or target
df_model = df[["country", "year"] + MODEL_FEATURES + [TARGET]].dropna()
print(f"  Model-ready rows: {len(df_model):,}")

# ─────────────────────────────────────────────────────────────
# Step 4 — Train/test split & model training
# ─────────────────────────────────────────────────────────────
print()
print("[4/5] Training models ...")

X = df_model[MODEL_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Fit scaler on training data only (never fit on test data)
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# Wrap in DataFrames to preserve feature names (avoids sklearn warnings)
X_train_sc_df = pd.DataFrame(X_train_sc, columns=MODEL_FEATURES)
X_test_sc_df  = pd.DataFrame(X_test_sc,  columns=MODEL_FEATURES)

# OLS regression (scaled input)
model_ols = LinearRegression()
model_ols.fit(X_train_sc_df, y_train)
y_pred_ols = model_ols.predict(X_test_sc_df)
print(f"  OLS    —  R²: {r2_score(y_test, y_pred_ols):.4f}  "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f}")

# Ridge regression (scaled input, cross-validated alpha)
ridge_model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
ridge_model.fit(X_train_sc_df, y_train)
y_pred_ridge = ridge_model.predict(X_test_sc_df)
print(f"  Ridge  —  R²: {r2_score(y_test, y_pred_ridge):.4f}  "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}  "
      f"alpha: {ridge_model.alpha_}")

# Random Forest (unscaled — trees don't need normalisation)
model_rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print(f"  RF     —  R²: {r2_score(y_test, y_pred_rf):.4f}  "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")

# ─────────────────────────────────────────────────────────────
# Step 5 — Save all artifacts
# ─────────────────────────────────────────────────────────────
print()
print("[5/5] Saving artifacts ...")

# Trained models
joblib.dump(model_ols,   "model_ols.pkl")
joblib.dump(ridge_model, "model_ridge.pkl")
joblib.dump(model_rf,    "model_rf.pkl")

# Scaler (must match what app.py uses for prediction)
joblib.dump(scaler, "scaler.pkl")

# Feature stats — drives slider min/max/default in the dashboard
RAW_LOG_COLS   = ["co2", "methane", "nitrous_oxide",
                   "oil_co2", "primary_energy_consumption", "total_ghg"]
DIRECT_COLS    = ["co2_per_unit_energy", "energy_per_capita", "ghg_per_capita"]

feat_stats = {}
for col in RAW_LOG_COLS + DIRECT_COLS:
    feat_stats[col] = {
        "min":    float(df[col].quantile(0.01)),
        "max":    float(df[col].quantile(0.99)),
        "median": float(df[col].median()),
    }
joblib.dump(feat_stats, "feature_stats.pkl")

# CSVs used by the dashboard tabs
df.to_csv("owid_clean_dash.csv",  index=False)   # Tab 1 full table
df_model.to_csv("owid_model_data.csv", index=False)  # Tab 2/3 model data

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Setup complete. Files saved:")
import os
for fname in ["model_ols.pkl", "model_ridge.pkl", "model_rf.pkl",
              "scaler.pkl", "feature_stats.pkl",
              "owid_clean_dash.csv", "owid_model_data.csv"]:
    size = os.path.getsize(fname)
    print(f"  {fname:<30s}  {size/1024:.1f} KB" if size < 1024**2
          else f"  {fname:<30s}  {size/1024**2:.1f} MB")
print()
print("Next step:")
print("  streamlit run app.py")
print("=" * 55)
