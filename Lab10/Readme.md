# Spurious Correlation & Multicollinearity in Macroeconomic Time-Series Data

> **Data Science Portfolio Project** · Economics & Econometrics

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=flat)
![FRED API](https://img.shields.io/badge/FRED_API-C8102E?style=flat)

---

## Overview

This project investigates a critical but often underappreciated pitfall in applied econometrics: the systematic inflation of correlation coefficients in raw macroeconomic level data. Using the FRED API to source real U.S. macroeconomic time-series — CPI, Unemployment, the Federal Funds Rate, Industrial Production, and M2 Money Supply — the analysis proceeds through three methodological stages to expose, quantify, and structurally map spurious relationships in multivariate macroeconomic data.

---

## Methodology

### Stage 1 — Exposing Correlation Traps in Raw Level Data

Raw level data for all five variables was retrieved via the FRED API and loaded into a pandas DataFrame. Seaborn heatmaps were used to visualize the pairwise Pearson correlation matrix. As expected under non-stationarity, the raw levels produced near-perfect correlations between trending series (e.g., CPI and M2 approaching *r* = 0.98) — a canonical illustration of the Granger-Newbold spurious regression problem. These inflated correlations reflect shared deterministic trends rather than genuine structural co-movement.

### Stage 2 — Quantifying Redundancy via VIF Diagnostics

Variance Inflation Factors (VIF) were computed using the `variance_inflation_factor` utility from `statsmodels` to formally quantify multicollinearity across the regressor set. VIF scores exceeding conventional thresholds (VIF > 10) confirmed severe redundancy among the raw-level predictors, establishing that OLS estimates would be unreliable and standard errors inflated. This diagnostic step provided quantitative grounding for the transformation approach in Stage 3.

### Stage 3 — Stationarity Correction and Structural Identification via DAGs

Non-stationary level variables were transformed into Year-over-Year (YoY) percentage growth rates via `pct_change(periods=12)`, rendering each series approximately stationary and removing shared trending behavior. The re-estimated correlation heatmap demonstrated a marked reduction in inflated coefficients. Directed Acyclic Graphs (DAGs) were then constructed to represent hypothesized structural causal relationships, distinguishing direct causal channels from confounded paths and clarifying which observed correlations reflect genuine economic mechanisms.

---

## Key Findings

- **Raw level correlations** between CPI and M2 exceeded *r* = 0.95, driven entirely by shared upward trends rather than structural linkage.
- **VIF diagnostics** returned scores well above 10 for multiple predictors, confirming that any regression on raw levels would yield severely unreliable coefficient estimates.
- **YoY transformation** reduced most cross-variable correlations toward structurally plausible magnitudes, with economically meaningful relationships (e.g., Fed Funds and CPI inflation) surviving and sharpening post-transformation.
- **DAG analysis** revealed that several apparent correlations in the raw data were d-separation violations — paths open only due to shared common causes, not direct causal effects.

---

## Tools & Stack

| Layer | Tools |
|---|---|
| Data Retrieval | FRED API (Federal Reserve Bank of St. Louis) |
| Data Manipulation | Python, pandas |
| Visualization | Seaborn (static heatmaps), Plotly (interactive correlation dashboard) |
| Econometric Diagnostics | `statsmodels` — `variance_inflation_factor` |
| Causal Inference | Directed Acyclic Graphs (DAGs) for structural identification |
| Transformation | YoY growth rates via `pct_change(periods=12)` |

---

## Project Structure

```
├── data/
│   └── fred_data.csv            # Raw FRED series
├── notebooks/
│   ├── 01_raw_correlations.ipynb
│   ├── 02_vif_diagnostics.ipynb
│   └── 03_yoy_dags.ipynb
├── visuals/
│   ├── heatmap_raw_levels.png
│   ├── heatmap_yoy_rates.png
│   └── fred_correlation_heatmap.html
└── README.md
```

---

## References

- Granger, C.W.J. & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Federal Reserve Bank of St. Louis. FRED Economic Data. https://fred.stlouisfed.org
