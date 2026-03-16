# Architecting the Prediction Engine
### Hedonic Pricing OLS · Residual Forensics Dashboard
*Zillow ZHVI 2026 Micro Dataset · Python · statsmodels · Plotly*

---

## Objective

Engineer a multivariate OLS prediction system on cross-sectional real estate data to minimize out-of-sample forecast error and quantify algorithmic valuation risk in dollar terms — then surface model pathologies through an interactive residual forensics dashboard.

---

## Stack

| Layer | Tools |
|---|---|
| Data Wrangling | `pandas`, `numpy` |
| Modeling | `statsmodels`, Patsy Formula API |
| Visualization | `plotly.express` |
| Dataset | Zillow ZHVI 2026 Micro Dataset (cross-sectional) |

---

## Methodology

**1. Data & Feature Engineering**
- Sourced the Zillow ZHVI 2026 Micro Dataset as the cross-sectional base, capturing a modern real estate market snapshot across property characteristics, location signals, and pricing outcomes
- Features: `Square_Footage`, `Property_Age`, `Distance_to_Transit` (continuous); `School_District_Rating` (categorical — one-hot encoded, baseline level dropped to satisfy the dummy variable constraint)

**2. Model Specification & Estimation**
- Constructed and fit a multivariate OLS model using `statsmodels` and the Patsy Formula API, translating hedonic pricing theory into a structured prediction specification
- Shifted the analytical frame from classical inference (explaining coefficients) to predictive engineering (minimizing loss), treating model performance as an operational business question rather than a statistical one

**3. Loss Quantification**
- Evaluated predictive performance via Root Mean Squared Error (RMSE), computed directly in US dollars to anchor error magnitude in financially interpretable units — expressing prediction uncertainty as a business risk instrument rather than an abstract fit statistic

**4. Residual Forensics Dashboard**
- Extracted `result.fittedvalues` (ŷ) and `result.resid` (ε = y − ŷ) directly from the `statsmodels` results object post-estimation
- Built an interactive scatter plot of fitted values (x-axis) vs. residuals (y-axis) using `plotly.express`
- Rendered a horizontal zero-reference line (ε = 0) and ±2σ threshold bands to frame the residual distribution
- Flagged outliers exceeding ±2σ in stark crimson; standard observations rendered in steel blue
- Enriched hover tooltips with raw feature values (`Square_Footage`, `Property_Age`, `Distance_to_Transit`, `School_District_Rating`, `Actual Home Value`) for observation-level forensic inspection

---

## Key Findings

The model successfully operationalized the transition from explanatory regression to a deployable prediction engine. Expressing RMSE in nominal dollar terms produced a precise financial error margin — the average dollar deviation between algorithmic valuation and realized market price — directly framing the model's output as a business risk instrument.

The residual forensics dashboard extended this analysis from summary statistics to structural diagnostics. By plotting ε against ŷ, the visualization exposes three classes of model pathology:

- **Heteroscedasticity** — a fanning residual cloud (widening spread as ŷ increases) signals non-constant error variance, common in hedonic models where high-value properties carry greater pricing noise
- **Structural breaks** — discontinuous vertical shifts in the residual band at a ŷ threshold indicate the model is misspecified across market segments (e.g., a luxury vs. starter-home boundary where pricing mechanics diverge)
- **Nonlinearity** — a curved or arch-shaped residual pattern suggests a missing quadratic term or interaction, with the model systematically over- or under-predicting across the fitted range

Crimson outliers (|ε| > 2σ) flagged for manual review; clustering on one side of the zero-line would indicate omitted variable bias rather than random noise.

---

## Repository Structure

```
.
├── hedonic_ols.py                  # OLS model estimation + RMSE
├── residual_forensics.py           # Interactive dashboard
└── README.md
```
