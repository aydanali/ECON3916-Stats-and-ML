# High-Dimensional GDP Growth Forecasting with Lasso and Ridge Regularization

## Objective
Forecast five-year average GDP per capita growth across 120+ countries using World Development Indicators, using OLS overfitting as a diagnostic baseline to motivate Lasso and Ridge regularization with cross-validated hyperparameter selection.

---

## Methodology

- **Data Acquisition:** Downloaded World Bank WDI data live via the `wbgapi` API, covering 36 macroeconomic, trade, education, health, infrastructure, finance, governance, and natural resource indicators across ~150 countries over 2013–2019. Target variable constructed as the country-level five-year average GDP per capita growth rate.

- **Baseline OLS Estimation:** Fit an unregularized OLS model to establish a performance ceiling on training data and expose variance inflation at a high p/n ratio — where the number of predictors approaches the number of observations and the model is free to overfit noise.

- **Regularization with Cross-Validated Lambda Selection:** Applied `RidgeCV` and `LassoCV` from scikit-learn, using k-fold cross-validation to select the penalty parameter λ that minimizes out-of-sample prediction error. Features were standardized prior to fitting to ensure penalty comparability across indicators with different scales.

- **Lasso Path Analysis:** Traced the regularization path using `lasso_path` to visualize the sequence in which predictors enter the model as λ decreases — identifying which indicators carry signal robust enough to survive aggressive penalization.

---

## Key Findings

OLS achieved a training R² approaching 1.0 while posting a substantially lower test R², a textbook illustration of variance explosion under high-dimensional estimation. Both RidgeCV and LassoCV meaningfully closed the train-test gap, confirming that regularization recovered out-of-sample generalizability at modest cost to in-sample fit.

Lasso produced a sparse model retaining roughly 5–10 predictors from the full 30+ indicator set, with the Lasso path revealing which development indicators carry the most robust cross-country growth signal regardless of what else is controlled for. Ridge, by contrast, retained all predictors with shrunk coefficients, distributing explanatory weight across correlated indicators rather than forcing selection.

---

## Stack

| Tool | Purpose |
|---|---|
| `wbgapi` | Live World Bank WDI data retrieval |
| `scikit-learn` | OLS, RidgeCV, LassoCV, lasso_path, StandardScaler |
| `pandas` / `numpy` | Data wrangling and feature construction |
| `matplotlib` | Lasso path and R² diagnostic visualizations |
