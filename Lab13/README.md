# The Architecture of Dimensionality: Hedonic Pricing & the FWL Theorem

> *Econometrics Lab · ECON 3916 — Statistics and Machine Learning for Economics*
> *Dataset: 2026 California Real Estate Metrics (Zillow Synthetic) · Python 3.10+*

---

## Objective

Estimate a multivariate hedonic pricing model of California residential real estate and formally verify the Frisch-Waugh-Lovell (FWL) theorem by hand, demonstrating that partial regression residuals exactly recover the multivariate OLS coefficient — and that omitting a spatially correlated covariate systematically corrupts inference on every remaining regressor.

---

## Background

Hedonic pricing theory treats an asset's market value as a bundle of implicit prices paid for its constituent attributes. In residential real estate, buyers do not purchase square footage or proximity in isolation — they purchase a joint package of structural and locational amenities. Correctly decomposing these implicit prices requires that each regressor's contribution be estimated **net of all other included covariates**; failure to do so transfers their shared variance onto whichever attribute is left holding the bag.

The Frisch-Waugh-Lovell theorem formalizes this logic algebraically. It states that the OLS coefficient on any regressor $X_k$ in a multivariate model is numerically identical to the coefficient recovered by (1) regressing $X_k$ on all other regressors and extracting residuals $\tilde{X}_k$, and (2) regressing the outcome $Y$ on all other regressors and extracting residuals $\tilde{Y}$, then regressing $\tilde{Y}$ on $\tilde{X}_k$ alone. The partial regression has removed all shared variance with the control set, leaving only the orthogonal, *ceteris paribus* variation.

---

## Data

| Field | Description |
|---|---|
| `Sale_Price` | Residential transaction price (USD), outcome variable |
| `Property_Age` | Age of structure in years at time of sale |
| `Distance_to_Tech_Hub` | Euclidean distance (km) to nearest major technology employment center |

**Source:** Zillow synthetic 2026 California real estate metrics · *n* = 1,000 observations

---

## Methodology

### Step 1 — Multivariate OLS baseline

Estimated the full hedonic pricing model using `statsmodels.formula.api`:

$$\widehat{\text{Sale\_Price}} = \beta_0 + \beta_1 \cdot \text{Property\_Age} + \beta_2 \cdot \text{Distance\_to\_Tech\_Hub} + \varepsilon$$

`sm.add_constant` prepends an intercept column to the design matrix. Coefficients are extracted from `result.params`, a `pandas.Series` indexed by variable name, giving $\hat{\beta}_1$ and $\hat{\beta}_2$ as the multivariate partial effects.

### Step 2 — Bivariate OLS (omitted variable bias demonstration)

Re-estimated a restricted model excluding `Distance_to_Tech_Hub`:

$$\widehat{\text{Sale\_Price}} = \beta_0 + \beta_1^{*} \cdot \text{Property\_Age} + u$$

Compared $\hat{\beta}_1^{*}$ against the multivariate $\hat{\beta}_1$ to isolate the direction and magnitude of omitted variable bias.

### Step 3 — Manual FWL residualization

Applied the partialling-out procedure in two sequential OLS regressions:

1. Regressed `Property_Age` on `Distance_to_Tech_Hub` → extracted **Age residuals** ($\tilde{X}_1$): the component of property age orthogonal to proximity.
2. Regressed `Sale_Price` on `Distance_to_Tech_Hub` → extracted **Price residuals** ($\tilde{Y}$): the component of sale price orthogonal to proximity.

Both residual series are stored in the working dataframe as `Age_Residuals` and `Price_Residuals`.

### Step 4 — FWL verification regression

Regressed $\tilde{Y}$ (`Price_Residuals`) on $\tilde{X}_1$ (`Age_Residuals`) with no additional controls. The resulting coefficient was compared numerically to $\hat{\beta}_1$ from Step 1.

### Step 5 — 3D regression hyperplane visualization

Rendered an interactive 3D scatter plot of the full dataset alongside the fitted OLS hyperplane using `plotly.graph_objects`:

- **Surface construction:** `np.meshgrid` spans a 40×40 grid across the observed ranges of both predictors. The OLS equation $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 \cdot \text{Age} + \hat{\beta}_2 \cdot \text{Distance}$ is evaluated vectorially across all 1,600 grid nodes to produce the surface `z`-matrix.
- **Residual encoding:** Individual data points are colored on a diverging `RdBu` scale centered at zero — blue nodes sit above the hyperplane (positive residuals, under-predicted), red nodes sit below (negative residuals, over-predicted). This makes heteroscedasticity and structural non-linearity immediately visible against the fitted surface.
- **Partial slope interpretation:** The hyperplane's tilt along each axis represents the *ceteris paribus* effect: the Age slope is read holding Distance fixed at any constant cross-section; the Distance slope is read holding Age fixed. This geometry is the 3D analogue of what FWL proves algebraically.

**Stack:** `plotly.graph_objects`, `numpy`, `statsmodels`, `pandas`

---

## Key Findings

### Omitted variable bias — magnitude and direction

Excluding `Distance_to_Tech_Hub` from the pricing model produced a severely inflated coefficient on `Property_Age`. The bias is mechanically explained by the OVB formula:

$$\text{Bias} = \hat{\beta}_2^{\text{full}} \cdot \frac{\text{Cov}(\text{Age},\ \text{Distance})}{\text{Var}(\text{Age})}$$

In the California dataset, older properties tend to cluster closer to legacy urban tech corridors — a positive correlation between `Property_Age` and proximity. Because proximity commands a price premium (negative $\hat{\beta}_2$ in the full model), the restricted regression incorrectly *absorbs* this premium into the age coefficient, inflating $\hat{\beta}_1^{*}$ well beyond its true partial effect. The structural characteristic of the home was being credited for value generated entirely by its location.

### FWL exact numerical recovery

After residualizing both `Property_Age` and `Sale_Price` on `Distance_to_Tech_Hub`, the simple bivariate regression of `Price_Residuals` on `Age_Residuals` returned a coefficient **numerically identical** to $\hat{\beta}_1$ from the multivariate model, confirming the FWL theorem to floating-point precision.

This result proves that OLS does not merely *adjust* for covariates — it geometrically projects $Y$ and $X_k$ onto the orthogonal complement of the control space before computing their relationship. The multivariate estimator is, in effect, running dozens of these partialling operations simultaneously, one for each included regressor.

### Algorithmic ceteris paribus

The FWL procedure is the algebraic implementation of *ceteris paribus* reasoning. Once `Distance_to_Tech_Hub` variance is stripped from both sides of the regression, the remaining variation in `Age_Residuals` is, by construction, uncorrelated with proximity. Any movement in `Price_Residuals` explained by `Age_Residuals` can therefore be attributed solely to age — not to the confound. The multivariate OLS estimator automates this orthogonalization internally via the annihilator matrix $M_{X_{-k}} = I - X_{-k}(X_{-k}^\top X_{-k})^{-1} X_{-k}^\top$.

---

## Stack

```
Python 3.10+
├── pandas               — data wrangling and residual storage
├── statsmodels          — OLS estimation, coefficient extraction, model diagnostics
│   └── formula.api      — R-style formula interface for readable model specs
├── matplotlib           — static diagnostic plots
└── plotly.graph_objects — interactive 3D regression hyperplane visualization
```

---

## Repo Structure

```
.
├── README.md
├── data/
│   └── Zillow_California_2026_Hedonic.csv
├── notebooks/
│   └── StatsML_Lab13.ipynb
└── outputs/
    └── regression_3d_hyperplane.html
```

---

*Lab completed as part of ECON 3916 — Statistics and Machine Learning for Economics, Northeastern University.*
