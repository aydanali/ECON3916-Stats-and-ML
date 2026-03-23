# AI Capex Diagnostic Modeling

## Objective
Diagnose and correct structural failures in an OLS regression model predicting AI Software Revenue from capital expenditure and deployment metrics, using 2026 Nvidia AI Capex and Deployment data to expose the inferential distortions introduced by heteroscedasticity.

---

## Methodology

- **Baseline OLS Estimation** — Fit a naive Ordinary Least Squares model regressing AI Software Revenue on capital expenditure and deployment covariates; extracted residuals and fitted values for diagnostic analysis.

- **Heteroscedasticity Detection** — Constructed residual-vs-fitted plots and scale-location plots to visually characterize error variance structure; identified a pronounced fan-shaped expansion in residuals at high capex tiers, consistent with multiplicative heteroscedasticity.

- **Multicollinearity Audit** — Computed Variance Inflation Factors (VIFs) across all regressors to assess collinearity pressure within the covariate matrix; flagged predictors with elevated VIF scores as potential confounds to coefficient stability.

- **HC3 Robust Correction** — Re-estimated the model using Heteroscedasticity-Consistent (HC3) standard errors via `statsmodels`; HC3 applies leverage-adjusted residual weighting, making it the preferred estimator under unknown, non-constant error variance — particularly in finite samples with influential observations.

- **Inferential Comparison** — Conducted a side-by-side diagnostic comparison of naive OLS vs. HC3-corrected p-values and standard errors to quantify the degree of false precision introduced by the uncorrected model.

---

## Key Findings

The naive OLS model exhibited **severe heteroscedasticity expanding monotonically with capital expenditure**, a structural pattern consistent with proportional error variance in high-investment deployment environments. This violated the Gauss-Markov assumption of homoscedastic disturbances, causing the model to systematically **understate standard errors** and generate artificially low p-values — inflating apparent statistical significance for several deployment metrics.

Applying HC3 robust standard errors corrected this distortion: confidence intervals widened appropriately, and the adjusted significance levels provided a materially more conservative — and credible — picture of which deployment covariates carry genuine predictive weight on AI Software Revenue. The exercise demonstrates that model diagnostics are not a formality; in high-variance, capital-intensive data environments, skipping them actively misleads inference.

---

**Stack:** Python · pandas · statsmodels · matplotlib · seaborn
