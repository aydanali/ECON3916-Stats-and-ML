## NY Fed Yield Curve Recession Model Replication

### Objective
Replicated the Federal Reserve Bank of New York's yield curve recession model by fitting a logistic regression on FRED macroeconomic data to forecast NBER-dated recessions 12 months ahead, validating the predictive power of the Treasury yield spread as a leading economic indicator.

---

### Methodology

- **Data sourcing:** Pulled two FRED series via the `fredapi` library — `T10Y3M` (10-Year minus 3-Month Treasury yield spread, daily frequency) and `USREC` (NBER recession indicator, monthly) — spanning 1970 to present. Resampled the daily yield spread to monthly frequency and applied a 12-month lag to align predictors with forward recession outcomes.

- **Benchmark comparison — Linear Probability Model:** Estimated an OLS regression of the recession indicator on the lagged yield spread to establish a baseline. Documented the LPM's fundamental flaw on real data: predicted probabilities that breach the [0, 1] interval, rendering the model logically incoherent for binary outcomes.

- **Logistic regression:** Replaced the LPM with a logistic regression (`scikit-learn`) to enforce probabilistic outputs. Fitted the characteristic S-curve mapping yield spread values to recession probabilities, confirming the model's theoretical and practical superiority over OLS for binary classification.

- **Inference via statsmodels:** Re-estimated the logit model using `statsmodels` to recover coefficient-level statistics — standard errors, z-scores, p-values, and 95% confidence intervals — which `scikit-learn` does not natively provide. Extracted the odds ratio for the yield spread with a bootstrapped confidence interval.

- **Extended specification:** Augmented the baseline model with the lagged unemployment rate as a second predictor to assess incremental explanatory power, comparing single- and two-predictor odds ratios across model specifications.

- **Temporal validation:** Applied `TimeSeriesSplit` cross-validation to respect the sequential structure of macroeconomic data and avoid look-ahead bias in model evaluation.

- **Recession probability time series:** Generated a full out-of-sample predicted probability series from 1970 to present, with particular focus on the contested 2022–2024 inversion episode.

---

### Key Findings

The logistic model reproduces the NY Fed's core result: a wider (more positive) yield spread is associated with significantly lower recession risk 12 months out, with an odds ratio of **0.557 [0.373, 0.830]** — statistically significant at the 1% level (p = 0.004). A flattening or inverted yield curve correspondingly elevates predicted recession probability, consistent with the model's long-run track record.

The Linear Probability Model generated predicted probabilities outside [0, 1] on observed data, providing a concrete, empirical demonstration of why OLS is inappropriate for binary outcomes — not merely a theoretical objection.

Adding the lagged unemployment rate as a second predictor attenuated the yield spread coefficient (OR: 0.454 → 0.557), consistent with partial correlation between the two series. Unemployment's own effect was directionally sensible but statistically insignificant (p = 0.150), suggesting the yield spread remains the dominant signal in this specification.

The 2022–2024 inversion period — the most severe yield curve inversion in decades — generated the highest model-implied recession probabilities in the sample. The absence of an NBER-dated recession during this window represents the most prominent false positive in the model's history, illustrating both the limits of single-indicator forecasting and the difficulty of applying historically-calibrated models to structurally unusual monetary policy environments.
