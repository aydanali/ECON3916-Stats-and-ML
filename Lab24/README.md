# Causal ML — Double Machine Learning for 401(k) Policy Evaluation

## Objective

Apply the Double Machine Learning framework to estimate the causal effect of 401(k) eligibility on net financial assets, using cross-fitted Random Forest nuisance learners to eliminate regularization bias and recover an unbiased Average Treatment Effect from observational data.

## Methodology

- **Regularization Bias Demonstration:** Simulated a data-generating process with a known true ATE of 5.0 to show that naive LASSO application shrinks the treatment coefficient toward zero, motivating the need for orthogonalization before causal estimation.

- **DoubleML Partially Linear Regression (PLR):** Estimated the ATE using the Partially Linear Regression formulation, which partials out the confounding influence of covariates (income, age, family size, marital status) from both the treatment and outcome equations before identifying the treatment effect.

- **Random Forest Nuisance Learners with 5-Fold Cross-Fitting:** Used Random Forest regressors and classifiers for the nuisance functions E[Y|X] and E[D|X], with cross-fitting across 5 folds to prevent overfitting the nuisance stage from contaminating the treatment effect estimate.

- **Conditional ATE (CATE) Analysis by Income Quartile:** Estimated heterogeneous treatment effects across income subgroups to assess whether the savings response to 401(k) eligibility varies systematically by income level, with confidence intervals constructed for each quartile.

- **Sensitivity Analysis:** Assessed robustness to unmeasured confounding using the DoubleML sensitivity framework (cf_y=0.03, cf_d=0.03), bounding how much an omitted variable such as financial literacy could plausibly explain.

## Key Findings

The DML PLR model estimated that 401(k) eligibility increases net financial assets by approximately **$[ATE] (95% CI: $[LB], $[UB])**, a precisely estimated effect that survived sensitivity analysis with a robustness value of [RV].

CATE analysis revealed [monotone increasing / non-monotone / relatively flat] treatment effect heterogeneity across income quartiles. The lowest income quartile showed an estimated effect of approximately $[Q1_CATE], rising to $[Q4_CATE] in the highest quartile — suggesting that [higher-income households derive greater savings benefit from eligibility / the policy effect is broadly uniform across the income distribution]. This pattern is consistent with [liquidity constraints limiting take-up at the lower end / a behavioral nudge that operates independently of income].

The naive LASSO benchmark substantially understated the treatment effect, confirming that regularization bias is a material concern in high-dimensional causal inference and that orthogonalization via cross-fitting is necessary to recover credible policy-relevant estimates.
