# ECON 3916 — Statistics & Machine Learning for Economics
## Assignment 3: Bootstrap Inference, Permutation Testing & Propensity Score Matching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydanali/ECON3916-Stats-and-ML/blob/main/Assignment%203/Econ_3916_Assignment_3.ipynb)

---

## Overview

This notebook applies three core techniques from causal inference and computational statistics to real and simulated data. The assignment is structured across four phases, each building on key econometric concepts: non-parametric bootstrap inference, permutation-based hypothesis testing, and propensity score matching (PSM) with a Love Plot diagnostic.

---

## Repository Structure

```
Assignment 3/
├── Econ_3916_Assignment_3.ipynb   # Main notebook
├── swiftcart_loyalty.csv          # Observational dataset (Google Drive)
└── README.md
```

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

All phases run in Google Colab. Phase 3 requires the `swiftcart_loyalty.csv` dataset to be present in your Google Drive root directory.

---

## Phases

### Phase 1 — Bootstrap Confidence Interval for Driver Tips

**Objective:** Estimate a 95% confidence interval for the median tip amount using non-parametric bootstrapping, and compare it to a normal parametric interval.

**Data Generating Process:**
- 100 drivers receive zero tip
- 150 drivers receive a tip drawn from an Exponential distribution (λ = 5.0)
- Combined into a single `driver_tips` array (n = 250)

**Method:** 10,000 bootstrap resamples are drawn with replacement. The 2.5th and 97.5th percentiles of the resampled medians form the bootstrap confidence interval. A parametric interval using the sample mean ± 1.96 standard deviations is computed for comparison.

**Key Results:**

| Interval Type | Lower Bound | Upper Bound |
|---|---|---|
| Bootstrap (percentile) | 0.282 | 1.364 |
| Parametric (normal) | 0.230 | 1.312 |

**Interpretation:** The bootstrap interval is asymmetric — the upper half is wider than the lower half — consistent with right-skew in the underlying tip distribution. The parametric interval assumes symmetry and therefore underestimates the upper tail. This asymmetry is a key advantage of non-parametric bootstrap methods when the population distribution is unknown or non-normal.

---

### Phase 2 — Permutation Test: Delivery Time Distributions

**Objective:** Test whether two groups (control vs. treatment) have statistically different mean delivery times using a permutation test, without assuming any parametric distribution.

**Data:**
- Control group: 1,000 delivery times drawn from a Normal distribution (μ = 35, σ = 5)
- Treatment group: 1,000 delivery times drawn from a Log-Normal distribution (μ_log = 3.4, σ_log = 0.4)

**Method:** The observed difference in means is computed. Then, 5,000 permutations randomly shuffle group labels and recompute the group difference each time. The p-value is calculated using the conservative formula:

```
p = (r + 1) / (n + 1)
```

where r is the number of permuted differences as extreme or more extreme than the observed difference, and n is the total number of permutations.

**Key Results:**

| Statistic | Value |
|---|---|
| Observed difference in means | 3.053 |
| Permutation p-value | 0.0002 |
| Extreme values in null distribution | 0 |

**Interpretation:** The p-value of 0.0002 provides very strong evidence against the null hypothesis of equal means. Not a single permuted difference matched or exceeded the observed difference, indicating the two delivery time distributions are statistically distinct. The permutation test is appropriate here because the log-normal treatment group violates the normality assumption required for a standard t-test.

---

### Phase 3 — Propensity Score Matching: SwiftCart Loyalty Program

**Objective:** Estimate the causal effect of SwiftCart's loyalty subscription on post-period customer spending, correcting for selection bias using Propensity Score Matching (PSM).

**Dataset:** `swiftcart_loyalty.csv` — 8,941 customers

| Column | Description |
|---|---|
| `subscriber` | Binary treatment indicator (1 = subscriber, 0 = non-subscriber) |
| `pre_spend` | Pre-period spending ($) |
| `account_age` | Account age in months |
| `support_tickets` | Number of support interactions |
| `post_spend` | Outcome variable — post-period spending ($) |

**Method:**

1. **Naive SDO (Simple Difference in Outcomes):** Subtract the mean post-spend of non-subscribers from that of subscribers. This estimate ignores confounding.

2. **Propensity Score Estimation:** A logistic regression is fit on the three covariates (`pre_spend`, `account_age`, `support_tickets`) to predict the probability of subscribing for each customer.

3. **1-Nearest-Neighbor Matching:** Each treated customer (subscriber) is matched to the control customer with the closest propensity score using scikit-learn's `NearestNeighbors`. The matched pairs are used to compute the Average Treatment Effect on the Treated (ATT).

**Key Results:**

| Estimator | Estimate |
|---|---|
| Naive difference in means (SDO) | $17.57 |
| ATT via Propensity Score Matching | $9.91 |

**Interpretation:** The naive SDO of $17.57 overstates the causal effect because subscribers are systematically different from non-subscribers — they tend to be higher spenders and more engaged customers to begin with. Once matched on observable pre-treatment characteristics, the estimated causal effect of the loyalty subscription drops to $9.91. This result calls into question the company's initial claim of a 300% spending increase, which was based on unadjusted comparisons. The ATT is the appropriate estimand when the policy question concerns the effect on those who actually chose to subscribe.

---

### Phase 4 — Love Plot: Covariate Balance Diagnostics

**Objective:** Visually assess covariate balance before and after PSM using a publication-ready Love Plot of Standardised Mean Differences (SMD).

**SMD Formula (Austin, 2009 pooled-variance normalisation):**

```
SMD = |μ₁ − μ₀| / √[(s₁² + s₀²) / 2]
```

An |SMD| ≤ 0.10 is the standard threshold for adequate covariate balance (Stuart, 2010; Ho et al., 2007).

**Functions:**

`compute_smd(df, treatment_col)` — Computes the absolute SMD for every covariate column in the DataFrame.

`love_plot(df_unmatched, df_matched, ...)` — Generates a styled dot plot showing pre- and post-match SMDs for all covariates, a threshold line, connecting arrows, and a summary statistics footer.

`balance_summary(df_unmatched, df_matched, ...)` — Returns a tidy DataFrame with `smd_before`, `smd_after`, `reduction_pct`, and `balanced` columns.

**`love_plot()` Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `treatment_col` | `"treatment"` | Name of binary treatment column |
| `threshold` | `0.10` | Balance threshold line |
| `sort_by` | `"unmatched"` | Sort rows by `"unmatched"`, `"matched"`, or `"covariate"` |
| `save_path` | `None` | File path to save the figure |
| `dpi` | `180` | Output resolution |

**How to use with your own matched data:**

```python
fig = love_plot(
    df_unmatched,   # Full observational DataFrame
    df_matched,     # Matched DataFrame after PSM
    treatment_col="subscriber",
    threshold=0.10,
    sort_by="unmatched",
    save_path="love_plot.png"
)
```

**What constitutes conclusive visual evidence of bias mitigation:**

1. **Universal leftward shift** — Every post-match point (blue diamond) lies strictly left of its pre-match point (red circle). Rightward movement indicates matching introduced imbalance.
2. **All post-match points cross the threshold** — Every |SMD| after matching falls below 0.10. Covariates remaining to the right violate the Conditional Independence Assumption (CIA).
3. **No drift on pre-balanced covariates** — Variables already near-balanced before matching should not worsen after matching.
4. **Mean |SMD| reduction ≥ 80%, post-match mean ≤ 0.05** — The summary footer reports both statistics.
5. **Binary covariates balanced** — Dummy variables must satisfy the threshold alongside continuous covariates.

---

## Methods Summary

| Phase | Technique | Key Output |
|---|---|---|
| 1 | Bootstrap percentile CI | Asymmetric 95% CI for median tip |
| 2 | Permutation test | p = 0.0002; reject equal means |
| 3 | Propensity Score Matching (1-NN) | ATT = $9.91 vs. naive SDO = $17.57 |
| 4 | Love Plot (SMD) | Visual covariate balance diagnostic |

---

## References

- Austin, P.C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. *Statistics in Medicine*, 28(25), 3083–3107.
- Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2007). Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. *Political Analysis*, 15(3), 199–236.
- Stuart, E.A. (2010). Matching methods for causal inference: A review and a look forward. *Statistical Science*, 25(1), 1–21.

---

*ECON 3916 — Statistics and Machine Learning for Economics | Northeastern University*
