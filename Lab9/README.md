# Recovering Experimental Truths via Propensity Score Matching

## Objective
Applied Propensity Score Matching (PSM) to the observational Lalonde dataset to eliminate selection bias and recover a credible causal estimate of the Average Treatment Effect on the Treated (ATT) — demonstrating that rigorous quasi-experimental design can rehabilitate observational data when randomized control is unavailable.

## Methodology
- **Selection Bias Modeling:** Characterized the confounding structure of the observational sample by identifying systematic differences in pre-treatment covariates (age, education, earnings history, race, marital status) between treated and control units.
- **Propensity Score Estimation:** Fit a logistic regression model to estimate each unit's conditional probability of treatment assignment given observed covariates, collapsing the multi-dimensional covariate space into a single balancing score.
- **Nearest-Neighbor Matching:** Implemented a nearest-neighbor matching algorithm to pair each treated unit with its closest control counterpart in propensity score space, constructing a comparison group that approximates the covariate balance of a randomized experiment.
- **Balance Validation:** Evaluated post-match covariate overlap to confirm that the matched sample satisfied the common support assumption required for unconfounded inference.

*Stack: Python, Pandas, Scikit-Learn*

## Key Findings

The naive comparison of means on the raw observational data produced a treatment effect estimate of **−$15,204** — a result so severely biased by negative selection that it implies the job training program *harmed* participants' earnings. This is a textbook failure of unadjusted observational inference.

After propensity score matching and covariate rebalancing, the recovered ATT converged to approximately **+$1,800**, closely aligning with the benchmark experimental estimate from the Lalonde RCT. The delta between the two estimates — a swing of over **$17,000** — illustrates the magnitude of confounding present in the raw observational sample and the power of matching methods to correct it.

> **Takeaway:** Selection bias, left unaddressed, can not only attenuate a treatment effect but reverse its sign entirely. PSM, applied carefully, restores the signal buried beneath that noise.
