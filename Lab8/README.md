# Hypothesis Testing & Causal Evidence Architecture

## Objective

Most analytical workflows stop at *estimation* — producing a number and moving on. This project pivots deliberately to *falsification*: using the Lalonde (1986) experimental dataset as a testbed for adjudicating between competing causal narratives through the formal machinery of the scientific method.

The central question is not "what is the effect?" but rather "can we *reject* the claim that no effect exists?" — a subtle but consequential reframe that separates rigorous inference from sophisticated-looking data storytelling.

---

## Technical Approach

- **Parametric Testing via Welch's T-Test (SciPy):** Computed the Average Treatment Effect (ATE) of job training by treating the t-statistic as a signal-to-noise ratio — the estimated lift in real earnings ($re78$) scaled by pooled sampling variability. Welch's formulation was chosen over Student's T to avoid the assumption of equal variances across the treated and control groups, improving robustness without sacrificing interpretability.

- **Non-Parametric Validation via Permutation Test (SciPy / NumPy):** Ran 10,000 resamples under the sharp null hypothesis of no treatment effect, constructing an empirical reference distribution against which the observed ATE was benchmarked. This approach makes no distributional assumptions about earnings — a deliberate design choice given the well-documented right-skew and mass at zero in labor market income data.

- **Type I Error Control:** Significance thresholds were defined *a priori* at α = 0.05 before any test statistics were computed, enforcing a strict separation between the hypothesis-generation and hypothesis-testing phases. This discipline guards against p-hacking and ensures that the reported rejection of the null reflects genuine signal rather than an artifact of iterative threshold shopping.

---

## Key Findings

The analysis found a statistically significant lift in real earnings of approximately **$1,795** attributable to the job training intervention, with the result holding under both the parametric and non-parametric testing regimes. The null hypothesis of zero average treatment effect was rejected, providing convergent evidence of a positive causal impact.

---

## Business Insight: Hypothesis Testing as the Safety Valve of the Algorithmic Economy

Modern data infrastructure has made it trivially easy to generate correlations at scale. Any sufficiently large dataset, queried with sufficient creativity, will yield patterns — most of them noise. The proliferation of dashboards, A/B testing platforms, and ML feature pipelines has not solved this problem; it has industrialized it.

Rigorous hypothesis testing — pre-registered significance thresholds, non-parametric validation, explicit Type I error budgets — functions as the **safety valve** of this system. It is the mechanism by which data teams distinguish *discovered structure* from *imposed structure*, and it is what separates an insight that should drive resource allocation from one that will quietly reverse on the next cohort.

In practice, this matters most in high-stakes decisions: pricing algorithms, credit models, clinical trial endpoints, and policy interventions where a spurious correlation acted upon carries real downstream costs. The Lalonde dataset is a canonical example of how seemingly reasonable observational estimates can diverge dramatically from experimental ground truth — a reminder that the methodology used to generate a number is inseparable from the credibility of the number itself.

---

*Dataset: Lalonde, R.J. (1986). Evaluating the Econometric Evaluations of Training Programs with Experimental Data. American Economic Review.*
