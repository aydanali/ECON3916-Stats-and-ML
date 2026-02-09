# Lab 5: The Architecture of Bias

## Overview
Investigated the Data Generating Process (DGP) and sampling bias mechanisms in machine learning pipelines. This project demonstrates how biased sampling compromises model validity and provides forensic techniques to detect and mitigate these issues in production systems.

## Technical Implementation

**Tech Stack:** Python, pandas, numpy, scipy, scikit-learn

### 1. Simple Random Sampling Analysis
Manually simulated simple random sampling on the Titanic dataset to quantify sampling error and variance. Demonstrated that while unbiased in expectation, simple random sampling produces high variance in class distributions across bootstrap iterations, particularly problematic for imbalanced datasets.

### 2. Stratified Sampling for Covariate Shift Elimination
Implemented stratified sampling using `sklearn.model_selection.StratifiedShuffleSplit` to preserve class proportions across train/test splits. This approach eliminates covariate shift by ensuring the empirical distribution P(X) in the sample matches the population distribution, critical for maintaining model calibration and preventing distribution mismatch between training and deployment.

### 3. Sample Ratio Mismatch (SRM) Forensic Audit
Conducted Chi-Square goodness-of-fit tests to detect Sample Ratio Mismatch in A/B test assignments. Applied statistical hypothesis testing (α = 0.01) to distinguish between natural sampling variance and systematic engineering failures in randomization mechanisms (e.g., load balancer bugs, broken assignment logic).

**Key Result:** A 550/450 split in a planned 50/50 experiment yields p = 0.0016, indicating a <0.2% probability of occurring under correct randomization—strong evidence of infrastructure failure rather than random chance.

## Theoretical Deep Dive: Survivorship Bias in Unicorn Analysis

### The Problem
Analyzing only successful unicorn startups featured on TechCrunch creates **survivorship bias**—a selection bias where the sample systematically excludes failures. This violates the random sampling assumption: P(Selected | Success) ≠ P(Selected | Failure).

**Why this matters:** Any regression of "success factors" (funding rounds, team size, market timing) on observed unicorns will produce **upward-biased coefficients**. You're fitting a model on E[Y | Y > threshold, X] rather than E[Y | X], making it impossible to distinguish causation from post-hoc rationalization.

### The Ghost Data: What's Missing
To apply a **Heckman Correction** (two-stage selection model), you need data on the **selection mechanism itself**:

1. **Failed startups with comparable observables** - Companies that raised seed funding, had similar team credentials, targeted similar markets, but failed to achieve unicorn status or media coverage

2. **Selection equation covariates** (Z variables) - Factors that predict media coverage but don't directly cause success:
   - PR budget and media relationships
   - Founder social media presence/personal brand
   - Geographic proximity to tech journalism hubs (SF, NYC)
   - "Buzz-worthy" narrative (contrarian thesis, celebrity investors)

3. **Binary selection indicator** - Whether each startup (successful or not) received TechCrunch coverage

### How Heckman Fixes It
**Stage 1:** Estimate P(TechCrunch coverage | Z, X) using a probit model on the full population (survivors + failures)

**Stage 2:** Include the Inverse Mills Ratio (λ) from Stage 1 as a regressor in your outcome equation to control for selection bias:
```
E[Valuation | X, Selected] = βX + γλ + ε
```

The λ term captures the correlation between unobserved factors affecting both selection (media coverage) and outcomes (unicorn status), correcting the bias.

**Bottom line:** Without data on failed startups and their selection probabilities, any causal claims about "what makes unicorns successful" are fundamentally unidentified—you're reverse-engineering survivorship, not discovering alpha.

## Applications
- **A/B Testing QA:** Automated SRM detection in experimentation platforms
- **Model Monitoring:** Covariate shift detection in production ML systems
- **Causal Inference:** Selection bias correction in observational studies
- **Survey Design:** Stratified sampling for representative population inference

---
*This lab reinforces that data quality begins at the sampling layer—no amount of sophisticated modeling can recover from biased data collection.*
