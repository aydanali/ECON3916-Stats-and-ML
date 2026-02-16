# ECON 3916 Assignment 2: Statistical Bias Detection & A/B Testing Audit

This assignment explores critical concepts in statistics and machine learning through hands-on data analysis and simulation. The project demonstrates practical applications of robust statistics, Bayesian inference, hypothesis testing, and survivorship bias.

## Overview

This notebook contains four phases of analysis that showcase real-world applications of statistical methods:

1. **Phase 1**: Robust Statistics - Median Absolute Deviation (MAD) vs Standard Deviation
2. **Phase 2**: Bayesian Inference - Plagiarism Detection and Base Rate Fallacy
3. **Phase 3**: Hypothesis Testing - Chi-Square Goodness of Fit for A/B Testing
4. **Phase 4**: Survivorship Bias - Crypto Market Simulation

## Prerequisites

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Phase 1: Robust Statistics (MAD vs Standard Deviation)

### Objective
Compare the robustness of Median Absolute Deviation (MAD) versus Standard Deviation (SD) when dealing with outlier-heavy data.

### Scenario
Analyzing server latency logs where 98% of traffic is normal (20-50ms) but 2% experiences extreme spikes (1000-5000ms), simulating DDoS attacks or system issues.

### Key Concepts
- **Standard Deviation**: Sensitive to outliers due to squaring deviations
- **Median Absolute Deviation**: Resistant to outliers, uses median of absolute deviations

### Results
The example demonstrates how SD (488.97) is dramatically inflated by outliers while MAD (8.0) provides a more accurate representation of typical data dispersion.

### Implementation
```python
def calculate_mad(data):
    median = np.median(data)
    absol_dev = np.abs(data - median)
    mad = np.median(absol_dev)
    return mad
```

## Phase 2: Bayesian Inference (Plagiarism Detection)

### Objective
Apply Bayes' Theorem to understand the base rate fallacy in automated plagiarism detection.

### Scenario
"IntegrityAI" claims 98% accuracy (sensitivity and specificity), but different base rates dramatically affect the probability that a flagged paper is actually plagiarized.

### Key Formula
```
P(Cheater | Flagged) = (Sensitivity × Prior) / (Sensitivity × Prior + (1 - Specificity) × (1 - Prior))
```

### Test Scenarios
- **Scenario A** (Bootcamp, 50% base rate): 98.00% - High accuracy
- **Scenario B** (Econ Class, 5% base rate): 72.06% - Moderate accuracy
- **Scenario C** (Honors Seminar, 0.1% base rate): 4.68% - Most flags are false positives!

### Key Insight
Even with 98% accuracy, in low-prevalence environments (Honors Seminar), a flagged paper has only ~5% chance of being actual plagiarism. This demonstrates the critical importance of base rates.

## Phase 3: Chi-Square Goodness of Fit Test

### Objective
Detect potential engineering bias in A/B test user assignment.

### Scenario
"FinFlash" claims a 50/50 split with 100,000 users, but Control has 50,250 users and Treatment has 49,750 users (500 missing).

### Hypothesis Test
- **H₀**: Users are randomly assigned (50/50 split)
- **H₁**: Systematic bias exists in assignment
- **Critical Value**: χ² > 3.84 (p < 0.05) indicates invalid experiment

### Calculation
```python
chi_square = Σ[(Observed - Expected)² / Expected]
           = (50,250 - 50,000)² / 50,000 + (49,750 - 50,000)² / 50,000
           = 2.5
```

### Result
χ² = 2.5 < 3.84, so the experiment is **statistically valid** (though the 500 missing users warrant investigation).

## Phase 4: Survivorship Bias in Crypto Markets

### Objective
Visualize survivorship bias using a Pareto distribution to simulate crypto token launches.

### Simulation Parameters
- **N**: 10,000 token launches
- **Distribution**: Pareto (α = 1.16) - creates extreme inequality
- **Scale**: $1,000 minimum market cap

### Key Statistics
- **All Tokens Mean**: $4,765.03
- **Top 1% Mean**: $149,081.06
- **Bias Multiplier**: 31.29x

### Critical Insight
If you only study successful tokens (survivors), you overestimate success probability by **31x**. This explains why "average crypto returns" in media are misleading.

### Visualization
The code generates dual histograms:
1. **The Graveyard**: All 10,000 tokens (99% near zero)
2. **The Survivors**: Top 1% only (what media reports)

## Running the Code

### In Google Colab
1. Open the notebook in Google Colab (link in header)
2. Run all cells sequentially (`Runtime` → `Run all`)
3. View outputs and visualizations inline

### Locally
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn

# Run in Jupyter
jupyter notebook Econ_3916_Assignment_2_Audit.ipynb
```

## Key Takeaways

1. **Robust Statistics**: Always check for outliers; MAD is superior to SD for skewed data
2. **Base Rate Fallacy**: Test accuracy depends heavily on prevalence (prior probability)
3. **A/B Testing**: Even small imbalances may indicate systematic issues
4. **Survivorship Bias**: Success stories hide the vast graveyard of failures

## Mathematical Concepts Demonstrated

- **Bayesian Inference**: P(A|B) calculation
- **Chi-Square Test**: Goodness of fit with 1 degree of freedom
- **Pareto Distribution**: Power law for wealth/success concentration
- **Robust Statistics**: Resistance to outliers

## Applications

These concepts apply to:
- **Finance**: Risk assessment, fraud detection, investment analysis
- **Tech**: A/B testing, system monitoring, feature flagging
- **Healthcare**: Diagnostic test interpretation, clinical trials
- **Social Science**: Survey analysis, experimental design

## Author

Created for ECON 3916: Statistics and Machine Learning

## License

Educational use only

## Acknowledgments

This assignment demonstrates real-world applications of statistical methods taught in ECON 3916, with scenarios inspired by actual industry challenges in tech, finance, and data science.
