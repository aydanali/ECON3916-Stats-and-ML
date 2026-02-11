# Lab 7: Statistics and Machine Learning

A comprehensive Jupyter notebook demonstrating fundamental statistical concepts through practical simulations and visualizations.

## Overview

This lab explores key statistical principles using real-world scenarios including sports betting, sampling theory, and Monte Carlo simulation for risk analysis.

## Contents

### 1. House Edge Simulation
Demonstrates why even a 50% win rate sports bettor loses money due to -110 odds (52.38% breakeven requirement).

### 2. Sampling Error Analysis
Visualizes how sample means naturally fluctuate around the true population mean through random variation.

### 3. Central Limit Theorem Demonstration
Illustrates how sampling distributions converge to normality regardless of the underlying population distribution shape.

### 4. Population Size vs. Sample Size ("Soup Analogy")
Proves that margin of error depends only on sample size (n), not population size (N).

### 5. Variance Impact on Investment Decisions
Shows how two companies with identical mean LTV/CAC ratios can have drastically different risk profiles due to variance.

### 6. Monte Carlo Bankruptcy Simulation
Compares independent vs. correlated revenue/burn models to demonstrate how correlation affects risk metrics.

## Key Concepts Covered

- **Law of Large Numbers**: Convergence of empirical frequencies to theoretical probabilities
- **Sampling Error**: Natural variation in sample estimates
- **Central Limit Theorem**: Normality of sampling distributions
- **Confidence Intervals**: Quantifying statistical uncertainty
- **Correlation Effects**: Impact of variable relationships on risk
- **Monte Carlo Methods**: Simulation-based analysis

## Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Key Findings

| Concept | Result |
|---------|--------|
| Breakeven for -110 odds | 52.38% win rate required |
| CLT sample size threshold | n ≥ 30 typically sufficient |
| Variance in decision-making | Critical despite identical means |
| Correlation risk reduction | 42% lower cash flow variance |

## Notable Results

- **Sports Betting**: -110 odds create a 2.38% house edge requiring above 52.38% accuracy for profitability
- **Sample Size**: Margin of error = 1.96 × σ / √n (population size irrelevant)
- **Startup Risk**: Positive correlation (ρ=0.7) between revenue and expenses reduces bankruptcy probability by stabilizing cash flow variance
- **Variance Formula**: Var(Revenue - Burn) drops from 1B to 580M with ρ=0.7

## Theoretical Foundation

### Variance Reduction Formula
```
Var(CF) = Var(Revenue) + Var(Burn) - 2·Cov(Revenue, Burn)
```

With positive correlation, the covariance term reduces total variance, reflecting realistic management behavior (cost-cutting during revenue declines).

## Usage

Run cells sequentially in Google Colab or Jupyter Notebook. Each section includes:
1. Setup and parameter definition
2. Simulation logic
3. Visualization
4. Interpretation
