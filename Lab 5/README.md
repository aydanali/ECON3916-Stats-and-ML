# Monte Carlo Risk Simulation for SaaS Revenue Forecasting

A Python-based Monte Carlo simulation project demonstrating various statistical concepts including the Law of Large Numbers, the Monty Hall Problem, and SaaS revenue risk modeling using both Normal and Fat-Tail (Student's t) distributions.

## Overview

This project contains several simulation exercises that demonstrate fundamental probability and statistics concepts, culminating in a business application for SaaS revenue forecasting and risk analysis.

## Contents

### 1. Law of Large Numbers Demonstration
Simulates 5,000 coin flips to demonstrate how sample averages converge to the theoretical probability (0.5) as the number of trials increases.

**Key Concepts:**
- Cumulative probability
- Convergence to theoretical mean
- Visual demonstration of the Law of Large Numbers

### 2. Monty Hall Problem Simulation
Simulates 10,000 games of the famous Monty Hall problem to empirically demonstrate the counter-intuitive probability advantage of switching doors.

**Results:**
- Win rate when staying: ~33%
- Win rate when switching: ~67%

### 3. SaaS Revenue Risk Model

#### Normal Distribution Model
Models SaaS revenue using normally distributed variables:
- **Base Revenue**: $10,000,000
- **Churn Rate**: Normal(μ=10%, σ=2%)
- **New Sales**: Normal(μ=$1,500,000, σ=$500,000)

**Key Metrics:**
- Probability of Revenue Decline: 17.61%
- 95% Value at Risk (VaR): $9,626,321.38
- Required Capital Buffer: $373,678.62

#### Fat-Tail Model (Student's t-distribution)
More realistic model accounting for extreme events:
- **Base Revenue**: $10,000,000
- **Churn Rate**: Normal(μ=10%, σ=2%)
- **New Sales**: Student's t(df=3), scaled to μ=$1,500,000, σ=$500,000

**Key Metrics:**
- Probability of Revenue Decline: 21.84%
- 95% Value at Risk (VaR): $9,264,220.74
- Required Capital Buffer: $735,779.26
- **Additional Reserve Required**: $362,100.64

## Installation

### Requirements
```bash
pip install numpy matplotlib
```

### Dependencies
- `numpy`: For statistical operations and random sampling
- `matplotlib`: For data visualization

## Usage

Simply run the Jupyter notebook cells in order. Each section is self-contained and will produce its own visualizations and outputs.
```python
# Example: Run SaaS Risk Model
saas_risk_model(10000)  # 10,000 simulations
```

## Key Findings

### Business Implications

The comparison between Normal and Fat-Tail models reveals:

1. **Higher Risk in Reality**: The Fat-Tail model shows a 4.23 percentage point increase in probability of revenue decline

2. **Increased Capital Requirements**: An additional $362K in capital reserves is recommended under the Fat-Tail model

3. **Conservative Planning**: The Fat-Tail model better captures realistic market volatility including:
   - Budget freezes
   - Executive turnover
   - Competitive disruptions
   - Other "black swan" events

### Risk Management Recommendation

**Capital Reserve Strategy:**
- Normal Model Recommendation: $374K (3.74% of base revenue)
- **Fat-Tail Model Recommendation: $736K (7.36% of base revenue)**
- Difference: $362K additional buffer needed

## Visualization Outputs

The notebook generates three types of visualizations:

1. **Law of Large Numbers**: Line plot showing convergence to theoretical probability
2. **Monty Hall Results**: Console output comparing win rates
3. **SaaS Revenue Forecasts**: 
   - Individual histograms for Normal and Fat-Tail models
   - Overlay comparison histogram
   - VaR thresholds marked on each plot

## Technical Details

### Monte Carlo Methodology
- **Simulation Count**: 10,000 iterations per model
- **Confidence Level**: 95% (5th percentile VaR)
- **Random Sampling**: Uses NumPy's random number generators

### Statistical Distributions
- **Normal Distribution**: `np.random.normal(mean, std_dev, size)`
- **Student's t Distribution**: `np.random.standard_t(df=3, size)` with scaling

### Value at Risk (VaR) Calculation
VaR is calculated as the 5th percentile of simulated revenues, meaning we're 95% confident revenue will exceed this threshold.

## File Structure
```
.
├── Lab5_StatsML.ipynb          # Main Jupyter notebook
└── README.md                    # This file
```

## Business Use Case

This simulation framework can be adapted for:
- Revenue forecasting with uncertainty
- Risk assessment for financial planning
- Capital reserve requirement calculations
- Stress testing business models
- Communicating risk to stakeholders

## Memo to Chief Risk Officer

A sample memo is included demonstrating how to communicate these findings to executive leadership, including:
- Clear explanation of probability vs. reality
- Quantified risk metrics
- Actionable recommendations
- Business context for technical findings

## Future Enhancements

Potential extensions to this project:
- [ ] Add more revenue factors (marketing spend, seasonality)
- [ ] Implement correlation between variables
- [ ] Add confidence intervals to visualizations
- [ ] Create interactive dashboard
- [ ] Add sensitivity analysis
- [ ] Implement scenario planning capabilities

## License

This project is for educational purposes.

## Author

Created as part of ECON3916 - Statistics and Machine Learning coursework.

## Acknowledgments

- Course: ECON3916 - Stats and ML
- Institution: Northeastern University
- Student ID: 55524
