# California Housing Dataset Analysis: Outlier Detection & Comparative Forensics

## Overview
This notebook performs advanced outlier detection and comparative statistical analysis on the California Housing dataset using Isolation Forest and distributional forensics techniques.

## Dataset
- **Source**: California Housing Dataset (sklearn)
- **Size**: 20,640 observations
- **Target Variable**: Median House Value
- **Key Features**: 
  - MedInc (Median Income)
  - HouseAge
  - AveRooms
  - AveBedrms
  - Population

## Analysis Pipeline

### 1. Data Loading & Exploration
- Loads California housing data from sklearn
- Visualizes the "ceiling effect" at $500k cap
- Generates distribution histogram with KDE overlay

### 2. Manual Outlier Detection (IQR Method)
**Tukey Fence Methodology**:
- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Computes Interquartile Range (IQR = Q3 - Q1)
- Defines bounds:
  - Lower: Q1 - 1.5 × IQR
  - Upper: Q3 + 1.5 × IQR
- Flags observations outside these bounds

**Results**: 681 manual outliers detected (primarily wealthy districts)

### 3. Algorithmic Outlier Detection (Isolation Forest)
**Parameters**:
```python
IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
```

**Features Used**:
- MedInc (Median Income)
- HouseAge
- AveRooms
- AveBedrms
- Population

**Results**: 1,032 algorithmic outliers detected (~5% of dataset)

### 4. Visualization: Anomaly Detection
- Scatter plot: Income vs. Average Rooms
- Color coding:
  - Blue: Normal observations
  - Red: Outliers (Isolation Forest)
- Y-axis limited to 0-20 for clarity

### 5. Comparative Forensics Report

#### Methodology
**Data Split**:
- Normal: 19,608 observations (95%)
- Outliers: 1,032 observations (5%)

**Metrics Calculated**:

1. **Central Tendency**
   - Mean
   - Median

2. **Volatility Measures**
   - Standard Deviation (sensitive to extremes)
   - MAD (Median Absolute Deviation - robust)

3. **Inequality Wedge**
   - Formula: Mean - Median
   - Interpretation: Positive wedge → Right-skewed distribution

#### Key Findings

**Normal Group**:
- MedInc Wedge: $2,540 (Right-skewed)
- HouseVal Wedge: $24,800 (Right-skewed)

**Outlier Group**:
- MedInc Wedge: $15,338 (Heavily right-skewed)
- HouseVal Wedge: $61,117 (Heavily right-skewed)

**Insight**: Outliers show significantly stronger right-skew, indicating high-value tail effects (luxury market segment).

### 6. Visualization: The Pareto World
**Two-panel histogram comparison**:

**Left Panel - Core Market** (Normal Observations):
- Distribution of Median Income
- Mean vs. Median lines
- Color: Steel blue

**Right Panel - The Tail** (Outlier Observations):
- Distribution of Median Income
- Mean vs. Median lines  
- Color: Crimson

## Key Insights

### Forensic Value
This comparative approach is excellent for:
- **Fraud Detection**: Outliers show different statistical signatures
- **Market Segmentation**: Luxury vs. mainstream housing
- **Risk Assessment**: Understanding tail behavior
- **Policy Analysis**: Identifying structural inequality

### Statistical Interpretation
- **Inequality Wedge**: When Mean > Median, indicates right-skewness from high-value observations pulling average up
- **Std Dev vs MAD**: Large differences signal extreme outliers affecting traditional volatility measures
- **Tail Distribution**: Outliers represent distinct market segment with different economic characteristics

## Requirements
```python
pandas
numpy
matplotlib
seaborn
sklearn
scipy
```

## Usage
```python
# Run all cells sequentially
# Or execute specific sections:

# 1. Load data
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Detect outliers
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05)
df['outlier_iso'] = iso_forest.fit_predict(df[features]) == -1

# 3. Generate report
# (See full code for complete implementation)
```

## Output

### Console Output
- Outlier detection statistics
- Comparative forensics tables:
  - Central tendency metrics
  - Volatility metrics (Std Dev vs MAD)
  - Inequality wedge analysis
  - Forensic insights summary

### Visualizations
1. Distribution of house values (histogram + KDE)
2. Boxplot of median income (Tukey fence)
3. Scatter plot: Income vs. Rooms (anomaly detection)
4. Dual histogram: Core Market vs. The Tail

## Interpretation Guide

**Positive Inequality Wedge** → Right-skewed distribution
- High-value observations pull mean above median
- Indicates presence of luxury/premium segment

**Large Std Dev / Small MAD** → Extreme outliers present
- Standard deviation inflated by extreme values
- MAD remains robust

**Application**: Use to identify whether outliers represent opportunity, risk, or data quality issues.

## Notes
- Contamination parameter (0.05) assumes ~5% of data is anomalous
- Isolation Forest uses ensemble of 100 trees
- Y-axis zoom (0-20) removes extreme errors for clarity
- Mean/median comparison reveals distributional shape

## Author
Generated for statistical analysis and anomaly detection in real estate data.

## License
Educational purposes - California Housing Dataset via sklearn.
