# Student Price Index (SPI) Analysis: Measuring Real College Inflation

## 📊 Project Overview

This project constructs a **Student Price Index (SPI)** to measure inflation as experienced by college students, comparing it against the official U.S. Consumer Price Index (CPI) and Boston-area regional CPI. The analysis reveals significant disparities between official inflation measures and the actual cost pressures facing students.

## 🎯 Key Findings

- **Student inflation significantly exceeds national measures**: Students experienced **182.22%** cumulative inflation since January 2016, compared to **135.74%** for the general population
- **Regional disparities exist but are complex**: Boston CPI shows **33.16%** inflation (rebased to Jan 2016 = 100), revealing measurement methodology differences between regional and national indices
- **Housing drives student inflation**: With rent weighted at 60% of the student basket, housing costs are the primary driver of elevated student inflation

## 📁 Project Structure

### Phase 1: Manual Inflation Calculation
- Calculates inflation rates for individual student goods (2016-2024)
- Items analyzed: Rent, Groceries, Spotify subscription, Sandwich prices
- **Key insight**: Sandwich prices inflated **71.43%**, highlighting food service cost pressures

### Phase 2: FRED API Data Collection
- Fetches official CPI component series from Federal Reserve Economic Data (FRED)
- Series codes used:
  - `CPIAUCSL`: National CPI-U (All Urban Consumers)
  - `CUSR0000SEHA`: Rent of primary residence
  - `CUSR0000SAF11`: Groceries (Food at home)
  - `CUSR0000SERA02`: Recreation services (proxy for Spotify/streaming)
  - `CUSR0000SEFV`: Food away from home (proxy for sandwiches)
- Normalizes all series to common baseline (Jan 1992 = 100)

### Phase 3: Student Price Index Construction
- **Student basket weights**:
  - Rent: 60%
  - Groceries: 30%
  - Spotify/streaming: 2%
  - Sandwich/food service: 8%
- Creates weighted composite index
- Compares Student SPI vs. Official CPI with visualization

### Phase 4: Regional Comparison Analysis
- Incorporates Boston-Cambridge-Newton metro area CPI (`CUUSA103SA0`)
- Rebases Boston CPI to January 2016 = 100 for direct comparison
- Handles bimonthly data publication with forward-fill interpolation
- Three-way comparison: National CPI vs. Boston CPI vs. Student SPI

## 🔧 Technical Requirements
```python
# Required packages
pandas
matplotlib
seaborn
fredapi
```

Install dependencies:
```bash
pip install pandas matplotlib seaborn fredapi
```

## 🔑 Setup Instructions

1. **Obtain FRED API Key**:
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Create free account and generate API key
   - Replace `'API_KEY'` in code with your actual key

2. **Run the analysis**:
```python
   from fredapi import Fred
   fred = Fred(api_key='YOUR_API_KEY_HERE')
```

## 📈 Methodology

### Index Construction
The Student Price Index uses a **weighted average** formula:
```
SPI = (Rent × 0.60 + Groceries × 0.30 + Spotify × 0.02 + Sandwich × 0.08) / 4
```

### Normalization Process
All series are normalized to a common baseline:
```python
normalized_value = (current_value / base_value) × 100
```

This allows direct comparison across different CPI components with different base years.

### Data Crime Example
Phase 3 includes a deliberate "data crime" visualization showing **un-normalized indices**. This demonstrates why comparing raw indices with different base years produces misleading results—like comparing apples to oranges.

## 📊 Visualizations

1. **Individual Component Trends** (Phase 3): Four-line plot showing divergent inflation paths for Rent, Groceries, Spotify, and Sandwiches
2. **SPI vs. National CPI** (Phase 3): Highlights the growing gap between student and general inflation
3. **Three-Way Regional Comparison** (Phase 4): National, Boston regional, and Student inflation trajectories

## 🎓 Economic Insights

### Why Student Inflation Differs
1. **Housing weight**: Students allocate ~60% of budget to rent vs. ~30% in official CPI
2. **Geographic concentration**: Students cluster in high-cost urban areas (e.g., Boston)
3. **Lifecycle effects**: Students consume more services experiencing above-average inflation

### Policy Implications
- **Financial aid calculations** using national CPI may underestimate true student cost burdens
- **Student loan policy** should account for differential inflation in education-related costs
- **University budgeting** for student services requires region-specific inflation adjustments

## 🐛 Known Issues & Solutions

### SettingWithCopyWarning
The code generates pandas warnings when modifying `df_norm`. This is cosmetic and doesn't affect results. To suppress:
```python
df_norm = df_norm.copy()  # Create explicit copy before modifications
```

### Missing Boston CPI Data
Boston metro CPI is published **bimonthly**, creating gaps in monthly data. Solution implemented:
```python
df_merged['Boston_CPI'] = df_merged['Boston_CPI'].ffill()  # Forward-fill
```

## 📚 Data Sources

- **FRED (Federal Reserve Economic Data)**: All CPI series
- **Bureau of Labor Statistics**: Original data provider for CPI
- **Time period**: January 1992 - December 2025 (407 observations)

## 🔄 Replication

To replicate this analysis:
1. Ensure FRED API key is configured
2. Run all code blocks sequentially (Phases 1-4)
3. Final DataFrame `df_merged` contains all three comparable indices
4. Analysis summary prints automatically after Phase 4

## 📝 Citation

If using this methodology in research:
```
Student Price Index Analysis (2025)
Data: Federal Reserve Economic Data (FRED)
Methodology: Weighted composite index construction with regional comparison
```

## 🤝 Contributing

This project is designed for educational purposes in macroeconomics and econometrics courses. Suggested extensions:
- Add more student-relevant categories (textbooks, technology, transportation)
- Incorporate additional metro areas for broader regional analysis
- Calculate real wage adjustments for student employment
- Extend historical analysis back to 1980s

## 📧 Contact

For questions about methodology or implementation, consult macroeconomic price index literature or FRED documentation.

---

**Last Updated**: December 2025  
**Python Version**: 3.12+  
**Pandas Version**: 2.2.2+
