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
