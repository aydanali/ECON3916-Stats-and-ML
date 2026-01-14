## Lab 2: The Illusion of Growth & The Composition Effect

### Objective
Built a Python pipeline to ingest live economic data from the Federal Reserve API (FRED) to analyze long-term wage stagnation and correct for statistical biases in labor market data. This project demonstrates how composition effects can create misleading signals in economic indicators, particularly during periods of structural labor market disruption.

### Methodology

**Tech Stack:** Python, fredapi, pandas, matplotlib

**Data Pipeline:**
1. **API Integration:** Connected to FRED API to fetch real-time economic time series:
   - `AHETPI`: Average Hourly Earnings of Production & Nonsupervisory Employees
   - `CPIAUCSL`: Consumer Price Index for All Urban Consumers
   - `ECIWAG`: Employment Cost Index for Total Compensation

2. **Real Wage Calculation:** Deflated nominal wages using CPI to compute real purchasing power over a 50+ year period (1964-present), revealing the gap between nominal growth and actual living standards.

3. **Anomaly Detection:** Identified a statistical spike in real wages during 2020—the "Pandemic Paradox"—where average wages appeared to surge despite economic contraction.

4. **Composition Bias Correction:** Fetched the Employment Cost Index (ECI), a fixed-weight measure that holds workforce composition constant, to isolate the composition effect from true wage growth. Rebased both series to 2015=100 for direct comparison.

### Key Findings

**The Money Illusion (1964-Present):**
- Visualized 50+ years of wage data showing that while nominal wages have risen dramatically, real wages adjusted for inflation have remained essentially flat—demonstrating the "money illusion" where workers perceive growth that doesn't exist in purchasing power terms.

**The Pandemic Paradox (2020):**
- Standard wage measures showed an artificial spike in 2020, creating the illusion of a wage boom during the pandemic.
- By comparing against the ECI, I proved this spike was a **statistical artifact** caused by the composition effect: low-wage workers (service, hospitality) exited the labor force disproportionately, mechanically raising the average without any individual worker receiving higher wages.
- The ECI showed stable, modest growth during the same period—revealing that the "boom" was purely compositional, not driven by increased labor demand or bargaining power.

**Economic Interpretation:**
This analysis demonstrates a critical lesson in empirical economics: raw aggregates can mislead policymakers. The 2020 wage data, if taken at face value, would suggest strong labor market conditions when the reality was mass unemployment among vulnerable workers. Fixed-weight indices like the ECI are essential for distinguishing real economic signals from compositional noise.

---

**Visualization:** Time-series comparison charts showing nominal vs. real wages (1964-present) and standard average wages vs. composition-adjusted ECI (2015-present), with annotations highlighting the 2020 divergence.
