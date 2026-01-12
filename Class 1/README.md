# Big Mac Index Analysis

A Python-based analysis of currency valuation using The Economist's Big Mac Index data. This project demonstrates purchasing power parity (PPP) concepts through the iconic Big Mac burger pricing across different countries.

## Overview

The Big Mac Index is an informal way to measure the purchasing power parity (PPP) between currencies. This analysis calculates how much currencies are over or undervalued relative to the US dollar based on the price of a Big Mac in different countries.

## Features

- **Currency Valuation Calculation**: Computes implied PPP and valuation percentages for multiple currencies
- **Data Visualization**: Creates horizontal bar charts showing currency under/overvaluation
- **Historical Data Access**: Connects to The Economist's full Big Mac Index dataset via GitHub

## Dataset

The project uses two data sources:

1. **Manual Dataset** (January 2015): A curated snapshot with 19 countries
2. **Full Dataset**: Historical data from The Economist's GitHub repository spanning multiple years

### Countries Included in Manual Dataset
United States, Argentina, Australia, Brazil, Britain, China, Egypt, Euro area, Hong Kong, Indonesia, Japan, Mexico, Norway, Pakistan, Philippines, Russia, Saudi Arabia, South Africa, South Korea

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

### Running in Google Colab
The code is designed to run in Google Colab and can be executed directly without additional setup.

## Usage

1. **Load the dataset**: The code creates a manual DataFrame with Big Mac prices and exchange rates
2. **Calculate valuations**: Automatically computes dollar prices, implied PPP, and valuation percentages
3. **Visualize results**: Generates a bar chart showing which currencies are overvalued or undervalued

```python
# Run all cells in sequence
# The visualization will display currencies ranked by valuation percentage
```

## Key Calculations

- **Dollar Price**: `local_price / dollar_ex`
- **Implied PPP**: `local_price / US_price`
- **Valuation**: `(implied_ppp - dollar_ex) / dollar_ex * 100`

## Results Interpretation

- **Negative values**: Currency is undervalued (Big Mac is cheaper than in the US)
- **Positive values**: Currency is overvalued (Big Mac is more expensive than in the US)
- **Zero**: Currency is at fair value relative to the US dollar

## Visualization

The project creates a horizontal bar chart using seaborn's "vlag" color palette, making it easy to identify:
- Undervalued currencies (left side, cooler colors)
- Overvalued currencies (right side, warmer colors)
- Fair value reference line at 0%

## Data Source

Full historical data: [The Economist Big Mac Data Repository](https://github.com/TheEconomist/big-mac-data)

## Notes

- This is an educational tool and informal economic indicator
- Results are based on January 2015 data for the manual analysis
- The full dataset provides access to historical trends and more recent data

## License

This project uses publicly available data from The Economist. Please refer to their repository for data usage terms.

## Author

Created for educational purposes to demonstrate economic concepts using real-world data.
