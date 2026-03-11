# Data Wrangling & Engineering Pipeline

## Objective
Engineer a robust set of structural features and systematically impute missing values within a real-world HR economics dataset, producing a clean, model-ready matrix suitable for downstream econometric inference.

---

## Methodology

- **Missingness Diagnostics (MAR Analysis):** Leveraged `missingno` to visualize and characterize the dataset's missing data architecture. Confirmed a Missing At Random (MAR) mechanism by identifying systematic correlations between missingness indicators and observed covariates — ruling out MCAR and flagging potential bias risks under listwise deletion.

- **Feature Engineering & Type Casting:** Parsed and restructured raw columns into semantically appropriate dtypes (ordinal, nominal, continuous), establishing a coherent schema that satisfies the assumptions required by `statsmodels` OLS and GLM estimators.

- **Dummy Variable Trap Mitigation:** Applied one-hot encoding across categorical regressors while deliberately dropping a reference class per variable group. This full-rank encoding strategy eliminates perfect multicollinearity, ensuring the design matrix **X'X** remains invertible and coefficient estimates are identified.

- **Target Encoding for High-Cardinality Geographic Features:** Compressed high-dimensional geographic identifiers (e.g., city or region codes) using `category_encoders.TargetEncoder`, which replaces each category level with its conditional mean of the target variable. This preserves predictive signal while avoiding the dimensionality explosion inherent to naive one-hot encoding of geographic data.

- **Imputation Strategy:** Applied domain-informed imputation techniques consistent with the confirmed MAR structure, using conditional mean/median fills stratified by relevant subgroup covariates to minimize imputation bias.

---

## Key Findings

The pipeline successfully resolved all pre-modeling data quality issues present in `messy_hr_economics.csv`:

- **Missingness structure confirmed as MAR**, validating that imputation — rather than case deletion — is the statistically appropriate treatment. A naive complete-case analysis would have introduced selection bias and reduced effective sample size significantly.

- **Perfect multicollinearity bypassed** via reference-class dropping. The resulting design matrix is full-rank, satisfying the identification conditions necessary for unbiased OLS estimation and stable standard errors.

- **High-cardinality geographic noise compressed** through Target Encoding, reducing feature dimensionality while retaining the economic signal embedded in regional variation — a common challenge in labor and HR economics datasets with granular location identifiers.

The final engineered dataset is clean, structurally sound, and ready for causal or predictive econometric modeling.

---

**Stack:** Python · pandas · statsmodels · missingno · category_encoders
**Data:** `messy_hr_economics.csv`
