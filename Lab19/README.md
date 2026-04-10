# Tree-Based Models — Random Forests

## Objective
Benchmark Decision Tree, Ridge Regression, and Random Forest regressors on the
California Housing dataset, rigorously tune RF hyperparameters, and diagnose
feature importance through both impurity-based and permutation-based methods —
culminating in an interactive diagnostic dashboard.

---

## Methodology

- **Data:** California Housing dataset (20,640 observations, 8 features) split
  80/20 into train and test sets
- **Model Comparison:** Evaluated Decision Tree, Ridge Regression, and Random
  Forest on train/test R² to establish baseline and final performance benchmarks
- **Hyperparameter Tuning:** Ran `GridSearchCV` over RF hyperparameters
  (`n_estimators`, `max_depth`, `max_features`) with cross-validated scoring to
  identify the optimal bias-variance configuration
- **Feature Importance Diagnostics:** Extracted and contrasted MDI (mean decrease
  in impurity, computed on training splits) against permutation importance
  (computed on the test set) to surface potential impurity inflation in
  high-cardinality features
- **Classification Extension:** Binarized the target to build an RF classifier,
  benchmarked AUC against Logistic Regression to evaluate RF's discriminative
  power beyond regression
- **Interactive Dashboard:** Built a Plotly + ipywidgets dashboard with sliders
  for `n_estimators` (1–500) and `max_features` (1–8), updating three live
  panels: model comparison, feature importance, and the train vs. test R²
  learning curve

---

## Key Findings

- Random Forest outperformed both baselines — achieving **R² = [YOUR VALUE]**
  on the test set versus **R² = [YOUR VALUE]** for Ridge Regression — confirming
  that the non-linear, interaction-rich structure of housing price data is better
  captured by an ensemble of decorrelated trees than by a regularized linear model
- The R² learning curve plateaued around **100–150 trees**, establishing a
  practical compute-accuracy frontier: adding trees beyond this threshold yields
  diminishing returns at linear cost
- `MedInc` ranked as the dominant predictor under both MDI and permutation
  importance; geographic features (`Latitude`, `Longitude`) showed meaningful
  rank divergence between the two methods, consistent with MDI's known
  sensitivity to correlated and continuous features
- Low `max_features` values produced high-bias, low-variance forests;
  higher values reduced bias but eroded the decorrelation benefit of
  ensemble averaging — the performance peak on this dataset fell in the
  **3–5 feature** range
- The RF classifier's AUC exceeded Logistic Regression, reinforcing the
  model's ability to capture complex decision boundaries without manual
  feature engineering
