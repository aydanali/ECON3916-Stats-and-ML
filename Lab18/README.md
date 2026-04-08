# Fraud Detection Model Evaluation — Metrics that Matter

## Objective

Evaluate a logistic regression fraud classifier on a severely imbalanced real-world dataset by moving beyond accuracy to a rigorous suite of threshold-aware metrics, culminating in a business-constrained operating point selection.

---

## Data

**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 real European credit card transactions
- Features: PCA-anonymized components `V1`–`V28`, transaction `Amount`, and binary fraud label `Class`
- Class imbalance: **0.172% positive (fraud) class** — a canonical high-stakes imbalance problem

---

## Methodology

- **Established the accuracy paradox baseline** — demonstrated that a naive all-negative classifier achieves 99.83% accuracy with zero fraud recall, exposing accuracy as a misleading metric in imbalanced settings
- **Trained a logistic regression classifier** using scikit-learn on the imbalanced dataset, with probability outputs retained for threshold analysis
- **Constructed a full confusion matrix** decomposition, reporting true positives, false positives, false negatives, and true negatives to characterize error asymmetry
- **Computed precision, recall, and F1-score** across the fraud class, isolating the tradeoff between false alarms and missed detections
- **Generated a ROC curve and calculated ROC-AUC**, measuring the model's discriminatory power across all classification thresholds
- **Generated a Precision-Recall curve and calculated PR-AUC**, prioritizing the fraud-class signal over background negative performance
- **Identified the F1-optimal threshold** — showed that the decision boundary maximizing fraud detection performance differs meaningfully from the default 0.5 cutoff
- **Applied a business capacity constraint** — simulated a real-world investigation desk limited to 500 daily reviews, and selected an operationally viable threshold that maximizes fraud caught within that bound

---

## Tech Stack

| Tool | Usage |
|---|---|
| `scikit-learn` | `LogisticRegression`, `confusion_matrix`, `classification_report`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `f1_score` |
| `matplotlib` / `seaborn` | ROC curves, PR curves, confusion matrix heatmaps, threshold sweep plots |
| `pandas` / `numpy` | Data manipulation, threshold grid construction |

---

## Key Findings

**The accuracy paradox is real and consequential.** A classifier that predicts no fraud achieves 99.83% accuracy — outperforming many naive models on paper while catching zero fraudulent transactions. In high-stakes imbalanced settings, accuracy is not a performance metric; it is a liability.

**Logistic regression recovers meaningful signal.** Despite the severe class imbalance, the trained classifier achieved strong ROC-AUC, indicating substantial rank-order discriminatory power across the full threshold range. PR-AUC — a more informative metric under imbalance — confirmed that the model captures a meaningful share of fraud cases at operationally acceptable precision levels.

**The default 0.5 threshold is arbitrary, not optimal.** The F1-maximizing threshold differed from the 0.5 default, underscoring that threshold selection is a modeling decision, not a default. Choosing the wrong operating point leaves detectable fraud on the table.

**Business constraints drive the final operating point.** Under a capacity constraint of 500 maximum daily investigations, threshold selection shifts from a pure statistical optimization to a resource allocation problem. The selected threshold reflects the practical tradeoff between fraud capture rate and investigator bandwidth — the kind of decision that separates a deployed model from a notebook exercise.

---

## Broader Takeaway

Fraud detection is fundamentally a decision problem under asymmetric costs — missing fraud is not equivalent to a false alarm. This lab operationalizes that insight by treating threshold selection as an economic choice: the model generates probabilities, but the operating point is set by the loss function of the business.
