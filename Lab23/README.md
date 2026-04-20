# FedSpeak Analysis: NLP on FOMC Minutes

## Objective

Apply natural language processing techniques to over two decades of Federal Open Market Committee meeting minutes to extract latent sentiment signals, quantify policy uncertainty, and identify structural shifts in Fed communication across distinct macroeconomic regimes.

---

## Methodology

**Text Preprocessing**
- Loaded a corpus of 20+ years of FOMC meeting minutes and applied standard NLP preprocessing: tokenization, lemmatization, and stop word removal to produce a clean document representation

**Feature Engineering**
- Constructed a TF-IDF document-term matrix incorporating both unigrams and bigrams to capture not just individual policy terms but meaningful two-word phrases (e.g., "interest rate", "labor market", "price stability")

**Sentiment Scoring**
- Applied the Loughran-McDonald financial sentiment lexicon to compute two core measures per document: net sentiment (positive minus negative word counts, normalized by document length) and an uncertainty score capturing hedged or equivocal language

**Dimensionality Reduction and Clustering**
- Reduced the high-dimensional TF-IDF matrix via PCA, then applied K-Means clustering to group FOMC documents into latent policy regimes based on linguistic similarity
- Identified two dominant clusters corresponding to (1) crisis and monetary easing episodes (e.g., the Global Financial Crisis, COVID-19 shock) and (2) high-uncertainty and transitional periods marked by elevated policy ambiguity in Fed communication

**Temporal and Distributional Analysis**
- Visualized net sentiment and uncertainty scores as time series across the full sample to identify structural breaks and cyclical patterns
- Compared the pre-COVID and post-COVID sentiment distributions to evaluate whether the pandemic represented a permanent shift in the tone and character of Fed communication

---

## Key Findings

- **Regime clustering aligns with macro history.** K-Means on PCA-reduced TF-IDF vectors recovered two interpretable clusters without any labeled training data. Crisis and easing periods (GFC, COVID) cluster together based on shared language around accommodation and risk, while a separate cluster captures transitional phases defined by uncertainty and policy equivocation.

- **Post-COVID sentiment is durably more negative.** Net sentiment scores under the Loughran-McDonald lexicon declined meaningfully after March 2020 and did not revert to pre-pandemic levels over the post-COVID sample window. This suggests the pandemic represented not just a temporary shock but a structural shift in how the FOMC characterizes economic conditions in its public communications.

- **Bigrams add signal.** Phrase-level features in the TF-IDF matrix capture policy-relevant language that unigrams alone miss, strengthening both the clustering structure and the interpretability of dominant terms within each regime.

---

## Tools and Libraries

`Python` `scikit-learn` `NLTK` `pandas` `matplotlib` `Loughran-McDonald Lexicon`

---

## Context

This project is part of an applied econometrics and machine learning portfolio developed through ECON 3916 (Statistics and Machine Learning for Economics) at Northeastern University. It demonstrates the application of unsupervised learning and domain-specific NLP to central bank communication, bridging computational text analysis with macroeconomic interpretation.
