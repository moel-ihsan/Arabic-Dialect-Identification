# Evaluating Oversampling Methods for Imbalanced Arabic Dialect Identification

[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-yellow.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and experimental pipeline for the paper: **"Evaluating Oversampling Methods for Imbalanced Arabic Dialect Identification"**. 

This project aims to classify Arabic dialects using a feature-engineered machine learning approach, focusing specifically on handling **severe class imbalance** through density-based oversampling techniques in a highly sparse 5,644-dimensional text feature space.

## 📈 Imbalance Handling Methods Evaluated
This project explores and rigorously evaluates several data-level and algorithm-level approaches:
* **Original** (Imbalanced data without resampling)
* **SMOTE** (Standard Synthetic Minority Over-sampling Technique)
* **ASTRA-SMOTE** (*Adaptive Safe-Threshold Resampling Algorithm* - implemented in files as `knsmote`)
* **SMOTE-RADIANT** (*Robust Auto-Density Inter-class Anomaly Neutralization Technique* - implemented in files as `dbscansmote`)
* **ClassWeight** (Cost-sensitive learning via inverse-frequency weights)

**Classifiers Used:** LightGBM (Leaf-wise) and XGBoost (Level-wise).

## 🏆 Key Findings & Results

Our empirical evaluation in a 5,644-dimensional sparse text space revealed that conventional generative oversampling (SMOTE) and cost-sensitive weighting (ClassWeight) often distort decision boundaries, sacrificing majority-class precision for minority-class recall. 

The proposed density-filtered approach, **SMOTE-RADIANT**, paired with a leaf-wise gradient boosting architecture (LightGBM), emerged as the champion configuration:

* **Superior Performance:** Achieved the highest Macro-F1 score (**0.8539**) and overall accuracy (**89.70%**).
* **Safe Minority Rescue:** Successfully improved the recognition of the severely underrepresented Jordanian dialect without cannibalizing the majority Syrian class.
* **Statistical Significance:** 1,000-iteration bootstrapping confirmed that the improvement over the original baseline is highly significant (*p < 0.001*) with a large effect size (Cohen's *r* = 0.511).
* **Noise Reduction:** RADIANT autonomously filtered out ~18% of overlapping synthetic anomalies, maintaining structural tension at the inter-class boundaries.

*(For detailed confusion matrices, per-class metrics, and feature importance analysis, please refer to the main paper).*

## ⚙️ Installation & Setup

It is highly recommended to use a virtual environment to run this pipeline:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

This study utilizes the publicly available **Shami Corpus** (Kwaik et al., 2018) for Levantine Arabic dialect classification (Jordanian, Lebanese, Palestinian, Syrian). 

Please download the dataset from the official repository and place it in the designated directory before running the notebooks:
`dataset/`

> **Note:** Due to licensing and size constraints, the dataset is not included in this repository.


## 📁 Repository Structure & ▶️ Execution Order

To reproduce the exact findings reported in the paper and ensure a strict, leakage-free protocol, please run the Jupyter Notebooks in the following sequential order:

### 🔹 Step 0 — Preprocessing & Utilities
* `0. feature_engineering.ipynb` *(Text normalization, TF-IDF, MI filtering, OOF Stacking)*
* `0. hybrid_oversampling.ipynb` *(Implementation of ASTRA-SMOTE and SMOTE-RADIANT algorithms)*
* `0. utils_ml.ipynb` *(Helper functions for evaluation)*

### 🔹 Step 1 — Original Models (No Resampling)
* `1. lightgbm_original.ipynb`
* `1. xgboost_original.ipynb`

### 🔹 Step 1.5 — Hyperparameter Optimization
* `1.5. joint_search.ipynb` *(Grid search for sampling targets and model hyperparameter tuning)*

### 🔹 Step 2 — SMOTE
* `2. lightgbm_smote.ipynb`
* `2. xgboost_smote.ipynb`

### 🔹 Step 3A — ASTRA-SMOTE
* `3A. lightgbm_knsmote.ipynb`
* `3A. xgboost_knsmote.ipynb`

### 🔹 Step 3B — SMOTE-RADIANT (Proposed Champion)
* `3B. lightgbm_dbscansmote.ipynb`
* `3B. xgboost_dbscansmote.ipynb`

### 🔹 Step 4 — Cost-Sensitive Learning
* `4. lightgbm_balanced_weighted.ipynb`
* `4. xgboost_balanced_weighted.ipynb`

### 🔹 Effect Size Analysis
* `z. effect_size.ipynb` *(1,000-iteration Bootstrapping, Wilcoxon signed-rank tests, and Cohen's effect sizes)*

## ⚠️ Important Notes

- Temporary files such as `cache`, `__pycache__`, and generated artifacts are intentionally excluded via `.gitignore`.

- The pipeline is designed with a strict train/validation/test split to prevent data leakage during out-of-fold stacking and resampling.

- Ensure that the notebooks are executed in the exact order listed above to maintain consistency in feature block generation.

## 📝 Citation

If you find this code, methodology, or experimental pipeline useful for your research, please consider citing:

```bibtex
@article{ahmad2026evaluating,
  title={Evaluating Oversampling Methods for Imbalanced Arabic Dialect Identification},
  author={Ahmad, Maulana Ihsan and Musdholifah, Aina and Nurwidyantoro, Arif},
  journal={Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)},
  year={2026},
  publisher={IJEECS}
}
```

> ⚠️ **Note:** Citation details (volume, issue, and DOI) will be updated upon publication.



## 👥 Authors & Affiliation

- **Maulana Ihsan Ahmad**, **Aina Musdholifah**, **Arif Nurwidyantoro**
- *Department of Computer Science and Electronics, Universitas Gadjah Mada, Indonesia*