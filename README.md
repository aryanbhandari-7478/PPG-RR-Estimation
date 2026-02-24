# Respiration Rate (RR) Estimation from PPG Signals Using Machine Learning

## Project Overview

This project focuses on **predicting Respiration Rate (RR)** from pre-processed **Photoplethysmogram (PPG) signals**. The goal is to develop **robust regression models** that can accurately estimate RR in a **subject-independent setting**.

We compare **Random Forest (RF)** and **LightGBM (LGBM)** models under **baseline and tuned conditions** with **subject-wise cross-validation**, including **feature selection** for improved performance.

---

## Dataset Source

This project uses the **BIDMC PPG and Respiratory Annotated Dataset** from PhysioNet.  
The dataset can be accessed here:  
🔗 **https://physionet.org/content/bidmc/1.0.0/**

The dataset contains:
- Impedance respiratory signal
- Photoplethysmogram (PPG)
- ECG lead II
- Manual breath annotations by two independent annotators
- Physiological parameters (HR, RR, SpO2)
- Static variables (age, gender)

Each recording is ~8 minutes long and was used to extract features for the machine learning models.

---



## Repository Structure

```text
healthcare_ppg/
├── data/
│   ├── bidmc_csv/              # Raw BIDMC dataset files
│   └── processed/              # Cleaned and transformed data for training
│       ├── features_rr.csv     # Extracted features in CSV format
│       ├── features_rr.pkl     # Serialized features for fast loading
│       ├── X_windows.npy       # Numpy array of input windows/features
│       └── y_labels.npy        # Numpy array of target labels
├── models/                     # Saved model pipelines and weights
│   ├── best_lgbm_rr_pipeline.pkl
│   ├── best_RF1_rr_pipeline.pkl
│   ├── best_RF2_rr_pipeline.pkl
│   ├── LGBM_rr_pipeline.pkl
│   └── RF_rr_pipeline.pkl
└── notebooks/                  # Step-by-step development workflow
    ├── 01_data_overview.ipynb        # Initial EDA and data inspection
    ├── 02_data_preprocessing.ipynb   # Cleaning and signal filtering
    ├── 03_feature_engineering.ipynb  # Extraction of PPG/RR features
    ├── 04_RF_baseline_model.ipynb    # Initial Random Forest baseline
    ├── 05_LGBM_baseline_model.ipynb  # Initial LightGBM baseline
    ├── 06_RF1_model_training.ipynb   # Advanced training for RF Model 1
    ├── 07_RF2_model_training.ipynb   # Advanced training for RF Model 2
    ├── 08_LGBM_model_training.ipynb  # Hyperparameter tuning for LightGBM
    └── 09_predictions.ipynb          # Final inference and evaluation
```

## Features

### Feature Set

- Frequency-domain: dominant_freq, resp_power, rr_dominant_freq, total_power, rr_resp_power, rr_total_power  
- Time-domain: mean, std, rms, ptp  
- Statistical: skew, kurtosis  

Usage in models:  
- Baseline models → all 12 features  
- Tuned models → top 8 features  

---
## Models

### 1. Random Forest (RF)

- **Baseline RF:** Uses all 12 features, default hyperparameters  
- **Lightly Tuned RF:** Focuses on top features, moderate hyperparameter tuning  
- **Highly Tuned RF:** Uses top features, extensive hyperparameter tuning (n_estimators, max_depth, min_samples_leaf, max_features)  

### 2. LightGBM (LGBM)

- **Baseline LGBM:** Default parameters, 8 features  
- **Tuned LGBM:** Optimized via **GridSearchCV** over max_depth, num_leaves, learning_rate, subsample, colsample_bytree  

---

## Performance Comparison

| Model | Features | Mean MAE (bpm) | Std MAE | Notes |
|-------|----------|----------------|---------|-------|
| RF Baseline | 12 | 2.138 | 0.183 | Stable, relies on frequency-domain |
| RF Tuned (Light) | 8 | 2.120 | 0.170 | Improved feature selection |
| RF Tuned (High) | 8 | 2.105 | 0.162 | Best RF performance |
| LGBM Baseline | 8 | 2.186 | 0.291 | Higher variance, vertical column predictions |
| LGBM Tuned | 8 | 2.140 | 0.180 | Slightly better than baseline |

**Observations:**

- RF models are **more stable**, especially on small datasets.  
- LGBM shows **higher variance** on subject-wise CV due to boosting sensitivity.  
- Frequency-domain features dominate feature importance; statistical features have minor contributions.  


---
## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
2. Load a Model
import joblib

# Load tuned LGBM model
model_lgbm = joblib.load("models/best_lgbm_rr_pipeline.pkl")

# Load highly tuned RF model
model_rf = joblib.load("models/best_RF2_rr_pipeline.pkl")
3. Prepare Input Features

Note:

Tuned models: use top 8 features

Baseline models: use all 12 features

Sample Input for 8-feature Models (Tuned)
import numpy as np

X_sample_8 = np.array([
    [0.12, 0.35, 0.28, 0.44, 0.21, 0.18, 0.31, 0.25],
    [0.11, 0.36, 0.29, 0.43, 0.20, 0.19, 0.30, 0.24],
    [0.13, 0.34, 0.27, 0.45, 0.22, 0.17, 0.32, 0.26],
    [0.12, 0.35, 0.28, 0.44, 0.21, 0.18, 0.31, 0.25],
    [0.11, 0.36, 0.29, 0.43, 0.20, 0.19, 0.30, 0.24]
])
Sample Input for 12-feature Models (Baseline)
X_sample_12 = np.array([
    [0.12,0.35,0.28,0.44,0.21,0.18,0.31,0.25,0.02,0.05,0.03,0.01],
    [0.11,0.36,0.29,0.43,0.20,0.19,0.30,0.24,0.01,0.06,0.02,0.02],
    [0.13,0.34,0.27,0.45,0.22,0.17,0.32,0.26,0.03,0.04,0.01,0.03],
    [0.12,0.35,0.28,0.44,0.21,0.18,0.31,0.25,0.02,0.05,0.03,0.01],
    [0.11,0.36,0.29,0.43,0.20,0.19,0.30,0.24,0.01,0.06,0.02,0.02]
])
4. Make Predictions
# Predictions for tuned 8-feature model
y_pred_8 = model_lgbm.predict(X_sample_8)
print("Predictions (8-feature model):", y_pred_8)

# Predictions for baseline 12-feature model
y_pred_12 = model_rf.predict(X_sample_12)
print("Predictions (12-feature model):", y_pred_12)
