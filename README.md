# PPG-RR-Estimation
Overview

This repository contains machine learning models for non-invasive estimation of Respiration Rate (RR) from photoplethysmogram (PPG) signals. RR is a key vital sign, and accurate estimation from wearable PPG sensors can improve patient monitoring in healthcare settings.

We provide both baseline and tuned models for Random Forest (RF) and LightGBM (LGBM), including multiple tuning levels for RF.

Repository Structure
.
├── data/
│   └── processed/
│       └── features_rr.csv          # Extracted features from PPG segments
├── models/
│   ├── best_lgbm_rr_pipeline.pkl    # Tuned LGBM (top 8 features)
│   ├── best_RF1_rr_pipeline.pkl     # Tuned RF (light tuning, top 8 features)
│   ├── best_RF2_rr_pipeline.pkl     # Tuned RF (highly tuned, top 8 features)
│   ├── LGBM_rr_pipeline.pkl         # Baseline LGBM (all 12 features)
│   └── RF_rr_pipeline.pkl           # Baseline RF (all 12 features)
├── notebooks/
│   ├── training.ipynb               # Model training and hyperparameter tuning
│   └── prediction.ipynb             # Step-by-step inference with sample inputs
├── README.md
└── requirements.txt                 # Python dependencies
Features
Feature Set

Frequency-domain: dominant_freq, resp_power, rr_dominant_freq, total_power, rr_resp_power, rr_total_power

Time-domain: mean, std, rms, ptp

Statistical: skew, kurtosis

Baseline models: Use all 12 features.

Tuned models: Use top 8 features selected based on feature importance.

Models
Model	Type	Features	Description
RF_rr_pipeline	Baseline	12	Random Forest baseline with default hyperparameters
LGBM_rr_pipeline	Baseline	12	LightGBM baseline
best_RF1_rr_pipeline	Tuned RF (light)	8	RF with light hyperparameter tuning
best_RF2_rr_pipeline	Tuned RF (high)	8	RF with extensive hyperparameter tuning
best_lgbm_rr_pipeline	Tuned LGBM	8	LGBM with GridSearchCV-tuned hyperparameters
How to Use
1. Install Dependencies
pip install -r requirements.txt
2. Load a Model
import joblib

# Load tuned LGBM
model = joblib.load("models/best_lgbm_rr_pipeline.pkl")
3. Prepare Input Features

Top 8 features for tuned models

All 12 features for baseline models

Example: 5 Sample Inputs for 8-feature Models
import numpy as np

X_sample_8 = np.array([
    [0.12, 0.35, 0.28, 0.44, 0.21, 0.18, 0.31, 0.25],
    [0.11, 0.36, 0.29, 0.43, 0.20, 0.19, 0.30, 0.24],
    [0.13, 0.34, 0.27, 0.45, 0.22, 0.17, 0.32, 0.26],
    [0.12, 0.35, 0.28, 0.44, 0.21, 0.18, 0.31, 0.25],
    [0.11, 0.36, 0.29, 0.43, 0.20, 0.19, 0.30, 0.24]
])
Example: 5 Sample Inputs for 12-feature Models
X_sample_12 = np.array([
    [0.12,0.35,0.28,0.44,0.21,0.18,0.31,0.25,0.02,0.05,0.03,0.01],
    [0.11,0.36,0.29,0.43,0.20,0.19,0.30,0.24,0.01,0.06,0.02,0.02],
    [0.13,0.34,0.27,0.45,0.22,0.17,0.32,0.26,0.03,0.04,0.01,0.03],
    [0.12,0.35,0.28,0.44,0.21,0.18,0.31,0.25,0.02,0.05,0.03,0.01],
    [0.11,0.36,0.29,0.43,0.20,0.19,0.30,0.24,0.01,0.06,0.02,0.02]
])
4. Make Predictions
y_pred_8 = model.predict(X_sample_8)
y_pred_12 = model.predict(X_sample_12)

print("Predictions for 8-feature model:", y_pred_8)
print("Predictions for 12-feature model:", y_pred_12)
Evaluation Metrics

Mean Absolute Error (MAE): primary metric

Standard deviation across 5-fold subject-wise CV: stability measure

Key observations:

Tuned RF and LGBM models achieve MAE ≈ 2.18 bpm

Baseline RF slightly more stable due to small dataset

Frequency-domain features dominate importance in all models

Visualization

Feature Importance Bar Charts

True vs Predicted Scatter Plots

Error Distribution Histograms

Combined Prediction Plots for Sample Inputs

These visualizations are available in notebooks/prediction.ipynb.

Notes

All models were trained using subject-wise cross-validation to ensure generalization.

Tuned models use top 8 features, while baseline models use all 12 features.

Dataset: 1638 PPG segments from 53 subjects.

RF models: bagging-based, stable on small datasets

LGBM models: boosting-based, sensitive to small dataset and binning effects

References

Random Forest: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

LightGBM: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.

License

This project is licensed under the MIT License.
