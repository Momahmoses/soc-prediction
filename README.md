# Battery State of Charge (SOC) Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML regression pipeline predicting battery State of Charge (SOC) as a continuous percentage, enabling accurate range estimation for electric vehicles and real-time Battery Management System (BMS) feedback.

---

## Problem Statement

Inaccurate SOC estimation leads to EV range anxiety, over-charging, and premature battery degradation. This pipeline trains and evaluates multiple regression models to predict SOC from electrochemical and operational features.

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Model Training | Random Forest, Gradient Boosting, Linear Regression, Ridge, Decision Tree |
| Cross-Validation | K-fold evaluation with MAE, RMSE, and R² metrics |
| Feature Importance | Identify which battery parameters most influence SOC |
| Model Serialisation | Best model saved via joblib for deployment |
| EDA Visualisations | SOC distribution, feature correlations, degradation plots |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | joblib |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/soc-prediction.git
cd soc-prediction
pip install pandas scikit-learn matplotlib seaborn joblib numpy
python data_generator.py
python eda.py
python train.py
python predict.py
```

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
