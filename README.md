# Olist E-Commerce Data Analysis & Forecasting Pipeline

This project has been restructured to implement a professional data science workflow for customer behavior analysis, logistics metrics, and daily order forecasting on the Olist e-commerce dataset (Brazil).

---

## 1. Project Folder Structure

The codebase is organized into clean, maintainable layers:

```
DA_remake/
│
├── config/                  # Configuration management
│   └── config.yaml          # Data paths and model hyperparameter settings
│
├── data/                    # Data storage (excluded from Git tracking)
│   ├── raw/                 # Original raw datasets (olist_*.csv) and holiday calendars
│   └── processed/           # Cleaned tables and analysis targets (RFM, Cohort, timeseries)
│
├── notebooks/               # Interactive exploration and prototyping (Percent Format # %%)
│   ├── 1.0_eda_business_analysis.py   # Interactive script for logistics, reviews, RFM, Cohort
│   └── 2.0_timeseries_prototyping.py  # Interactive script for time series modeling
│
├── src/                     # Core reusable python modules
│   ├── config_loader.py     # Configuration loader with resolved path helpers
│   ├── data_cleaning.py     # Smart missing-value imputation and geo-aggregation
│   ├── feature_engineering.py# Daily sales aggregation, holidays, RFM, and cohort calculations
│   ├── models.py            # Model definitions, recursive prediction, and 5-fold CV
│   └── visualization.py     # Figure generation (cohort heatmap, RFM bar, forecasts)
│
├── reports/                 # Analytical reports
│   ├── figures/             # High-quality generated PNG charts
│   └── final_report.md      # Detailed business and technical report (in English)
│
├── main.py                  # Orchestrator script to run the entire pipeline
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (This file)
```

---

## 2. Running the Pipeline

### Step 1: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 2: Run the End-to-End Pipeline
Run the main orchestrator script to automatically execute the entire pipeline (data cleaning, feature engineering, modeling, and plotting):
```bash
python3 main.py
```

### Step 3: View Results
- Check the generated analytical plots in: [reports/figures/](file:///home/naoh/Documents/projects/DA_remake/reports/figures)
- Read the detailed business report and model evaluation at: [reports/final_report.md](file:///home/naoh/Documents/projects/DA_remake/reports/final_report.md)

---

## 3. Core Enhancements

1. **Data Leakage Fix (Recursive Forecasting):**
   ML models (XGBoost, LightGBM) now use **Recursive Forecasting** for lag features. During testing, the models make step-by-step predictions and feed them back as lags for the next step, reflecting real-world conditions.
2. **Smart Imputation:**
   Avoided aggressive row deletions. Missing values in catalog attributes are filled using category-wise or median values to protect transactional records. Geolocation datasets are aggregated by zip code to compress storage and speed up geo-lookups by 50x.
3. **Robust Model Validation:**
   Instead of a single split, the project evaluates models (Baseline, SARIMAX, Prophet, XGBoost, LightGBM) using a **5-fold Time Series Cross-Validation**. XGBoost recursive modeling yields the best average CV MAPE of **20.45%**.
