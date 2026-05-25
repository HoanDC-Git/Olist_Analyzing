# %% [markdown]
# # Time Series Forecasting Prototyping - Olist Daily Orders
# This notebook/script prototypes multiple forecasting models (Baseline, SARIMAX, Prophet, XGBoost, LightGBM)
# using time series cross-validation.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from config_loader import load_config
from models import compute_metrics, run_time_series_cv, run_final_forecast

# Load configuration
config = load_config()
proc_dir = config["paths"]["processed_data_dir"]

# %% [markdown]
# ## 1. Load Daily Order Volume Time Series
# %%
ts_path = os.path.join(proc_dir, "order_timeseries.csv")
df = pd.read_csv(ts_path)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df.index.freq = "D"

print(f"Time series loaded: {df.shape[0]} days of data.")
print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

# %% [markdown]
# ## 2. Execute Time Series Cross Validation (5 Folds)
# Let's perform rolling window cross-validation to get robust out-of-sample metrics.
# %%
cv_summary = run_time_series_cv(df, config)

# %% [markdown]
# ## 3. Final Forecast Prototyping on the Last 30 Days
# Let's see how each model performs on the final 30-day period (August 2018).
# %%
train_data, results_df = run_final_forecast(df, config)

# Plot forecasts inline
plt.figure(figsize=(14, 7))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="black", linewidth=2.5, marker="o", markersize=4)
for col in ["baseline", "sarimax", "prophet", "xgboost", "lightgbm"]:
    if col in results_df.columns:
        plt.plot(results_df.index, results_df[col], label=col.upper(), linestyle="--", alpha=0.8)
plt.title("Model Forecasts vs Actual (Last 30 Days of clean data)")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Analyze Model Strengths & Weaknesses
#
# Let's review the CV results and performance characteristics:
#
# 1. **Prophet (Additive Regression)**:
#    - **Strengths**: Robust to missing data, captures multiple seasonalities (weekly, yearly) naturally, handles events 
#      (like Black Friday spikes) smoothly using its built-in holiday and event regressors.
#    - **Weaknesses**: Can be slow on very large datasets (not an issue here), may overfit if the seasonal Fourier terms are too high.
#
# 2. **XGBoost (Autoregressive ML)**:
#    - **Strengths**: Excellent at modeling complex non-linear relationships. Highly flexible; can easily integrate arbitrary calendar 
#      features, holiday indicators, and multiple lag/rolling target features.
#    - **Weaknesses**: Tree models cannot extrapolate trends (it will never predict a value higher than what is seen in the training data, 
#      unless trend is detrended first). Requires recursive predictions to avoid data leakage, which increases computation.
#
# 3. **LightGBM (Autoregressive ML)**:
#    - **Strengths**: Extremely fast and light. Similar modeling power to XGBoost.
#    - **Weaknesses**: Can overfit on small datasets. Default hyperparameter values are usually optimized for larger tabular datasets, 
#      leading to worse performance out-of-the-box compared to XGBoost on small time series.
#
# 4. **SARIMAX (Classical Statistical)**:
#    - **Strengths**: Solid theoretical foundation. Excellent for short-term stationary series with strong linear autocorrelation.
#    - **Weaknesses**: Linear formulation cannot capture complex non-linear patterns. Heavily sensitive to model orders $(p, d, q) \times (P, D, Q)_s$. 
#      Extreme spikes (like Black Friday) or structural level shifts break its assumptions, leading to poor predictions.
#
# 5. **Seasonal Naive (Baseline)**:
#    - **Strengths**: 0 training time. Captures weekly seasonality perfectly by copying the value of the same day last week.
#    - **Weaknesses**: Cannot adapt to trends, level shifts, or specific moving holidays.
