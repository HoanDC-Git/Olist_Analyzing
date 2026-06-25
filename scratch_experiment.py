import sys
sys.path.append('src')
import pandas as pd
import numpy as np
from models import compute_metrics, create_ml_features, generate_lags_and_rolling, predict_ml_recursive
from xgboost import XGBRegressor

df = pd.read_csv("data/processed/order_timeseries.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df.index.freq = "D"

# Test different start dates
start_dates = ["2017-01-01", "2017-06-01", "2017-09-01", "2018-01-01", "2018-03-01"]
test_days = 30

print("Testing different start dates with basic XGBoost...")
for start in start_dates:
    df_subset = df[df.index >= start].copy()
    if len(df_subset) < 60:
        continue
    
    train_data = df_subset.iloc[:-test_days]
    test_data = df_subset.iloc[-test_days:]
    
    # ML Prep
    train_feat = create_ml_features(train_data.reset_index())
    
    lags_df_list = []
    for i in range(len(train_feat)):
        if i < 30:
            lags_df_list.append({
                "lag_1": train_feat["order_count"].iloc[i-1] if i > 0 else train_feat["order_count"].iloc[0],
                "lag_2": train_feat["order_count"].iloc[i-2] if i >= 2 else train_feat["order_count"].iloc[0],
                "lag_3": train_feat["order_count"].iloc[i-3] if i >= 3 else train_feat["order_count"].iloc[0],
                "lag_7": train_feat["order_count"].iloc[i-7] if i >= 7 else train_feat["order_count"].iloc[0],
                "lag_14": train_feat["order_count"].iloc[i-14] if i >= 14 else train_feat["order_count"].iloc[0],
                "lag_30": train_feat["order_count"].iloc[i-30] if i >= 30 else train_feat["order_count"].iloc[0],
                "rolling_mean_7": train_feat["order_count"].iloc[max(0, i-7):i].mean() if i > 0 else train_feat["order_count"].iloc[0],
                "rolling_mean_30": train_feat["order_count"].iloc[max(0, i-30):i].mean() if i > 0 else train_feat["order_count"].iloc[0],
                "rolling_std_7": 0.0,
                "ema_7": train_feat["order_count"].iloc[0],
                "ema_14": train_feat["order_count"].iloc[0]
            })
        else:
            history_slice = train_feat["order_count"].iloc[:i].values
            lags_df_list.append(generate_lags_and_rolling(history_slice))
            
    lags_df = pd.DataFrame(lags_df_list)
    train_full = pd.concat([train_feat, lags_df], axis=1)
    train_full = train_full.iloc[30:].reset_index(drop=True)
    
    feature_cols = [
        "is_holiday", "is_black_friday", "days_until_black_friday",
        "day_of_week", "day_of_month", "month", "quarter", "year",
        "is_weekend", "is_payday", "is_month_start", "is_month_end",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30", 
        "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
        "ema_7", "ema_14"
    ]
    
    X_train = train_full[feature_cols]
    y_train = train_full["order_count"]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    test_feat = create_ml_features(test_data.reset_index())
    preds = predict_ml_recursive(model, train_data["order_count"], test_feat, feature_cols)
    
    metrics = compute_metrics(test_data["order_count"].values, preds)
    print(f"Start: {start} | Train Size: {len(train_data)} | MAPE: {metrics['MAPE']:.2f}% | MAE: {metrics['MAE']:.2f}")

