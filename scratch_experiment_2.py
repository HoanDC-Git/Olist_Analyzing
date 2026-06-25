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

# Best start date from previous run
start = "2017-06-01"
test_days = 30

df_subset = df[df.index >= start].copy()
train_data = df_subset.iloc[:-test_days].copy()
test_data = df_subset.iloc[-test_days:].copy()

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

# EXPERIMENT 1: Log transform target
y_train_log = np.log1p(train_full["order_count"])

# Try out a robust tree ensemble
model = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, colsample_bytree=0.8, subsample=0.8, random_state=42)
model.fit(X_train, y_train_log)

# We need to rewrite the predict_ml_recursive to handle log predictions internally, 
# but for a quick test, let's just do a 1-step ahead proxy if we don't rewrite it.
# Actually, I can just rewrite predict_ml_recursive here.
def predict_ml_recursive_log(model, train_y, test_df, feature_cols):
    predictions = []
    history = list(train_y)
    
    for idx, row in test_df.iterrows():
        features_dict = generate_lags_and_rolling(history)
        row_dict = row.to_dict()
        for k, v in features_dict.items():
            row_dict[k] = v
        pred_input = pd.DataFrame([row_dict])[feature_cols]
        
        pred_log = model.predict(pred_input)[0]
        pred = max(0.0, np.expm1(pred_log))
        
        predictions.append(pred)
        history.append(pred)
        
    return np.array(predictions)

test_feat = create_ml_features(test_data.reset_index())
preds_log = predict_ml_recursive_log(model, train_data["order_count"], test_feat, feature_cols)
metrics_log = compute_metrics(test_data["order_count"].values, preds_log)

print(f"Log1p Transformed | MAPE: {metrics_log['MAPE']:.2f}% | MAE: {metrics_log['MAE']:.2f}")

# EXPERIMENT 2: Just raw model on the filtered dates with tuning
model2 = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, colsample_bytree=0.8, subsample=0.8, random_state=42)
model2.fit(X_train, train_full["order_count"])
preds2 = predict_ml_recursive(model2, train_data["order_count"], test_feat, feature_cols)
metrics2 = compute_metrics(test_data["order_count"].values, preds2)

print(f"Raw Target Filtered | MAPE: {metrics2['MAPE']:.2f}% | MAE: {metrics2['MAE']:.2f}")

