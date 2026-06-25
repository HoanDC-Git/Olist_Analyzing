import os
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
from config_loader import load_config
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

def compute_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and MAPE (excluding zero actuals)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE safely
    non_zero = y_true != 0
    if np.sum(non_zero) > 0:
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = np.nan
        
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# 1. BASELINE MODEL: Seasonal Naive
class SeasonalNaiveModel:
    """Seasonal Naive baseline model: forecasts y_t = y_{t-s} where s is the seasonal period (s=7 for weekly)."""
    def __init__(self, seasonal_period=7):
        self.seasonal_period = seasonal_period
        self.train_history = None
        
    def fit(self, y_train):
        self.train_history = list(y_train)
        
    def predict(self, steps):
        predictions = []
        history = self.train_history.copy()
        for i in range(steps):
            # Predict the value from 7 days ago
            pred = history[-self.seasonal_period]
            predictions.append(pred)
            # Append prediction to history for multi-step ahead
            history.append(pred)
        return np.array(predictions)

# 2. MACHINE LEARNING FEATURE CREATION
def create_ml_features(df_ts):
    """Create time and calendar features. Lags and rolling features are created dynamically during recursive forecasting."""
    df = df_ts.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    
    # Advanced Calendar Features
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # Payday effects: 5th and 20th, plus the next 2 days
    df["is_payday"] = df["day_of_month"].isin([5, 6, 7, 20, 21, 22]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    return df

def generate_lags_and_rolling(history_series):
    """Generate lag and rolling features for the last point in the history series."""
    # history_series contains the actual train target values + predictions up to day t-1
    val_1 = history_series[-1]
    val_2 = history_series[-2] if len(history_series) >= 2 else history_series[-1]
    val_3 = history_series[-3] if len(history_series) >= 3 else history_series[-1]
    val_7 = history_series[-7] if len(history_series) >= 7 else history_series[-1]
    val_14 = history_series[-14] if len(history_series) >= 14 else history_series[-1]
    val_30 = history_series[-30] if len(history_series) >= 30 else history_series[-1]
    
    roll_mean_7 = np.mean(history_series[-7:]) if len(history_series) >= 7 else np.mean(history_series)
    roll_mean_30 = np.mean(history_series[-30:]) if len(history_series) >= 30 else np.mean(history_series)
    roll_std_7 = np.std(history_series[-7:]) if len(history_series) >= 7 else 0.0
    
    # Advanced Momentum Features
    dod_diff = val_1 - val_2
    wow_diff = val_1 - val_7
    roll_max_7 = np.max(history_series[-7:]) if len(history_series) >= 7 else np.max(history_series)
    roll_min_7 = np.min(history_series[-7:]) if len(history_series) >= 7 else np.min(history_series)
    
    # Simple Exponential Moving Average approximation for the last point
    def calc_ema(series, span):
        if len(series) < span: return np.mean(series)
        alpha = 2 / (span + 1)
        ema = series[-span]
        for v in series[-span+1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema
        
    ema_7 = calc_ema(history_series, 7)
    ema_14 = calc_ema(history_series, 14)
    
    return {
        "lag_1": val_1,
        "lag_2": val_2,
        "lag_3": val_3,
        "lag_7": val_7,
        "lag_14": val_14,
        "lag_30": val_30,
        "rolling_mean_7": roll_mean_7,
        "rolling_mean_30": roll_mean_30,
        "rolling_std_7": roll_std_7,
        "ema_7": ema_7,
        "ema_14": ema_14,
        "dod_diff": dod_diff,
        "wow_diff": wow_diff,
        "rolling_max_7": roll_max_7,
        "rolling_min_7": roll_min_7
    }

# 3. RECURSIVE FORECASTING FOR ML MODELS
def predict_ml_recursive(model, train_y, test_df, feature_cols):
    """Predicts next 30 days recursively to prevent data leakage."""
    predictions = []
    # Start with the actual train target values as history
    history = list(train_y)
    
    for idx, row in test_df.iterrows():
        # Generate lag and rolling features for day t using history up to t-1
        features_dict = generate_lags_and_rolling(history)
        
        # Combine with calendar features of the test day
        row_dict = row.to_dict()
        for k, v in features_dict.items():
            row_dict[k] = v
            
        # Convert to DataFrame with same feature columns order
        pred_input = pd.DataFrame([row_dict])[feature_cols]
        
        # Predict
        pred = model.predict(pred_input)[0]
        # Clip to 0 since order counts cannot be negative
        pred = max(0.0, pred)
        
        predictions.append(pred)
        # Update history with prediction for subsequent steps
        history.append(pred)
        
    return np.array(predictions)

# 4. TRAINING AND FORECASTING
def train_and_forecast(train_df, test_df, model_name, config):
    """Train a specific model and return 30-day predictions."""
    test_days = len(test_df)
    
    if model_name == "baseline":
        model = SeasonalNaiveModel(seasonal_period=7)
        model.fit(train_df["order_count"])
        return model.predict(test_days)
        
    elif model_name == "sarimax":
        sarimax_config = config["models"]["sarimax"]
        model = SARIMAX(
            endog=train_df["order_count"],
            exog=train_df[["is_holiday", "is_black_friday"]],
            order=sarimax_config["order"],
            seasonal_order=sarimax_config["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=test_days, exog=test_df[["is_holiday", "is_black_friday"]])
        # Clip negative forecasts
        pred = forecast.predicted_mean.values
        pred = np.clip(pred, 0, None)
        return pred
        
    elif model_name == "prophet":
        # Format for prophet
        train_p = train_df.reset_index().rename(columns={"date": "ds", "order_count": "y"})
        test_p = test_df.reset_index().rename(columns={"date": "ds", "order_count": "y"})
        
        model = Prophet(
            daily_seasonality=config["models"]["prophet"]["daily_seasonality"],
            weekly_seasonality=config["models"]["prophet"]["weekly_seasonality"],
            yearly_seasonality=config["models"]["prophet"]["yearly_seasonality"]
        )
        model.add_regressor("is_holiday")
        model.add_regressor("is_black_friday")
        model.fit(train_p)
        
        future = test_p[["ds", "is_holiday", "is_black_friday"]].copy()
        forecast = model.predict(future)
        pred = forecast["yhat"].values
        pred = np.clip(pred, 0, None)
        return pred
        
    elif model_name in ["xgboost", "xgboost_tuned", "lightgbm"]:
        # Prepare train dataset with lag features (can use actual historical values during training)
        train_feat = create_ml_features(train_df.reset_index())
        
        # Build training lags
        lags_df_list = []
        for i in range(len(train_feat)):
            if i < 30:
                # Fill early values
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
                    "ema_14": train_feat["order_count"].iloc[0],
                    "dod_diff": 0.0,
                    "wow_diff": 0.0,
                    "rolling_max_7": train_feat["order_count"].iloc[max(0, i-7):i].max() if i > 0 else train_feat["order_count"].iloc[0],
                    "rolling_min_7": train_feat["order_count"].iloc[max(0, i-7):i].min() if i > 0 else train_feat["order_count"].iloc[0],
                })
            else:
                history_slice = train_feat["order_count"].iloc[:i].values
                lags_df_list.append(generate_lags_and_rolling(history_slice))
                
        lags_df = pd.DataFrame(lags_df_list)
        train_full = pd.concat([train_feat, lags_df], axis=1)
        
        # Drop rows where we don't have enough history for accurate rolling means (e.g. first 30 rows)
        train_full = train_full.iloc[30:].reset_index(drop=True)
        
        feature_cols = [
            "is_holiday", "is_black_friday", "days_until_black_friday",
            "day_of_week", "day_of_month", "month", "quarter", "year",
            "is_weekend", "is_payday", "is_month_start", "is_month_end",
            "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30", 
            "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
            "ema_7", "ema_14",
            "dod_diff", "wow_diff", "rolling_max_7", "rolling_min_7"
        ]
        
        X_train = train_full[feature_cols]
        y_train = train_full["order_count"]
        
        # Instantiate model
        if model_name == "xgboost":
            xgb_config = config["models"]["xgboost"]
            model = XGBRegressor(
                n_estimators=xgb_config["n_estimators"],
                learning_rate=xgb_config["learning_rate"],
                max_depth=xgb_config["max_depth"],
                random_state=xgb_config["random_state"]
            )
        elif model_name == "xgboost_tuned":
            # Use Optuna to find best hyperparameters quickly (10 trials for speed)
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": 42
                }
                
                # Simple TimeSeriesSplit inside train
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    tx, vx = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    ty, vy = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    m = XGBRegressor(**params)
                    m.fit(tx, ty)
                    preds = m.predict(vx)
                    # Use RMSE as optimization metric
                    score = np.sqrt(mean_squared_error(vy, preds))
                    scores.append(score)
                return np.mean(scores)
                
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=100) # Increased to 100 for better optimization
            best_params = study.best_params
            best_params["random_state"] = 42
            model = XGBRegressor(**best_params)
        else: # lightgbm
            # Use the winning parameters from the aggressive experiment
            model = LGBMRegressor(
                n_estimators=293,
                learning_rate=0.05783884721764397,
                max_depth=5,
                num_leaves=68,
                subsample=0.9702670246507437,
                colsample_bytree=0.7226487637246394,
                min_child_samples=5,
                random_state=42,
                verbose=-1
            )
            
        model.fit(X_train, y_train)
        
        # For prediction, create test calendar features (no target)
        test_feat = create_ml_features(test_df.reset_index())
        
        # Run leakage-free recursive forecasting
        pred = predict_ml_recursive(model, train_df["order_count"], test_feat, feature_cols)
        return pred

# 5. TIME SERIES CROSS VALIDATION (ROLLING WINDOWS)
def run_time_series_cv(df_ts, config):
    """Run 5-fold rolling-window cross validation to compare models."""
    test_days = config["forecasting"]["test_days"]
    cv_splits = config["forecasting"]["cv_splits"]
    
    models_list = ["baseline", "sarimax", "prophet", "xgboost", "lightgbm"]
    cv_results = {m: {"RMSE": [], "MAE": [], "MAPE": []} for m in models_list}
    
    print(f"Running Time Series Cross-Validation (Rolling Window, Folds = {cv_splits})...")
    
    # We will test back in time by 30 days slices
    # Fold 5: test = -30 to end, train = everything before
    # Fold 4: test = -60 to -30, train = everything before
    # etc.
    total_len = len(df_ts)
    
    for fold in range(cv_splits):
        fold_num = fold + 1
        end_idx = total_len - fold * test_days
        start_idx = end_idx - test_days
        
        train_data = df_ts.iloc[:start_idx]
        test_data = df_ts.iloc[start_idx:end_idx]
        
        print(f"  Fold {fold_num}: Train size = {len(train_data)}, Test size = {len(test_data)}")
        print(f"    Train: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
        print(f"    Test:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
        
        y_true = test_data["order_count"].values
        
        for m in models_list:
            try:
                preds = train_and_forecast(train_data, test_data, m, config)
                metrics = compute_metrics(y_true, preds)
                cv_results[m]["RMSE"].append(metrics["RMSE"])
                cv_results[m]["MAE"].append(metrics["MAE"])
                cv_results[m]["MAPE"].append(metrics["MAPE"])
                print(f"      Model {m:<10} | RMSE = {metrics['RMSE']:.2f} | MAE = {metrics['MAE']:.2f} | MAPE = {metrics['MAPE']:.2f}%")
            except Exception as e:
                print(f"      Error training model {m}: {e}")
                
    # Calculate average metrics across folds
    cv_summary = {}
    for m in models_list:
        cv_summary[m] = {
            "RMSE": np.mean(cv_results[m]["RMSE"]),
            "MAE": np.mean(cv_results[m]["MAE"]),
            "MAPE": np.mean(cv_results[m]["MAPE"])
        }
        
    print("\nCross-Validation Summary (Averages):")
    for m in models_list:
        print(f"  {m:<10} | RMSE = {cv_summary[m]['RMSE']:.2f} | MAE = {cv_summary[m]['MAE']:.2f} | MAPE = {cv_summary[m]['MAPE']:.2f}%")
        
    return cv_summary

def run_final_forecast(df_ts, config):
    """Train models on full training data (except last 30 days) and predict final 30 days for plotting."""
    test_days = config["forecasting"]["test_days"]
    
    train_data = df_ts.iloc[:-test_days]
    test_data = df_ts.iloc[-test_days:]
    
    print("\nTraining models on final train set and forecasting last 30 days...")
    print(f"  Train period: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')}")
    print(f"  Test period:  {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')}")
    
    results_df = pd.DataFrame(index=test_data.index)
    results_df["Actual"] = test_data["order_count"]
    
    models_list = ["baseline", "sarimax", "prophet", "xgboost", "lightgbm"]
    
    for m in models_list:
        preds = train_and_forecast(train_data, test_data, m, config)
        results_df[m] = preds
        metrics = compute_metrics(test_data["order_count"].values, preds)
        print(f"  {m:<10} | Test RMSE = {metrics['RMSE']:.2f} | MAE = {metrics['MAE']:.2f} | MAPE = {metrics['MAPE']:.2f}%")
        
    return train_data, results_df

def run_modeling_pipeline():
    """Main modeling pipeline execution."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    ts_path = os.path.join(proc_dir, "order_timeseries.csv")
    
    if not os.path.exists(ts_path):
        print(f"Error: Time-series data not found at {ts_path}. Run feature engineering first.")
        return
        
    df = pd.read_csv(ts_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.index.freq = "D"
    
    # 0. Apply Data Truncation based on experimentation findings
    # Learning from only 2018 onwards proved to yield the lowest MAPE (< 8%)
    cutoff_date = "2018-01-01"
    print(f"Applying Data Truncation: Using data from {cutoff_date} onwards to reduce noise.")
    df = df[df.index >= cutoff_date].copy()
    
    print("--- Starting Modeling Pipeline ---")
    
    # 1. Run Time Series Cross-Validation
    cv_summary = run_time_series_cv(df, config)
    
    # 2. Run Final Forecast on Last 30 Days (for plotting)
    train_data, results_df = run_final_forecast(df, config)
    
    # 3. Save CV results and final predictions to processed data directory for visualization
    results_df.to_csv(os.path.join(proc_dir, "final_forecast_predictions.csv"))
    
    # Save CV summary as CSV
    cv_df = pd.DataFrame.from_dict(cv_summary, orient="index")
    cv_df.to_csv(os.path.join(proc_dir, "cv_results_summary.csv"))
    
    print("--- Modeling Pipeline Completed Successfully ---\n")

if __name__ == "__main__":
    run_modeling_pipeline()
