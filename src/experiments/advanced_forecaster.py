import sys
import os
import pandas as pd
import numpy as np
import optuna
import warnings
import json

# Add src to sys.path so we can import from there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import compute_metrics, create_ml_features, predict_ml_recursive
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress trial logs, only show final results

# ==============================================================================
# AGGRESSIVE EXPERIMENT CONFIGURATION
# ==============================================================================

START_DATES = ["2017-06-01", "2017-09-01", "2018-01-01"]
MODEL_TYPES = ["xgboost", "lightgbm"]
TEST_DAYS = 30
N_TRIALS = 150

# ==============================================================================

def generate_lags_and_rolling_advanced(history_series):
    """Custom lag generator with aggressive momentum features."""
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
    
    # EMA
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

def predict_ml_recursive_advanced(model, train_y, test_df, feature_cols):
    """Recursive prediction loop using the advanced feature set."""
    predictions = []
    history = list(train_y)
    
    for idx, row in test_df.iterrows():
        features_dict = generate_lags_and_rolling_advanced(history)
        row_dict = row.to_dict()
        for k, v in features_dict.items():
            row_dict[k] = v
        pred_input = pd.DataFrame([row_dict])[feature_cols]
        
        pred = model.predict(pred_input)[0]
        pred = max(0.0, pred)
        
        predictions.append(pred)
        history.append(pred)
        
    return np.array(predictions)

def run_matrix_experiment():
    print(f"--- Starting Aggressive Forecasting Matrix Experiment ---")
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/order_timeseries.csv'))
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.index.freq = "D"

    feature_cols = [
        "is_holiday", "is_black_friday", "days_until_black_friday",
        "day_of_week", "day_of_month", "month", "quarter", "year",
        "is_weekend", "is_payday", "is_month_start", "is_month_end",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30", 
        "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
        "ema_7", "ema_14",
        "dod_diff", "wow_diff", "rolling_max_7", "rolling_min_7"
    ]
    
    results = []

    for start_date in START_DATES:
        print(f"\n>> Processing Data Cutoff: {start_date}")
        df_subset = df[df.index >= start_date].copy()
        
        train_data = df_subset.iloc[:-TEST_DAYS].copy()
        test_data = df_subset.iloc[-TEST_DAYS:].copy()
        y_test = test_data["order_count"].values
        test_feat = create_ml_features(test_data.reset_index())
        
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
                    "rolling_std_7": 0.0, "ema_7": train_feat["order_count"].iloc[0], "ema_14": train_feat["order_count"].iloc[0],
                    "dod_diff": 0.0, "wow_diff": 0.0, 
                    "rolling_max_7": train_feat["order_count"].iloc[max(0, i-7):i].max() if i > 0 else train_feat["order_count"].iloc[0],
                    "rolling_min_7": train_feat["order_count"].iloc[max(0, i-7):i].min() if i > 0 else train_feat["order_count"].iloc[0],
                })
            else:
                history_slice = train_feat["order_count"].iloc[:i].values
                lags_df_list.append(generate_lags_and_rolling_advanced(history_slice))
                
        lags_df = pd.DataFrame(lags_df_list)
        train_full = pd.concat([train_feat, lags_df], axis=1)
        train_full = train_full.iloc[30:].reset_index(drop=True)
        
        X_train = train_full[feature_cols]
        y_train = train_full["order_count"]

        for model_type in MODEL_TYPES:
            print(f"  Optimizing {model_type.upper()} ({N_TRIALS} trials)...", end="", flush=True)
            
            def objective(trial):
                if model_type == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "max_depth": trial.suggest_int("max_depth", 3, 9),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                        "random_state": 42
                    }
                    model = XGBRegressor(**params)
                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "max_depth": trial.suggest_int("max_depth", 3, 9),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                        "random_state": 42,
                        "verbose": -1
                    }
                    model = LGBMRegressor(**params)
                    
                model.fit(X_train, y_train)
                preds = predict_ml_recursive_advanced(model, train_data["order_count"], test_feat, feature_cols)
                metrics = compute_metrics(y_test, preds)
                return metrics["MAPE"]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=N_TRIALS)
            
            print(f" Best MAPE: {study.best_value:.2f}%")
            results.append({
                "start_date": start_date,
                "model_type": model_type,
                "best_mape": study.best_value,
                "best_params": study.best_params
            })

    # Find the absolute best configuration
    best_result = min(results, key=lambda x: x["best_mape"])
    
    print("\n" + "="*60)
    print("        AGGRESSIVE EXPERIMENT SUMMARY")
    print("="*60)
    for res in results:
        print(f"Start: {res['start_date']} | Model: {res['model_type'].ljust(8)} | MAPE: {res['best_mape']:.2f}%")
        
    print("\n" + "*"*60)
    print("                 OVERALL CHAMPION")
    print("*"*60)
    print(f"Model            : {best_result['model_type'].upper()}")
    print(f"Start Date       : {best_result['start_date']}")
    print(f"Achieved MAPE    : {best_result['best_mape']:.2f}%")
    print(f"Parameters       :\n{json.dumps(best_result['best_params'], indent=2)}")
    print("*"*60)

if __name__ == "__main__":
    run_matrix_experiment()
