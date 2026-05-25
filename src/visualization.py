import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config_loader import load_config

# Set professional plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (12, 6)
})

# Custom premium palette
PALETTE = {
    'primary': '#1d3557',     # Deep Blue
    'secondary': '#457b9d',   # Muted Blue
    'accent': '#e63946',      # Vibrant Coral/Red
    'light': '#f1faee',       # Warm Soft Off-White
    'dark': '#2b2d42',        # Dark Slate
    'highlight': '#f4a261',   # Warm Orange/Gold
    'neutral': '#8d99ae'      # Muted Gray
}

def plot_cohort_heatmap():
    """Generate and save Cohort Retention Heatmap."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    charts_dir = config["paths"]["charts_dir"]
    
    cohort_path = os.path.join(proc_dir, "cohort_retention.csv")
    if not os.path.exists(cohort_path):
        print(f"Cohort file not found at {cohort_path}. Skipping.")
        return
        
    retention = pd.read_csv(cohort_path, index_col=0 if 'cohort_month' in pd.read_csv(cohort_path).columns else None)
    # Ensure cohort_month is set as index
    if 'cohort_month' in retention.columns:
        retention.set_index('cohort_month', inplace=True)
        
    # Drop Cohort Month index 0 (which is 100% retention) for better heatmap contrast if desired,
    # or just show it but set vmax=0.05 to see subtle variations in subsequent months (1-3%)
    plt.figure(figsize=(14, 9))
    
    # We set vmax=0.05 (5%) because retention in Olist after month 0 is extremely low (usually < 2%)
    # This highlights the actual differences between cohorts.
    sns.heatmap(
        retention.iloc[:, 1:], # Skip month 0 (100%)
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        cbar_kws={'label': 'Retention Rate (%)'},
        vmax=0.03, # Set max scale to 3% to make subtle retention visible
        linewidths=0.5
    )
    
    plt.title("Olist Customer Cohort Retention Rate (Month 1 onwards)", pad=20, fontweight="bold", color=PALETTE["dark"])
    plt.xlabel("Months After First Purchase", labelpad=10)
    plt.ylabel("Cohort (First Purchase Month)", labelpad=10)
    plt.tight_layout()
    
    save_path = os.path.join(charts_dir, "01_cohort_retention_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved cohort heatmap to {save_path}")

def plot_rfm_segments():
    """Generate and save RFM Segment distribution chart."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    charts_dir = config["paths"]["charts_dir"]
    
    rfm_path = os.path.join(proc_dir, "customer_rfm.csv")
    if not os.path.exists(rfm_path):
        print(f"RFM file not found at {rfm_path}. Skipping.")
        return
        
    rfm = pd.read_csv(rfm_path)
    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    segment_counts["Percentage"] = (segment_counts["Count"] / len(rfm)) * 100
    
    plt.figure(figsize=(12, 7))
    
    # Sort and plot bar chart
    colors = [
        PALETTE["primary"] if s in ["Champions", "Loyal Customers"] else
        PALETTE["secondary"] if s in ["Recent Customers", "About to Sleep"] else
        PALETTE["neutral"] if s in ["Customers Needing Attention"] else
        PALETTE["accent"] for s in segment_counts["Segment"]
    ]
    
    ax = sns.barplot(
        x="Count",
        y="Segment",
        data=segment_counts,
        palette=colors,
        edgecolor=PALETTE["dark"],
        linewidth=0.8
    )
    
    # Add data labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        pct = segment_counts["Percentage"].iloc[i]
        ax.text(
            width + (max(segment_counts["Count"]) * 0.01),
            p.get_y() + p.get_height() / 2,
            f"{int(width):,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.title("Olist Customer Segments Distribution (RFM)", pad=20, fontweight="bold", color=PALETTE["dark"])
    plt.xlabel("Number of Customers", labelpad=10)
    plt.ylabel("Customer Segment", labelpad=10)
    plt.xlim(0, max(segment_counts["Count"]) * 1.15) # Give space for text labels
    plt.tight_layout()
    
    save_path = os.path.join(charts_dir, "02_rfm_segments_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved RFM distribution bar chart to {save_path}")

def plot_time_series_eda():
    """Generate and save historical daily time-series with events highlighted."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    charts_dir = config["paths"]["charts_dir"]
    
    ts_path = os.path.join(proc_dir, "order_timeseries.csv")
    if not os.path.exists(ts_path):
        print(f"Time series file not found at {ts_path}. Skipping.")
        return
        
    df = pd.read_csv(ts_path)
    df["date"] = pd.to_datetime(df["date"])
    
    plt.figure(figsize=(15, 6))
    
    # Plot original time series
    plt.plot(df["date"], df["order_count"], label="Daily Orders Count", color=PALETTE["secondary"], alpha=0.6, linewidth=1)
    
    # Plot 7-day rolling average to see trends clearly
    df["rolling_mean_7"] = df["order_count"].rolling(window=7, center=True).mean()
    plt.plot(df["date"], df["rolling_mean_7"], label="7-Day Moving Average", color=PALETTE["primary"], linewidth=2)
    
    # Highlight events
    # Black Friday
    bf_events = df[df["event_weight"] == 5]
    plt.scatter(bf_events["date"], bf_events["order_count"], color=PALETTE["accent"], s=100, label="Black Friday", zorder=5, marker="*")
    
    # Top holidays
    holiday_events = df[df["event_weight"] == 1]
    plt.scatter(holiday_events["date"], holiday_events["order_count"], color=PALETTE["highlight"], s=40, label="Public Holidays", zorder=4)
    
    plt.title("Olist Daily Order Volume & Special Events (2017 - 2018)", pad=20, fontweight="bold", color=PALETTE["dark"])
    plt.xlabel("Date", labelpad=10)
    plt.ylabel("Number of Orders", labelpad=10)
    plt.legend(loc="upper left", frameon=True, facecolor='white', edgecolor=PALETTE["neutral"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(charts_dir, "03_daily_orders_and_events.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved daily orders and events plot to {save_path}")

def plot_forecast_comparison():
    """Generate and save the forecast comparison plot."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    charts_dir = config["paths"]["charts_dir"]
    
    predictions_path = os.path.join(proc_dir, "final_forecast_predictions.csv")
    if not os.path.exists(predictions_path):
        print(f"Predictions file not found at {predictions_path}. Skipping.")
        return
        
    df_pred = pd.read_csv(predictions_path)
    df_pred["date"] = pd.to_datetime(df_pred["date"])
    df_pred.set_index("date", inplace=True)
    
    plt.figure(figsize=(14, 7))
    
    # Plot Actual
    plt.plot(df_pred.index, df_pred["Actual"], label="Actual", color=PALETTE["dark"], linewidth=2.5, marker="o", markersize=4)
    
    # Plot Models
    model_colors = {
        "baseline": PALETTE["neutral"],
        "sarimax": PALETTE["secondary"],
        "prophet": PALETTE["highlight"],
        "xgboost": PALETTE["accent"],
        "lightgbm": "#2a9d8f" # Emerald Green
    }
    
    model_styles = {
        "baseline": "--",
        "sarimax": "-",
        "prophet": "-",
        "xgboost": "-.",
        "lightgbm": "-."
    }
    
    for model_name in ["baseline", "sarimax", "prophet", "xgboost", "lightgbm"]:
        if model_name in df_pred.columns:
            plt.plot(
                df_pred.index,
                df_pred[model_name],
                label=model_name.upper(),
                color=model_colors[model_name],
                linestyle=model_styles[model_name],
                linewidth=1.8
            )
            
    plt.title("Model Forecasts Comparison (30-Day Testing Period)", pad=20, fontweight="bold", color=PALETTE["dark"])
    plt.xlabel("Date", labelpad=10)
    plt.ylabel("Number of Orders", labelpad=10)
    plt.legend(loc="upper left", frameon=True, facecolor='white', edgecolor=PALETTE["neutral"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(charts_dir, "04_model_forecasts_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved forecast comparison plot to {save_path}")

def plot_xgboost_vs_actual():
    """Generate and save a dedicated comparison plot for Actual vs XGBoost with metrics and error shading."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    charts_dir = config["paths"]["charts_dir"]
    
    predictions_path = os.path.join(proc_dir, "final_forecast_predictions.csv")
    if not os.path.exists(predictions_path):
        print(f"Predictions file not found at {predictions_path}. Skipping.")
        return
        
    df_pred = pd.read_csv(predictions_path)
    df_pred["date"] = pd.to_datetime(df_pred["date"])
    df_pred.set_index("date", inplace=True)
    
    if "xgboost" not in df_pred.columns:
        print("XGBoost predictions not found in final_forecast_predictions.csv. Skipping.")
        return
        
    # Calculate metrics for the test set
    y_true = df_pred["Actual"].values
    y_pred = df_pred["xgboost"].values
    
    from models import compute_metrics
    metrics = compute_metrics(y_true, y_pred)
    
    plt.figure(figsize=(14, 7))
    
    # Plot lines
    plt.plot(df_pred.index, df_pred["Actual"], label="Số lượng đơn hàng thực tế (Actual)", color=PALETTE["dark"], linewidth=2.5, marker="o", markersize=5)
    plt.plot(df_pred.index, df_pred["xgboost"], label="XGBoost Dự đoán (Recursive Prediction)", color=PALETTE["accent"], linewidth=2.5, marker="s", markersize=5)
    
    # Shade the error area between actual and predicted
    plt.fill_between(df_pred.index, df_pred["Actual"], df_pred["xgboost"], color=PALETTE["accent"], alpha=0.15, label="Sai lệch dự báo (Error)")
    
    # Add metrics text box
    textstr = '\n'.join((
        r'$\bf{Chỉ\ số\ đánh\ giá\ XGBoost:}$',
        f'MAPE: {metrics["MAPE"]:.2f}%',
        f'MAE: {metrics["MAE"]:.2f} đơn',
        f'RMSE: {metrics["RMSE"]:.2f} đơn'
    ))
    
    # Position text box in upper left
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=PALETTE["neutral"])
    plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.title("So Sánh Số Lượng Đơn Hàng Thực Tế và XGBoost Dự Đoán (30 Ngày Test Sạch)", pad=20, fontweight="bold", color=PALETTE["dark"])
    plt.xlabel("Ngày đặt hàng", labelpad=10)
    plt.ylabel("Số lượng đơn hàng", labelpad=10)
    plt.legend(loc="upper right", frameon=True, facecolor='white', edgecolor=PALETTE["neutral"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(charts_dir, "05_xgboost_vs_actual.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved XGBoost vs Actual comparison plot to {save_path}")

def run_visualization_pipeline():
    """Main visualization pipeline execution."""
    print("--- Starting Visualization Pipeline ---")
    plot_cohort_heatmap()
    plot_rfm_segments()
    plot_time_series_eda()
    plot_forecast_comparison()
    plot_xgboost_vs_actual()
    print("--- Visualization Pipeline Completed Successfully ---\n")

if __name__ == "__main__":
    run_visualization_pipeline()

