import os
import calendar
import pandas as pd
import numpy as np
from datetime import datetime
from config_loader import load_config

def get_black_friday_dates(years):
    """Calculate Black Friday dates for the given years (4th Friday of November)."""
    black_fridays = set()
    for year in years:
        november = 11
        cal = calendar.monthcalendar(year, november)
        fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
        if len(fridays) >= 4:
            black_friday = fridays[3]
            black_friday_date = datetime(year, november, black_friday).strftime('%Y-%m-%d')
            black_fridays.add(black_friday_date)
    return black_fridays

def create_time_series_data():
    """Load orders, filter for delivered, resample daily, and add event weights."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    orders_path = os.path.join(proc_dir, "olist_orders_dataset.csv")
    holiday_path = os.path.join(config["paths"]["raw_data_dir"], "Brazil_holiday.csv")
    output_path = os.path.join(proc_dir, "order_timeseries.csv")
    
    print("Creating daily time-series dataset...")
    
    # Read orders
    df = pd.read_csv(orders_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    
    # Filter for delivered orders (consistent with standard sales forecasting)
    df_delivered = df[df["order_status"] == "delivered"]
    
    # Resample to daily order counts
    df_daily = df_delivered.set_index("order_purchase_timestamp").resample("D")["order_id"].count()
    df_daily.name = "order_count"
    df_ts = df_daily.reset_index()
    df_ts.rename(columns={"order_purchase_timestamp": "date"}, inplace=True)
    
    # Extract years in dataset
    years_in_data = df_ts["date"].dt.year.unique()
    
    # Load holidays
    holidays_set = set()
    if os.path.exists(holiday_path):
        holiday_df = pd.read_csv(holiday_path)
        # Parse holidays
        holidays_set.update(pd.to_datetime(holiday_df["date"]).dt.strftime('%Y-%m-%d').tolist())
        print(f"  Loaded {len(holidays_set)} holidays from {holiday_path}")
    else:
        print(f"  Warning: Holiday file not found at {holiday_path}")
        
    # Get Black Friday dates
    bf_set = get_black_friday_dates(years_in_data)
    print(f"  Calculated Black Fridays: {bf_set}")
    
    # Initialize event weight (0 for normal, 1 for holiday, 5 for Black Friday)
    df_ts["event_weight"] = 0
    date_strings = df_ts["date"].dt.strftime('%Y-%m-%d')
    
    df_ts.loc[date_strings.isin(holidays_set), "event_weight"] = 1
    df_ts.loc[date_strings.isin(bf_set), "event_weight"] = 5
    
    # Save processed timeseries
    df_ts.to_csv(output_path, index=False)
    print(f"  Saved time-series to {output_path}")
    print(df_ts["event_weight"].value_counts().sort_index())
    
    return df_ts

def run_rfm_analysis():
    """Perform RFM Customer Segmentation."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    
    print("Running RFM Customer Segmentation...")
    
    # Load required tables
    customers = pd.read_csv(os.path.join(proc_dir, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(proc_dir, "olist_orders_dataset.csv"))
    payments = pd.read_csv(os.path.join(proc_dir, "olist_order_payments_dataset.csv"))
    
    # Convert orders dates
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    
    # Merge datasets: Customer unique ID -> Order ID -> Payment Value
    # 1. Merge orders with customer unique ID
    order_cust = pd.merge(orders, customers, on="customer_id", how="inner")
    
    # 2. Sum payments per order (an order can have multiple payment methods)
    order_payments = payments.groupby("order_id")["payment_value"].sum().reset_index()
    
    # 3. Merge order-customer with order-payments
    rfm_base = pd.merge(order_cust, order_payments, on="order_id", how="inner")
    
    # Define reference date as max purchase date + 1 day
    ref_date = rfm_base["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    
    # Aggregate to customer level
    rfm = rfm_base.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (ref_date - x.max()).days, # Recency
        "order_id": "nunique",                                          # Frequency
        "payment_value": "sum"                                          # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        "order_purchase_timestamp": "Recency",
        "order_id": "Frequency",
        "payment_value": "Monetary"
    }, inplace=True)
    
    # Calculate RFM Scores (1-5) using quantiles
    # For frequency, most customers have only 1 purchase, so quantiles will collapse.
    # We will use custom bins or rank with method='first' to avoid duplicate bin issues.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    
    # Handle frequency scoring: if frequency is 1, score is 1. If >1, we score higher.
    def score_frequency(freq):
        if freq == 1:
            return 1
        elif freq == 2:
            return 3
        elif freq == 3:
            return 4
        else:
            return 5
            
    rfm["F_Score"] = rfm["Frequency"].apply(score_frequency)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    
    # Combine scores to form string RFM segment score
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    
    # Map RFM scores to Customer Segments
    # We map using R and F scores (the most standard way)
    def segment_customer(row):
        r = int(row["R_Score"])
        f = int(row["F_Score"])
        
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f < 3:
            return "Recent Customers"
        elif r == 3 and f < 3:
            return "About to Sleep"
        elif r == 2:
            return "Customers Needing Attention"
        elif r == 1 and f >= 3:
            return "Can't Lose Them"
        elif r == 1 and f == 2:
            return "At Risk"
        else:
            return "Lost"
            
    rfm["Segment"] = rfm.apply(segment_customer, axis=1)
    
    # Save RFM results
    rfm_path = os.path.join(proc_dir, "customer_rfm.csv")
    rfm.to_csv(rfm_path, index=False)
    print(f"  Saved RFM results to {rfm_path}")
    print(rfm["Segment"].value_counts())
    
    return rfm

def run_cohort_analysis():
    """Perform Cohort Retention Analysis."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    
    print("Running Cohort Retention Analysis...")
    
    # Load required tables
    customers = pd.read_csv(os.path.join(proc_dir, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(proc_dir, "olist_orders_dataset.csv"))
    
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    
    # Merge orders with customers on customer_id
    df = pd.merge(orders, customers, on="customer_id", how="inner")
    
    # Keep only purchase date and customer unique ID
    df = df[["customer_unique_id", "order_purchase_timestamp"]].copy()
    
    # Create order month column
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    
    # Determine the first purchase month (Cohort Month) for each customer
    df["cohort_month"] = df.groupby("customer_unique_id")["order_month"].transform("min")
    
    # Calculate cohort index (number of months between first purchase and subsequent purchase)
    # Subtracting two Period objects returns the difference in months (as an integer offset)
    df["cohort_index"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)
    
    # Group by Cohort Month and Cohort Index, count unique customers
    cohort_data = df.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().reset_index()
    
    # Pivot the data to create a retention matrix
    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_unique_id")
    
    # Calculate retention percentage: divide each column by the cohort size (index 0)
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
    
    # Save Cohort Results
    cohort_path = os.path.join(proc_dir, "cohort_retention.csv")
    retention_matrix.to_csv(cohort_path)
    print(f"  Saved Cohort Retention Matrix to {cohort_path}")
    print("  Cohort Sizes:")
    print(cohort_sizes)
    
    return cohort_sizes, retention_matrix

def run_all_feature_engineering():
    """Run the entire feature engineering pipeline."""
    print("--- Starting Feature Engineering Pipeline ---")
    create_time_series_data()
    run_rfm_analysis()
    run_cohort_analysis()
    print("--- Feature Engineering Pipeline Completed Successfully ---\n")

if __name__ == "__main__":
    run_all_feature_engineering()
