import os
import pandas as pd
from config_loader import load_config

def clean_products(df):
    """Clean products dataset by filling missing values instead of dropping them."""
    df = df.copy()
    # Fill categorical missing values
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    
    # Fill numerical text attributes
    df["product_name_lenght"] = df["product_name_lenght"].fillna(0).astype(int)
    df["product_description_lenght"] = df["product_description_lenght"].fillna(0).astype(int)
    df["product_photos_qty"] = df["product_photos_qty"].fillna(0).astype(int)
    
    # Fill physical dimensions with category-wise median, falling back to global median
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        category_medians = df.groupby("product_category_name")[col].transform("median")
        df[col] = df[col].fillna(category_medians)
        df[col] = df[col].fillna(df[col].median())
        
    return df

def clean_orders(df, start_date, end_date):
    """Clean orders dataset, parsing datetimes, computing delivery metrics, and filtering dates."""
    df = df.copy()
    
    # Parse datetime columns
    time_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        
    # Filter by date range (on purchase timestamp)
    df = df[(df["order_purchase_timestamp"] >= start_date) & (df["order_purchase_timestamp"] <= end_date)]
    
    # Calculate delivery metrics (only where delivery dates are available)
    mask = df["order_delivered_customer_date"].notnull()
    
    # Actual delivery time: purchase to delivery
    df.loc[mask, "actual_delivery_time"] = (
        df.loc[mask, "order_delivered_customer_date"] - df.loc[mask, "order_purchase_timestamp"]
    ).dt.days
    
    # Estimated delivery time: purchase to estimate
    df.loc[mask, "estimated_delivery_time"] = (
        df.loc[mask, "order_estimated_delivery_date"] - df.loc[mask, "order_purchase_timestamp"]
    ).dt.days
    
    # Delivery delay: customer delivery minus estimated (positive means late, negative means early)
    df.loc[mask, "delivery_delay"] = (
        df.loc[mask, "order_delivered_customer_date"] - df.loc[mask, "order_estimated_delivery_date"]
    ).dt.days
    
    return df

def clean_reviews(df):
    """Clean reviews dataset by filling empty comments/titles."""
    df = df.copy()
    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")
    return df

def clean_geolocation(df):
    """Aggregate geolocation by zip code prefix to remove duplicates and compress the data."""
    df = df.copy()
    # Normalize city names to lowercase and strip whitespaces
    df["geolocation_city"] = df["geolocation_city"].astype(str).str.lower().str.strip()
    
    # Group by zip code prefix and get mean coordinates and first city/state name
    geo_agg = df.groupby("geolocation_zip_code_prefix").agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
        "geolocation_city": "first",
        "geolocation_state": "first"
    }).reset_index()
    
    return geo_agg

def run_cleaning_pipeline():
    """Main cleaning pipeline execution."""
    config = load_config()
    raw_dir = config["paths"]["raw_data_dir"]
    proc_dir = config["paths"]["processed_data_dir"]
    
    start_date = config["cleaning"]["start_date"]
    end_date = config["cleaning"]["end_date"]
    
    print("--- Starting Data Cleaning Pipeline ---")
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {proc_dir}")
    print(f"Filtering dates: {start_date} to {end_date}")
    
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".csv"):
            continue
            
        raw_path = os.path.join(raw_dir, filename)
        proc_path = os.path.join(proc_dir, filename)
        
        df = pd.read_csv(raw_path)
        initial_rows = len(df)
        
        print(f"Processing {filename}...")
        
        if "olist_products_dataset" in filename:
            df_cleaned = clean_products(df)
        elif "olist_orders_dataset" in filename:
            df_cleaned = clean_orders(df, start_date, end_date)
        elif "olist_order_reviews_dataset" in filename:
            df_cleaned = clean_reviews(df)
        elif "olist_geolocation_dataset" in filename:
            df_cleaned = clean_geolocation(df)
        else:
            # For other tables, just copy them without rows dropped
            df_cleaned = df.copy()
            
        cleaned_rows = len(df_cleaned)
        print(f"  Rows before: {initial_rows} | Rows after: {cleaned_rows} | Removed: {initial_rows - cleaned_rows}")
        
        df_cleaned.to_csv(proc_path, index=False)
        print(f"  Saved cleaned file to {proc_path}\n")
        
    print("--- Data Cleaning Pipeline Completed Successfully ---\n")

if __name__ == "__main__":
    run_cleaning_pipeline()
