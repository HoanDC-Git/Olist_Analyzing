import os
import pandas as pd
import numpy as np
from config_loader import load_config

def calculate_delivery_delays():
    """Calculate exact delivery delays and their correlation with review scores."""
    config = load_config()
    proc_dir = config["paths"]["processed_data_dir"]
    
    print("Running Deep Dive Analysis: Delivery Delays...")
    
    orders = pd.read_csv(os.path.join(proc_dir, "olist_orders_dataset.csv"))
    reviews = pd.read_csv(os.path.join(proc_dir, "olist_order_reviews_dataset.csv"))
    
    # Filter delivered orders
    delivered = orders[orders["order_status"] == "delivered"].copy()
    
    # Convert dates
    delivered["order_delivered_customer_date"] = pd.to_datetime(delivered["order_delivered_customer_date"])
    delivered["order_estimated_delivery_date"] = pd.to_datetime(delivered["order_estimated_delivery_date"])
    
    # Calculate delay in days (positive means late, negative means early)
    delivered["delivery_delay_days"] = (delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]).dt.total_seconds() / (24 * 3600)
    
    # Identify late orders
    delivered["is_late"] = delivered["delivery_delay_days"] > 0
    
    # Join with reviews
    order_reviews = pd.merge(delivered, reviews, on="order_id", how="inner")
    
    # Calculate average review score by delay bins
    bins = [-np.inf, 0, 3, 7, 14, 30, np.inf]
    labels = ["Early/On Time", "1-3 days late", "4-7 days late", "8-14 days late", "15-30 days late", "> 30 days late"]
    order_reviews["delay_bin"] = pd.cut(order_reviews["delivery_delay_days"], bins=bins, labels=labels)
    
    delay_impact = order_reviews.groupby("delay_bin")["review_score"].agg(["mean", "count"]).reset_index()
    delay_impact.rename(columns={"mean": "average_review_score", "count": "number_of_reviews"}, inplace=True)
    
    output_path = os.path.join(proc_dir, "deep_dive_delivery_delays.csv")
    delay_impact.to_csv(output_path, index=False)
    
    print("  Delivery Delay Impact on Reviews:")
    print(delay_impact)
    print(f"  Saved to {output_path}")
    
def analyze_product_defects():
    """Analyze which product categories receive the most negative reviews (1-2 stars)."""
    config = load_config()
    raw_dir = config["paths"]["raw_data_dir"]
    proc_dir = config["paths"]["processed_data_dir"]
    
    print("Running Deep Dive Analysis: Product Defects by Category...")
    
    order_items = pd.read_csv(os.path.join(raw_dir, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "olist_products_dataset.csv"))
    reviews = pd.read_csv(os.path.join(proc_dir, "olist_order_reviews_dataset.csv"))
    translation = pd.read_csv(os.path.join(raw_dir, "product_category_name_translation.csv"))
    
    # Merge products with english translations
    products_eng = pd.merge(products, translation, on="product_category_name", how="left")
    
    # Fill missing translations with original portuguese
    products_eng["product_category_name_english"] = products_eng["product_category_name_english"].fillna(products_eng["product_category_name"])
    
    # Merge items and products
    item_products = pd.merge(order_items, products_eng, on="product_id", how="inner")
    
    # Merge with reviews
    item_reviews = pd.merge(item_products, reviews, on="order_id", how="inner")
    
    # Filter for negative reviews (1 or 2 stars)
    negative_reviews = item_reviews[item_reviews["review_score"] <= 2]
    
    # Count negative reviews by category
    category_complaints = negative_reviews.groupby("product_category_name_english")["review_id"].count().reset_index()
    category_complaints.rename(columns={"review_id": "negative_review_count"}, inplace=True)
    
    # Count total sales by category to calculate defect rate
    category_sales = item_reviews.groupby("product_category_name_english")["order_item_id"].count().reset_index()
    category_sales.rename(columns={"order_item_id": "total_items_sold"}, inplace=True)
    
    defect_analysis = pd.merge(category_sales, category_complaints, on="product_category_name_english", how="left").fillna(0)
    defect_analysis["negative_review_rate"] = (defect_analysis["negative_review_count"] / defect_analysis["total_items_sold"]) * 100
    
    # Filter for categories with at least 100 sales to avoid noise
    defect_analysis = defect_analysis[defect_analysis["total_items_sold"] >= 100]
    
    # Sort by defect rate descending
    defect_analysis = defect_analysis.sort_values(by="negative_review_rate", ascending=False).head(20)
    
    output_path = os.path.join(proc_dir, "deep_dive_product_defects.csv")
    defect_analysis.to_csv(output_path, index=False)
    
    print("  Top 10 Categories with Highest Negative Review Rates:")
    print(defect_analysis.head(10))
    print(f"  Saved to {output_path}")

def run_deep_dive_analysis():
    print("--- Starting Deep Dive Analysis ---")
    calculate_delivery_delays()
    analyze_product_defects()
    print("--- Deep Dive Analysis Completed Successfully ---\n")

if __name__ == "__main__":
    run_deep_dive_analysis()
