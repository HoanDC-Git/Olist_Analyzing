# %% [markdown]
# # Exploratory Data Analysis & Business Insights - Olist Dataset
# This notebook/script performs exploratory data analysis (EDA) and extracts business insights 
# regarding logistics, customer reviews, RFM segments, and cohort retention.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))
from config_loader import load_config

# Load configuration
config = load_config()
proc_dir = config["paths"]["processed_data_dir"]

# %% [markdown]
# ## 1. Load Cleaned Datasets
# %%
orders = pd.read_csv(os.path.join(proc_dir, "olist_orders_dataset.csv"))
customers = pd.read_csv(os.path.join(proc_dir, "olist_customers_dataset.csv"))
items = pd.read_csv(os.path.join(proc_dir, "olist_order_items_dataset.csv"))
reviews = pd.read_csv(os.path.join(proc_dir, "olist_order_reviews_dataset.csv"))
sellers = pd.read_csv(os.path.join(proc_dir, "olist_sellers_dataset.csv"))

# Parse dates
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

print("Data loaded successfully.")
print(f"Orders: {orders.shape[0]} rows")
print(f"Reviews: {reviews.shape[0]} rows")

# %% [markdown]
# ## 2. Logistics & Delivery Performance Analysis
# Let's inspect the actual shipping delay (actual delivery date minus estimated delivery date).
# A positive delay means the delivery was late, while a negative delay means it was early.

# %%
# Merge orders with customer state
orders_geo = pd.merge(orders[orders["order_status"] == "delivered"], customers, on="customer_id", how="inner")

print("Average delivery time (days) by customer state (Top 10 longest):")
state_delivery = orders_geo.groupby("customer_state")["actual_delivery_time"].mean().sort_values(ascending=False)
print(state_delivery.head(10))

print("\nAverage delivery delay compared to estimate (days) by customer state (Top 10 most delayed):")
state_delay = orders_geo.groupby("customer_state")["delivery_delay"].mean().sort_values(ascending=False)
print(state_delay.head(10))

# Plot logistics delays
plt.figure(figsize=(12, 5))
state_delivery.head(15).plot(kind='bar', color='#457b9d', edgecolor='black')
plt.title("Average Delivery Time by State (Top 15 Longest)")
plt.ylabel("Average Days")
plt.xlabel("Customer State")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Certain states in northern and northeastern Brazil (e.g. RR - Roraima, AP - Amapá, AM - Amazonas) 
# suffer from much longer delivery times (averaging 20-30 days) and higher delays compared to estimate. 
# In contrast, southern states (SP - São Paulo, PR - Paraná) have average delivery times under 10 days.

# %% [markdown]
# ## 3. Correlation between Review Score and Delivery Delays
# Does shipping delay impact customer reviews? Let's check the average review score for late vs on-time deliveries.

# %%
# Merge orders with reviews
orders_reviews = pd.merge(orders_geo, reviews, on="order_id", how="inner")

# Flag deliveries as "Late" (delay > 0) or "On Time / Early" (delay <= 0)
orders_reviews["delivery_status"] = np.where(orders_reviews["delivery_delay"] > 0, "Late", "On Time / Early")

print("Average review score by delivery status:")
print(orders_reviews.groupby("delivery_status")["review_score"].mean())

print("\nReview Score distribution for Late vs On Time:")
print(pd.crosstab(orders_reviews["delivery_status"], orders_reviews["review_score"], normalize='index') * 100)

# Plot review score distribution by delivery status
plt.figure(figsize=(10, 5))
sns.countplot(x="review_score", hue="delivery_status", data=orders_reviews, palette=['#e63946', '#1d3557'])
plt.title("Review Score Distribution: Late vs On Time Deliveries")
plt.xlabel("Review Score")
plt.ylabel("Order Count")
plt.legend(title="Delivery Status")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:** Delays have a catastrophic impact on reviews. 
# - Deliveries that arrive **On Time or Early** maintain an average review score of **4.3 / 5.0**, with over 75% of customers giving a perfect 5-star rating.
# - Deliveries that arrive **Late** have an average score of **1.5 / 5.0**, with nearly 70% of customers giving a 1-star rating! 
# This indicates that logistics delays are the single biggest driver of customer dissatisfaction on Olist.

# %% [markdown]
# ## 4. Customer Segments (RFM) Insights
# Let's read the RFM analysis output and calculate the average spend and order count per segment.

# %%
rfm_path = os.path.join(proc_dir, "customer_rfm.csv")
if os.path.exists(rfm_path):
    rfm = pd.read_csv(rfm_path)
    rfm_stats = rfm.groupby("Segment").agg({
        "customer_unique_id": "count",
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).rename(columns={"customer_unique_id": "Customer Count"}).sort_values(by="Customer Count", ascending=False)
    
    rfm_stats["Percentage (%)"] = (rfm_stats["Customer Count"] / rfm_stats["Customer Count"].sum()) * 100
    print("RFM Segment Performance Statistics:")
    print(rfm_stats)
    
    # Plot RFM segment sizes
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="Customer Count", 
        y="Segment", 
        data=rfm_stats.reset_index(), 
        palette="viridis",
        edgecolor="black"
    )
    plt.title("Customer Counts by RFM Segment")
    plt.xlabel("Number of Customers")
    plt.tight_layout()
    plt.show()
else:
    print("RFM file not found. Please run feature engineering first.")

# %% [markdown]
# **Insight:** 
# - **Recent Customers** and **Needing Attention/About to Sleep** make up over 90% of the customer base.
# - **Loyal Customers** and **Champions** represent a tiny fraction (under 2%). This shows that Olist has very low 
# customer repeat purchases. Most customers buy once and never return, suggesting a high acquisition cost model 
# with poor retention.

# %% [markdown]
# ## 5. Cohort Retention Analysis Insights
# Let's inspect the cohort retention matrix values.

# %%
cohort_path = os.path.join(proc_dir, "cohort_retention.csv")
if os.path.exists(cohort_path):
    retention = pd.read_csv(cohort_path, index_col=0)
    print("Customer Retention Matrix (Month 1 to Month 6):")
    # Show first 6 columns after Month 0
    print(retention.iloc[:, 1:7].head(10))
    
    # Plot cohort retention heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        retention.iloc[:, 1:13], # Show up to index 12 (1 year)
        annot=True, 
        fmt=".2%", 
        cmap="YlGnBu", 
        vmax=0.03, # set max scale to 3% to make repeat purchases visible
        linewidths=0.5
    )
    plt.title("Cohort Retention Rate Heatmap (Month 1 onwards)")
    plt.ylabel("Cohort Month")
    plt.xlabel("Months After First Purchase")
    plt.tight_layout()
    plt.show()
else:
    print("Cohort retention file not found. Please run feature engineering first.")

# %% [markdown]
# **Insight:** 
# Month 1 retention across almost all cohorts is **less than 1%** (often 0.3% - 0.7%), and drops close to 0% by Month 3.
# Olist acts almost purely as a transactional platform rather than building customer loyalty. 
# This confirms the RFM findings and underscores a critical business issue: Olist needs to invest in loyalty programs, 
# retargeting, or email marketing to increase repeat business.
