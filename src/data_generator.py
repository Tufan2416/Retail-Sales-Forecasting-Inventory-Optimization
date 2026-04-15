# ============================================================
# src/data_generator.py
# Generates a realistic synthetic retail sales dataset
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, save_csv, ensure_dirs

logger = get_logger(__name__)


# -----------------------------------------------
# Product Catalog
# -----------------------------------------------
PRODUCTS = {
    "Electronics": [
        ("Smartphone",    15000, 25),
        ("Laptop",        45000, 10),
        ("Earbuds",        2500, 60),
        ("Smart Watch",    8000, 30),
    ],
    "Grocery": [
        ("Rice (5kg)",      250, 200),
        ("Cooking Oil (1L)",150, 180),
        ("Wheat Flour (5kg)",200, 220),
        ("Dal (1kg)",       120, 250),
    ],
    "Clothing": [
        ("Men T-Shirt",    599,  80),
        ("Women Kurti",    899,  70),
        ("Kids Jeans",     799,  50),
        ("Sports Shoes",  1999,  40),
    ],
    "Home & Kitchen": [
        ("Pressure Cooker",1500, 30),
        ("Non-Stick Pan",   800, 45),
        ("Water Bottle",    350, 90),
        ("Mixer Grinder",  3500, 20),
    ],
    "Personal Care": [
        ("Shampoo (200ml)", 180, 150),
        ("Face Wash",       120, 160),
        ("Moisturizer",     250, 100),
        ("Toothpaste",       80, 300),
    ],
}

STORES = ["Store_Delhi", "Store_Mumbai", "Store_Bangalore",
          "Store_Chennai", "Store_Hyderabad"]


# -----------------------------------------------
# Seasonality & Promotion Simulation
# -----------------------------------------------
def get_seasonality_factor(date: pd.Timestamp, category: str) -> float:
    """
    Returns a multiplier based on month/season and category.
    Simulates festival seasons, summer peaks, etc.
    """
    month = date.month
    day_of_week = date.dayofweek

    # Base weekend boost
    weekend_boost = 1.25 if day_of_week >= 5 else 1.0

    # Category-specific seasonality
    season_map = {
        "Electronics": {10: 1.8, 11: 2.2, 12: 2.0, 1: 1.3},    # Diwali, Xmas, New Year
        "Grocery":     {10: 1.4, 11: 1.3, 4: 1.2, 8: 1.2},      # Festivals, summer
        "Clothing":    {10: 1.9, 11: 1.8, 2: 1.5, 9: 1.4},      # Festive, New Year, Navratri
        "Home & Kitchen": {10: 1.6, 11: 1.5, 1: 1.3},
        "Personal Care": {4: 1.3, 5: 1.3, 6: 1.2},              # Summer skincare
    }

    cat_boost = season_map.get(category, {}).get(month, 1.0)
    return cat_boost * weekend_boost


def simulate_promotion(date: pd.Timestamp) -> tuple:
    """
    Returns (is_promotion, discount_pct) for a given date.
    Sale events: Republic Day, Holi, Independence Day, Diwali, End-of-season
    """
    month, day = date.month, date.day

    # Simulate major sale periods
    sale_periods = [
        (1, 26, 0.20),   # Republic Day Sale
        (3, 10, 0.15),   # Holi Sale
        (8, 15, 0.18),   # Independence Day Sale
        (10, 24, 0.30),  # Diwali Sale
        (11, 11, 0.35),  # Singles Day / Big Billion
        (12, 25, 0.25),  # Christmas Sale
    ]
    for (m, d, disc) in sale_periods:
        if month == m and abs(day - d) <= 3:
            return 1, disc

    # Random weekend promo (10% probability)
    if date.dayofweek >= 5 and np.random.rand() < 0.10:
        return 1, round(np.random.uniform(0.05, 0.15), 2)

    return 0, 0.0


# -----------------------------------------------
# Main Dataset Generator
# -----------------------------------------------
def generate_retail_dataset(
    start_date: str = "2022-01-01",
    end_date: str   = "2024-12-31",
    seed: int       = 42,
    output_path: str = "data/raw_sales_data.csv"
) -> pd.DataFrame:
    """
    Generates a synthetic retail sales dataset with:
    - Multiple stores, products, categories
    - Seasonal demand patterns
    - Promotions / discounts
    - Stockout simulation
    - Lead time & inventory tracking
    """
    np.random.seed(seed)
    ensure_dirs(["data"])

    dates      = pd.date_range(start=start_date, end=end_date, freq="D")
    records    = []

    logger.info(f"🔄 Generating data from {start_date} to {end_date} ...")
    logger.info(f"   Stores: {len(STORES)} | Date range: {len(dates)} days")

    total_products = sum(len(v) for v in PRODUCTS.values())
    logger.info(f"   Products: {total_products}")

    for store in STORES:
        for category, product_list in PRODUCTS.items():
            for (product, unit_price, base_demand) in product_list:

                # Simulate current stock (start random)
                current_stock = int(np.random.uniform(base_demand * 2, base_demand * 5))
                reorder_point = int(base_demand * 1.5)
                lead_time     = np.random.randint(5, 10)

                for date in dates:
                    # Seasonality + noise
                    season_factor  = get_seasonality_factor(date, category)
                    noise          = np.random.normal(1.0, 0.15)
                    raw_demand     = base_demand * season_factor * max(noise, 0.1)

                    is_promo, discount_pct = simulate_promotion(date)

                    # Promotions boost demand
                    if is_promo:
                        raw_demand *= (1 + discount_pct * 2)

                    demand = int(round(raw_demand))

                    # Stockout logic
                    actual_sales  = min(demand, current_stock)
                    lost_sales    = demand - actual_sales
                    stockout_flag = 1 if lost_sales > 0 else 0

                    # Update stock
                    current_stock -= actual_sales

                    # Reorder simulation
                    reorder_qty = 0
                    if current_stock <= reorder_point:
                        reorder_qty    = base_demand * lead_time * 2
                        current_stock += int(reorder_qty)

                    # Revenue
                    effective_price = unit_price * (1 - discount_pct)
                    revenue         = actual_sales * effective_price

                    records.append({
                        "date":           date,
                        "store":          store,
                        "category":       category,
                        "product":        product,
                        "unit_price":     unit_price,
                        "discount_pct":   discount_pct,
                        "effective_price":round(effective_price, 2),
                        "demand":         demand,
                        "actual_sales":   actual_sales,
                        "lost_sales":     lost_sales,
                        "revenue":        round(revenue, 2),
                        "current_stock":  current_stock,
                        "reorder_point":  reorder_point,
                        "reorder_qty":    reorder_qty,
                        "lead_time":      lead_time,
                        "is_promotion":   is_promo,
                        "stockout_flag":  stockout_flag,
                    })

    df = pd.DataFrame(records)
    save_csv(df, output_path, label="Raw Retail Dataset")
    logger.info(f"✅ Dataset shape: {df.shape}")
    return df


# -----------------------------------------------
# Quick Summary Printer
# -----------------------------------------------
def print_dataset_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Total Records   : {len(df):,}")
    print(f"  Date Range      : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stores          : {df['store'].nunique()}")
    print(f"  Categories      : {df['category'].nunique()}")
    print(f"  Products        : {df['product'].nunique()}")
    print(f"  Total Revenue   : ₹{df['revenue'].sum():,.0f}")
    print(f"  Total Units Sold: {df['actual_sales'].sum():,}")
    print(f"  Stockout Events : {df['stockout_flag'].sum():,}")
    print(f"  Promotions Days : {df['is_promotion'].sum():,}")
    print("="*55)


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    df = generate_retail_dataset()
    print_dataset_summary(df)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
