# ============================================================
# src/preprocessing.py
# Data cleaning, validation, and preprocessing pipeline
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, save_csv, load_csv, basic_stats

logger = get_logger(__name__)


# -----------------------------------------------
# 1. Load & Basic Validation
# -----------------------------------------------
def load_and_validate(path: str) -> pd.DataFrame:
    """
    Loads the raw CSV and performs basic structural checks.
    """
    df = load_csv(path, parse_dates=["date"])

    required_cols = [
        "date", "store", "category", "product",
        "unit_price", "actual_sales", "revenue",
        "current_stock", "stockout_flag", "is_promotion"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    logger.info(f"✅ All required columns present.")
    return df


# -----------------------------------------------
# 2. Handle Missing Values
# -----------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and fills/drops missing values appropriately.
    - Numeric cols  → fill with median
    - String cols   → fill with 'Unknown'
    - Date col      → drop rows with null dates
    """
    df = df.copy()

    # Drop rows where date is missing
    before = len(df)
    df.dropna(subset=["date"], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"⚠️  Dropped {dropped} rows with null dates.")

    # Fill numeric nulls
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"   Filled {null_count} nulls in '{col}' with median ({median_val:.2f})")

    # Fill string nulls
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col].fillna("Unknown", inplace=True)
            logger.info(f"   Filled {null_count} nulls in '{col}' with 'Unknown'")

    logger.info(f"✅ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


# -----------------------------------------------
# 3. Fix Data Types
# -----------------------------------------------
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures correct data types for all columns.
    """
    df = df.copy()

    df["date"]          = pd.to_datetime(df["date"])
    df["actual_sales"]  = df["actual_sales"].astype(int)
    df["demand"]        = df["demand"].astype(int)
    df["current_stock"] = df["current_stock"].astype(int)
    df["stockout_flag"] = df["stockout_flag"].astype(int)
    df["is_promotion"]  = df["is_promotion"].astype(int)
    df["unit_price"]    = df["unit_price"].astype(float)
    df["revenue"]       = df["revenue"].astype(float)

    logger.info("✅ Data types fixed.")
    return df


# -----------------------------------------------
# 4. Remove Duplicates
# -----------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows (same store+product+date = duplicate).
    """
    subset = ["date", "store", "product"]
    before = len(df)
    df.drop_duplicates(subset=subset, keep="first", inplace=True)
    removed = before - len(df)
    logger.info(f"✅ Removed {removed} duplicate rows. Remaining: {len(df):,}")
    return df


# -----------------------------------------------
# 5. Handle Outliers (IQR Method)
# -----------------------------------------------
def cap_outliers(df: pd.DataFrame, col: str, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Caps outliers using IQR * multiplier method.
    Values above upper bound are clipped to upper bound.
    """
    df = df.copy()
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    before_count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    logger.info(f"   Capped {before_count} outliers in '{col}' → [{lower:.1f}, {upper:.1f}]")
    return df


# -----------------------------------------------
# 6. Sort & Reset Index
# -----------------------------------------------
def sort_and_reset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store", "product", "date"]).reset_index(drop=True)
    logger.info("✅ Data sorted by [store → product → date].")
    return df


# -----------------------------------------------
# 7. Aggregate Daily Data
# -----------------------------------------------
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data to daily level per store+product.
    Useful for forecasting models.
    """
    agg = df.groupby(["date", "store", "category", "product"]).agg(
        actual_sales   = ("actual_sales",   "sum"),
        demand         = ("demand",         "sum"),
        revenue        = ("revenue",        "sum"),
        current_stock  = ("current_stock",  "last"),
        is_promotion   = ("is_promotion",   "max"),
        stockout_flag  = ("stockout_flag",  "max"),
        unit_price     = ("unit_price",     "first"),
        discount_pct   = ("discount_pct",   "mean"),
    ).reset_index()

    logger.info(f"✅ Aggregated to {len(agg):,} daily records.")
    return agg


# -----------------------------------------------
# 8. Missing Date Gaps Filler
# -----------------------------------------------
def fill_date_gaps(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Fills missing dates within each store-product group.
    Ensures a continuous time series without gaps.
    """
    dfs = []
    groups = df.groupby(["store", "product"])

    for (store, product), group in groups:
        group = group.set_index("date").asfreq(freq)
        group["store"]    = store
        group["product"]  = product
        # Forward-fill metadata, zero-fill sales
        meta_cols  = ["category", "unit_price"]
        sales_cols = ["actual_sales", "demand", "revenue",
                      "is_promotion", "stockout_flag", "discount_pct"]

        group[meta_cols]  = group[meta_cols].ffill()
        group[sales_cols] = group[sales_cols].fillna(0)
        group["current_stock"] = group["current_stock"].ffill()
        dfs.append(group.reset_index())

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Date gaps filled. New shape: {result.shape}")
    return result


# -----------------------------------------------
# MASTER PREPROCESSING PIPELINE
# -----------------------------------------------
def run_preprocessing_pipeline(
    raw_path:       str = "data/raw_sales_data.csv",
    processed_path: str = "data/processed_sales_data.csv"
) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline:
    1. Load & validate
    2. Handle missing values
    3. Fix data types
    4. Remove duplicates
    5. Cap outliers
    6. Sort & reset index
    7. Save processed data
    """
    logger.info("="*50)
    logger.info("  PREPROCESSING PIPELINE STARTED")
    logger.info("="*50)

    df = load_and_validate(raw_path)

    logger.info("\n--- Step 1: Missing Values ---")
    df = handle_missing_values(df)

    logger.info("\n--- Step 2: Fix Data Types ---")
    df = fix_dtypes(df)

    logger.info("\n--- Step 3: Remove Duplicates ---")
    df = remove_duplicates(df)

    logger.info("\n--- Step 4: Cap Outliers ---")
    for col in ["actual_sales", "demand", "revenue"]:
        df = cap_outliers(df, col)

    logger.info("\n--- Step 5: Sort & Reset ---")
    df = sort_and_reset(df)

    logger.info("\n--- Step 6: Fill Date Gaps ---")
    df = fill_date_gaps(df)

    save_csv(df, processed_path, label="Processed Data")

    # Print preprocessing summary
    print("\n" + "="*55)
    print("  PREPROCESSING COMPLETE")
    print("="*55)
    stats = basic_stats(df)
    print(stats[["count", "mean", "std", "min", "max", "missing", "missing_%"]].head(10))
    print("="*55)

    logger.info("✅ Preprocessing pipeline completed successfully.")
    return df


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(f"\nFinal shape: {df.shape}")
    print(df.head(3))
