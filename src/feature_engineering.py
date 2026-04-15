# ============================================================
# src/feature_engineering.py
# Feature engineering for the forecasting model (FIXED)
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, save_csv

logger = get_logger(__name__)


# -----------------------------------------------
# 1. Date / Time Features
# -----------------------------------------------
def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"]         = df[date_col].dt.year
    df["month"]        = df[date_col].dt.month
    df["day"]          = df[date_col].dt.day
    df["day_of_week"]  = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"]      = df[date_col].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"]   = df[date_col].dt.is_month_end.astype(int)
    df["day_of_year"]  = df[date_col].dt.dayofyear

    logger.info("✅ Date features added.")
    return df


# -----------------------------------------------
# 2. Lag Features
# -----------------------------------------------
def add_lag_features(df: pd.DataFrame, target_col: str = "actual_sales", lags=[1,7,14,30]):
    df = df.copy().reset_index(drop=True)
    df = df.sort_values(["store", "product", "date"])

    for lag in lags:
        df[f"lag_{lag}d"] = df.groupby(["store","product"])[target_col].shift(lag)

    logger.info("✅ Lag features added.")
    return df


# -----------------------------------------------
# 3. Rolling Features (🔥 FIXED HERE)
# -----------------------------------------------
def add_rolling_features(df: pd.DataFrame, target_col="actual_sales", windows=[7,14,30]):
    df = df.copy().reset_index(drop=True)
    df = df.sort_values(["store", "product", "date"])

    for w in windows:
        df[f"rolling_mean_{w}d"] = (
            df.groupby(["store","product"])[target_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

        df[f"rolling_std_{w}d"] = (
            df.groupby(["store","product"])[target_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )

    logger.info("✅ Rolling features added.")
    return df


# -----------------------------------------------
# 4. Expanding Features (🔥 FIXED HERE)
# -----------------------------------------------
def add_expanding_features(df: pd.DataFrame, target_col="actual_sales"):
    df = df.copy().reset_index(drop=True)
    df = df.sort_values(["store", "product", "date"])

    df["expanding_mean"] = (
        df.groupby(["store","product"])[target_col]
        .transform(lambda x: x.expanding().mean())
    )

    logger.info("✅ Expanding mean feature added.")
    return df


# -----------------------------------------------
# 5. Promotion Features
# -----------------------------------------------
def add_promotion_features(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)

    df["promo_x_weekend"] = df["is_promotion"] * df.get("is_weekend", 0)
    df["promo_x_month"]   = df["is_promotion"] * df.get("month", 1)

    df["promo_lag1"] = (
        df.groupby(["store","product"])["is_promotion"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    logger.info("✅ Promotion features added.")
    return df


# -----------------------------------------------
# 6. Encoding
# -----------------------------------------------
def encode_categoricals(df: pd.DataFrame):
    df = df.copy()

    store_map    = {s:i for i,s in enumerate(df["store"].unique())}
    category_map = {c:i for i,c in enumerate(df["category"].unique())}
    product_map  = {p:i for i,p in enumerate(df["product"].unique())}

    df["store_enc"]    = df["store"].map(store_map)
    df["category_enc"] = df["category"].map(category_map)
    df["product_enc"]  = df["product"].map(product_map)

    logger.info("✅ Encoding done.")
    return df, {"store":store_map,"category":category_map,"product":product_map}


# -----------------------------------------------
# 7. Demand Segmentation
# -----------------------------------------------
def segment_demand(df: pd.DataFrame, target_col="actual_sales"):
    df = df.copy()
    result = []

    for (store, product), group in df.groupby(["store","product"]):
        avg_sales = group[target_col].mean()
        zero_pct  = (group[target_col]==0).mean()

        if zero_pct > 0.4:
            seg = "INTERMITTENT"
        elif avg_sales >= group[target_col].quantile(0.75):
            seg = "HIGH"
        elif avg_sales <= group[target_col].quantile(0.25):
            seg = "LOW"
        else:
            seg = "MEDIUM"

        group = group.copy()
        group["demand_segment"] = seg
        result.append(group)

    df = pd.concat(result, ignore_index=True)
    logger.info("✅ Demand segmentation done.")
    return df


# -----------------------------------------------
# MASTER PIPELINE
# -----------------------------------------------
def run_feature_engineering(processed_path="data/processed_sales_data.csv",
                           features_path="data/features_data.csv"):

    from src.utils import load_csv
    df = load_csv(processed_path, parse_dates=["date"])

    df = df.reset_index(drop=True)   # ✅ CRITICAL FIX

    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_expanding_features(df)
    df = add_promotion_features(df)
    df, enc_maps = encode_categoricals(df)
    df = segment_demand(df)

    df.dropna(subset=["lag_7d","rolling_mean_7d"], inplace=True)

    df = df.sort_values(["store","product","date"]).reset_index(drop=True)

    save_csv(df, features_path, label="Feature Engineered Data")

    logger.info(f"✅ Feature engineering complete. Shape: {df.shape}")
    return df, enc_maps


if __name__ == "__main__":
    df, maps = run_feature_engineering()
    print(df.head())