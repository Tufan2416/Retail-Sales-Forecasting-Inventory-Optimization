# ============================================================
# src/utils.py
# Utility functions used across the project
# ============================================================

import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------------------------
# 1. Logger Setup
# -----------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with console output.
    Usage: logger = get_logger(__name__)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)


# -----------------------------------------------
# 2. Config Loader
# -----------------------------------------------
def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Loads the YAML configuration file and returns it as a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# -----------------------------------------------
# 3. Directory Creator
# -----------------------------------------------
def ensure_dirs(paths: list):
    """
    Creates directories if they don't exist.
    Pass a list of folder paths.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


# -----------------------------------------------
# 4. Save DataFrame to CSV
# -----------------------------------------------
def save_csv(df: pd.DataFrame, path: str, label: str = ""):
    """
    Saves a DataFrame to CSV and logs the action.
    """
    logger = get_logger("utils")
    df.to_csv(path, index=False)
    logger.info(f"✅ Saved {label} → {path} | Shape: {df.shape}")


# -----------------------------------------------
# 5. Load CSV
# -----------------------------------------------
def load_csv(path: str, parse_dates: list = None) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.
    Optionally parses date columns.
    """
    logger = get_logger("utils")
    if parse_dates:
        df = pd.read_csv(path, parse_dates=parse_dates)
    else:
        df = pd.read_csv(path)
    logger.info(f"📂 Loaded {path} | Shape: {df.shape}")
    return df


# -----------------------------------------------
# 6. Compute Basic Stats
# -----------------------------------------------
def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns basic statistics + missing value count.
    """
    stats = df.describe(include="all").T
    stats["missing"] = df.isnull().sum()
    stats["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    return stats


# -----------------------------------------------
# 7. Date Feature Extractor
# -----------------------------------------------
def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extracts date-related features from a datetime column:
    year, month, day, day_of_week, week_of_year, quarter, is_weekend
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"]         = df[date_col].dt.year
    df["month"]        = df[date_col].dt.month
    df["day"]          = df[date_col].dt.day
    df["day_of_week"]  = df[date_col].dt.dayofweek        # 0=Mon, 6=Sun
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"]      = df[date_col].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    return df


# -----------------------------------------------
# 8. Metric Printer
# -----------------------------------------------
def print_metrics(metrics, title):
    print("\n" + "="*40)
    print(title)
    print("="*40)

    for k, v in metrics.items():
        try:
            # Try formatting as float
            print(f"{k:<20}: {float(v):.4f}")
        except:
            # If not numeric, print as is
            print(f"{k:<20}: {v}")


# -----------------------------------------------
# 9. Z-Score for Service Level
# -----------------------------------------------
def get_z_score(service_level: float) -> float:
    """
    Returns the Z-score for a given service level.
    Common values:
        90% → 1.28
        95% → 1.65
        99% → 2.33
    """
    from scipy import stats
    return stats.norm.ppf(service_level)


# -----------------------------------------------
# 10. Timestamp for file naming
# -----------------------------------------------
def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
