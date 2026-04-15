# ============================================================
# src/forecasting.py
# Sales forecasting using Random Forest, XGBoost, and ARIMA
# Model comparison + prediction intervals
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import warnings
import joblib
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, save_csv, load_csv, print_metrics, ensure_dirs

logger = get_logger(__name__)

# -----------------------------------------------
# ML Libraries
# -----------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  XGBoost not installed. Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Statsmodels not installed. Skipping ARIMA model.")
    ARIMA_AVAILABLE = False


# -----------------------------------------------
# Feature columns used for ML models
# -----------------------------------------------
ML_FEATURES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "quarter", "is_weekend", "is_month_start", "is_month_end",
    "day_of_year", "lag_1d", "lag_7d", "lag_14d", "lag_30d",
    "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
    "rolling_std_7d", "rolling_std_14d",
    "expanding_mean",
    "is_promotion", "promo_x_weekend", "promo_lag1",
    "unit_price", "discount_pct",
    "store_enc", "category_enc", "product_enc",
]


# -----------------------------------------------
# 1. Train-Test Split (Time-Based)
# -----------------------------------------------
def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Splits data by date (not random) to respect time-series nature.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    split_date = df.iloc[split_idx]["date"]

    train = df[df["date"] < split_date].copy()
    test  = df[df["date"] >= split_date].copy()

    logger.info(f"✅ Train: {len(train):,} rows | Test: {len(test):,} rows")
    logger.info(f"   Split date: {split_date.date()}")
    return train, test


# -----------------------------------------------
# 2. Compute Metrics
# -----------------------------------------------
def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Computes MAE, RMSE, MAPE, R² for model evaluation.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE: avoid divide by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}
    return metrics


# -----------------------------------------------
# 3. Naive Baseline (Rolling Mean)
# -----------------------------------------------
def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target: str = "actual_sales") -> dict:
    """
    Baseline model: predicts using 7-day rolling mean from training data.
    """
    last_7 = train[target].tail(7).mean()
    y_pred = np.full(len(test), last_7)
    y_true = test[target].values
    metrics = compute_metrics(y_true, y_pred, "Naive Baseline")
    logger.info(f"📊 Naive baseline MAE: {metrics['MAE']:.2f}")
    return metrics, y_pred


# -----------------------------------------------
# 4. Random Forest Model
# -----------------------------------------------
def train_random_forest(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    target: str = "actual_sales",
    n_estimators: int = 100,
    max_depth: int = 10,
    save_path: str = "models/rf_model.pkl"
) -> tuple:
    """
    Trains a Random Forest Regressor and evaluates on test set.
    Returns (metrics, y_pred, model, feature_importance_df)
    """
    ensure_dirs(["models"])

    # Filter available feature cols
    feature_cols = [c for c in ML_FEATURES if c in train.columns]

    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[target]

    logger.info(f"🌲 Training Random Forest with {n_estimators} trees...")
    rf = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = 42,
        n_jobs       = -1
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_pred  = np.maximum(y_pred, 0)   # No negative sales

    metrics = compute_metrics(y_test.values, y_pred, "Random Forest")
    print_metrics(metrics, "Random Forest Metrics")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":   feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save model
    joblib.dump(rf, save_path)
    logger.info(f"💾 Random Forest saved → {save_path}")

    return metrics, y_pred, rf, fi_df


# -----------------------------------------------
# 5. XGBoost Model
# -----------------------------------------------
def train_xgboost(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    target: str = "actual_sales",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    save_path: str = "models/xgb_model.pkl"
) -> tuple:
    """
    Trains XGBoost model and evaluates on test set.
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available. Returning None.")
        return None, None, None, None

    ensure_dirs(["models"])
    feature_cols = [c for c in ML_FEATURES if c in train.columns]

    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[target]

    logger.info("⚡ Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators  = n_estimators,
        learning_rate = learning_rate,
        max_depth     = max_depth,
        random_state  = 42,
        verbosity     = 0
    )
    xgb.fit(X_train, y_train)

    y_pred  = xgb.predict(X_test)
    y_pred  = np.maximum(y_pred, 0)

    metrics = compute_metrics(y_test.values, y_pred, "XGBoost")
    print_metrics(metrics, "XGBoost Metrics")

    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": xgb.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    joblib.dump(xgb, save_path)
    logger.info(f"💾 XGBoost saved → {save_path}")

    return metrics, y_pred, xgb, fi_df


# -----------------------------------------------
# 6. ARIMA Model (per product-store)
# -----------------------------------------------
def train_arima_for_product(
    df: pd.DataFrame,
    store: str,
    product: str,
    target: str = "actual_sales",
    order: tuple = (1, 1, 1),
    forecast_steps: int = 30
) -> tuple:
    """
    Trains an ARIMA model for a specific store-product pair.
    Returns (forecast_df, metrics)
    """
    if not ARIMA_AVAILABLE:
        return None, None

    series = (
        df[(df["store"] == store) & (df["product"] == product)]
        .set_index("date")[target]
        .asfreq("D")
        .fillna(0)
    )

    if len(series) < 60:
        return None, None

    # Train-test split: last 30 days as test
    train_s = series.iloc[:-forecast_steps]
    test_s  = series.iloc[-forecast_steps:]

    try:
        model  = ARIMA(train_s, order=order)
        result = model.fit()
        forecast = result.forecast(steps=forecast_steps)

        metrics = compute_metrics(
            test_s.values,
            np.maximum(forecast.values, 0),
            f"ARIMA {store}-{product}"
        )

        future_dates = pd.date_range(
            start = series.index[-1] + pd.Timedelta(days=1),
            periods = forecast_steps, freq="D"
        )
        forecast_df = pd.DataFrame({
            "date":      future_dates,
            "store":     store,
            "product":   product,
            "forecast":  np.maximum(forecast.values, 0),
            "model":     "ARIMA"
        })
        return forecast_df, metrics

    except Exception as e:
        logger.warning(f"⚠️  ARIMA failed for {store}-{product}: {e}")
        return None, None


# -----------------------------------------------
# 7. Prediction Intervals (Bootstrap)
# -----------------------------------------------
def compute_prediction_intervals(
    model,
    X: pd.DataFrame,
    n_bootstraps: int = 50,
    ci: float = 0.95
) -> tuple:
    """
    Estimates prediction intervals using bootstrap sampling
    (works for tree-based models).
    Returns (lower_bound, upper_bound) arrays.
    """
    preds = []
    n = len(X)
    feature_cols = [c for c in ML_FEATURES if c in X.columns]
    X_arr = X[feature_cols].fillna(0).values

    for _ in range(n_bootstraps):
        # Add small noise to simulate uncertainty
        noise = np.random.normal(0, 0.05, X_arr.shape)
        X_noisy = X_arr + noise
        p = model.predict(X_noisy)
        preds.append(np.maximum(p, 0))

    preds = np.array(preds)
    alpha = 1 - ci
    lower = np.percentile(preds, alpha / 2 * 100, axis=0)
    upper = np.percentile(preds, (1 - alpha / 2) * 100, axis=0)

    return lower, upper


# -----------------------------------------------
# 8. Future Forecast Generator
# -----------------------------------------------
def generate_future_forecast(
    df: pd.DataFrame,
    model,
    forecast_days: int = 30,
    target: str = "actual_sales"
) -> pd.DataFrame:
    """
    Generates forecast for the next N days using the last known feature row.
    Works by iteratively extending the dataset and predicting.
    """
    feature_cols = [c for c in ML_FEATURES if c in df.columns]
    forecasts    = []

    for (store, product), group in df.groupby(["store", "product"]):
        group = group.sort_values("date").tail(60).copy()
        last_date = group["date"].max()

        for i in range(1, forecast_days + 1):
            next_date = last_date + pd.Timedelta(days=i)

            # Build feature row from last known row
            row = group.tail(1).copy()
            row["date"]          = next_date
            row["day"]           = next_date.day
            row["month"]         = next_date.month
            row["year"]          = next_date.year
            row["day_of_week"]   = next_date.dayofweek
            row["week_of_year"]  = next_date.isocalendar().week
            row["quarter"]       = next_date.quarter
            row["is_weekend"]    = int(next_date.dayofweek >= 5)
            row["is_month_start"]= int(next_date.day == 1)
            row["is_month_end"]  = int(next_date.day == pd.Period(next_date, "M").days_in_month)
            row["day_of_year"]   = next_date.dayofyear

            # Rolling/lag from last known sales
            recent_sales = group[target].tail(30).values
            row["lag_1d"]           = recent_sales[-1] if len(recent_sales) >= 1 else 0
            row["lag_7d"]           = recent_sales[-7] if len(recent_sales) >= 7 else 0
            row["lag_14d"]          = recent_sales[-14] if len(recent_sales) >= 14 else 0
            row["lag_30d"]          = recent_sales[-30] if len(recent_sales) >= 30 else 0
            row["rolling_mean_7d"]  = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else np.mean(recent_sales)
            row["rolling_mean_14d"] = np.mean(recent_sales[-14:]) if len(recent_sales) >= 14 else np.mean(recent_sales)
            row["rolling_mean_30d"] = np.mean(recent_sales)
            row["rolling_std_7d"]   = np.std(recent_sales[-7:])  if len(recent_sales) >= 7 else 0
            row["rolling_std_14d"]  = np.std(recent_sales[-14:]) if len(recent_sales) >= 14 else 0
            row["expanding_mean"]   = np.mean(recent_sales)
            row["is_promotion"]     = 0
            row["promo_x_weekend"]  = 0
            row["promo_lag1"]       = 0

            X_row = row[feature_cols].fillna(0)
            pred  = max(model.predict(X_row)[0], 0)

            forecasts.append({
                "date":         next_date,
                "store":        store,
                "product":      product,
                "category":     group["category"].iloc[-1],
                "forecast_qty": round(pred, 2),
                "unit_price":   group["unit_price"].iloc[-1],
                "forecast_revenue": round(pred * group["unit_price"].iloc[-1], 2),
                "model":        "RandomForest"
            })

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


# -----------------------------------------------
# MASTER FORECASTING PIPELINE
# -----------------------------------------------
def run_forecasting_pipeline(
    features_path:  str = "data/features_data.csv",
    forecast_path:  str = "outputs/forecast_results.csv",
    config_path:    str = "config/config.yaml"
) -> tuple:
    """
    Full forecasting pipeline:
    1. Load feature data
    2. Time-based train/test split
    3. Train & compare: Naive, RF, XGBoost
    4. Select best model
    5. Generate future forecast
    6. Save results
    Returns (forecast_df, comparison_df, best_model, test_df_with_preds)
    """
    ensure_dirs(["models", "outputs"])

    logger.info("="*50)
    logger.info("  FORECASTING PIPELINE STARTED")
    logger.info("="*50)

    df = load_csv(features_path, parse_dates=["date"])

    # Focus on a single store-product for model comparison demo
    # then generate full forecast
    train, test = time_based_split(df, train_ratio=0.8)

    target = "actual_sales"

    # --- Naive Baseline ---
    logger.info("\n--- Naive Baseline ---")
    naive_metrics, naive_pred = naive_forecast(train, test, target)

    # --- Random Forest ---
    logger.info("\n--- Random Forest ---")
    rf_metrics, rf_pred, rf_model, rf_fi = train_random_forest(train, test, target)

    # --- XGBoost ---
    logger.info("\n--- XGBoost ---")
    xgb_metrics, xgb_pred, xgb_model, xgb_fi = train_xgboost(train, test, target)

    # --- Model Comparison Table ---
    comparison = [naive_metrics, rf_metrics]
    if xgb_metrics:
        comparison.append(xgb_metrics)

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.round(4)

    print("\n" + "="*60)
    print("  MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)

    save_csv(comparison_df, "outputs/model_comparison.csv", "Model Comparison")

    # --- Best Model (lowest MAE) ---
    if xgb_metrics and xgb_metrics["MAE"] < rf_metrics["MAE"]:
        best_model  = xgb_model
        best_pred   = xgb_pred
        best_name   = "XGBoost"
        best_fi     = xgb_fi
    else:
        best_model  = rf_model
        best_pred   = rf_pred
        best_name   = "RandomForest"
        best_fi     = rf_fi

    logger.info(f"\n🏆 Best model: {best_name}")

    # Add predictions to test set
    test_with_preds = test.copy()
    test_with_preds["predicted_sales"] = best_pred
    test_with_preds["residual"]        = test_with_preds[target] - best_pred

    # Prediction intervals on test set
    feature_cols = [c for c in ML_FEATURES if c in test.columns]
    lower, upper = compute_prediction_intervals(best_model, test[feature_cols])
    test_with_preds["pred_lower"] = lower
    test_with_preds["pred_upper"] = upper

    save_csv(test_with_preds[["date", "store", "product", "category",
                               target, "predicted_sales", "pred_lower",
                               "pred_upper", "residual"]],
             "outputs/test_predictions.csv", "Test Predictions")

    # --- Future Forecast ---
    logger.info(f"\n--- Generating 30-day future forecast ---")
    forecast_df = generate_future_forecast(df, best_model, forecast_days=30)
    save_csv(forecast_df, forecast_path, "Future Forecast")

    # Save feature importance
    if best_fi is not None:
        save_csv(best_fi, "outputs/feature_importance.csv", "Feature Importance")

    logger.info("✅ Forecasting pipeline complete.")
    return forecast_df, comparison_df, best_model, test_with_preds


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    forecast_df, comp_df, best_model, test_preds = run_forecasting_pipeline()
    print(f"\nForecast shape: {forecast_df.shape}")
    print(forecast_df.head(5))
    print(f"\nSample predictions:")
    print(test_preds[["date", "store", "product", "actual_sales",
                       "predicted_sales", "pred_lower", "pred_upper"]].head(5))
