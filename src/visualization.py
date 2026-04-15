# ============================================================
# src/visualization.py
# All project visualizations – EDA, Forecast, Inventory, ABC
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_csv, ensure_dirs

logger = get_logger(__name__)

# Global style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "Set2"
FIG_DPI = 120


def save_fig(fig, path: str, label: str = ""):
    ensure_dirs([os.path.dirname(path)])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📸 Saved {label} → {path}")


# -----------------------------------------------
# 1. Dataset Preview
# -----------------------------------------------
def plot_dataset_preview(df: pd.DataFrame, save_path: str = "images/01_dataset_preview.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # Missing values heatmap
    ax = axes[0]
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing = missing[missing["Missing Count"] > 0]
    if len(missing) > 0:
        sns.barplot(data=missing, x="Missing Count", y="Column", palette="Reds_r", ax=ax)
        ax.set_title("Missing Values")
    else:
        ax.text(0.5, 0.5, "✅ No Missing Values", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Missing Values")
        ax.axis("off")

    # Data types distribution
    ax = axes[1]
    dtype_counts = df.dtypes.astype(str).value_counts()
    ax.pie(dtype_counts.values, labels=dtype_counts.index,
           autopct="%1.0f%%", colors=sns.color_palette(PALETTE))
    ax.set_title("Column Data Types")

    save_fig(fig, save_path, "Dataset Preview")


# -----------------------------------------------
# 2. Sales Trend Over Time
# -----------------------------------------------
def plot_sales_trend(df: pd.DataFrame, save_path: str = "images/02_sales_trend.png"):
    daily = df.groupby("date")["actual_sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily["date"], daily["actual_sales"], color="#2196F3", linewidth=0.8, alpha=0.7)

    # 30-day rolling average
    rolling = daily["actual_sales"].rolling(30).mean()
    ax.plot(daily["date"], rolling, color="#FF5722", linewidth=2, label="30-day Rolling Avg")

    ax.set_title("📈 Total Daily Sales Trend (All Stores)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_fig(fig, save_path, "Sales Trend")


# -----------------------------------------------
# 3. Category-wise Sales
# -----------------------------------------------
def plot_category_sales(df: pd.DataFrame, save_path: str = "images/03_category_sales.png"):
    cat_monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
          .groupby(["month", "category"])["actual_sales"]
          .sum()
          .reset_index()
    )
    cat_total = df.groupby("category").agg(
        total_sales   = ("actual_sales", "sum"),
        total_revenue = ("revenue", "sum")
    ).reset_index().sort_values("total_revenue", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Category-wise Performance", fontsize=16, fontweight="bold")

    # Horizontal bar - revenue by category
    ax = axes[0]
    colors = sns.color_palette(PALETTE, len(cat_total))
    ax.barh(cat_total["category"], cat_total["total_revenue"] / 1e6,
            color=colors, edgecolor="white")
    ax.set_title("Total Revenue by Category (₹M)")
    ax.set_xlabel("Revenue (₹ Millions)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))

    # Pie chart - sales share
    ax = axes[1]
    cat_share = df.groupby("category")["actual_sales"].sum()
    ax.pie(cat_share.values, labels=cat_share.index, autopct="%1.1f%%",
           colors=sns.color_palette(PALETTE, len(cat_share)), startangle=90)
    ax.set_title("Sales Volume Share by Category")

    save_fig(fig, save_path, "Category Sales")


# -----------------------------------------------
# 4. Store Performance Comparison
# -----------------------------------------------
def plot_store_performance(df: pd.DataFrame, save_path: str = "images/04_store_performance.png"):
    store_stats = df.groupby("store").agg(
        total_revenue  = ("revenue", "sum"),
        total_units    = ("actual_sales", "sum"),
        stockout_count = ("stockout_flag", "sum")
    ).reset_index().sort_values("total_revenue", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Store Performance Comparison", fontsize=16, fontweight="bold")

    colors = sns.color_palette(PALETTE, len(store_stats))

    # Revenue
    axes[0].bar(store_stats["store"], store_stats["total_revenue"] / 1e6, color=colors)
    axes[0].set_title("Total Revenue (₹M)")
    axes[0].set_xticklabels(store_stats["store"], rotation=20)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))

    # Units sold
    axes[1].bar(store_stats["store"], store_stats["total_units"], color=colors)
    axes[1].set_title("Total Units Sold")
    axes[1].set_xticklabels(store_stats["store"], rotation=20)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Stockout count
    axes[2].bar(store_stats["store"], store_stats["stockout_count"], color="tomato")
    axes[2].set_title("Stockout Events")
    axes[2].set_xticklabels(store_stats["store"], rotation=20)

    save_fig(fig, save_path, "Store Performance")


# -----------------------------------------------
# 5. Seasonality Heatmap
# -----------------------------------------------
def plot_seasonality_heatmap(df: pd.DataFrame, save_path: str = "images/05_seasonality_heatmap.png"):
    df2 = df.copy()
    df2["month"]       = df2["date"].dt.month
    df2["day_of_week"] = df2["date"].dt.dayofweek

    pivot = df2.pivot_table(values="actual_sales", index="day_of_week",
                             columns="month", aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Avg Units Sold"})
    ax.set_title("🗓️ Seasonality Heatmap: Avg Sales by Day-of-Week × Month",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    save_fig(fig, save_path, "Seasonality Heatmap")


# -----------------------------------------------
# 6. Promotion Impact Analysis
# -----------------------------------------------
def plot_promotion_impact(df: pd.DataFrame, save_path: str = "images/06_promotion_impact.png"):
    promo_stats = df.groupby("is_promotion")["actual_sales"].agg(["mean", "median", "sum"]).reset_index()
    promo_stats["is_promotion"] = promo_stats["is_promotion"].map({0: "No Promotion", 1: "Promotion"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("🎁 Promotion Impact on Sales", fontsize=15, fontweight="bold")

    # Mean comparison
    colors = ["#90CAF9", "#FF8A65"]
    axes[0].bar(promo_stats["is_promotion"], promo_stats["mean"], color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Average Daily Sales")
    axes[0].set_ylabel("Units")
    for i, v in enumerate(promo_stats["mean"]):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

    # Distribution comparison

    # 🔥 Force string type (FINAL FIX)
    df["is_promotion"] = df["is_promotion"].astype(str)

    palette = {"0": "#90CAF9", "1": "#FF8A65"}

    sns.boxplot(
    data=df,
    x="is_promotion",
    y="actual_sales",
    palette=palette,
    ax=axes[1]
    )
    axes[1].set_title("Sales Distribution: Promo vs Non-Promo")
    axes[1].set_xticklabels(["No Promotion", "Promotion"])
    axes[1].set_ylabel("Units Sold")

    save_fig(fig, save_path, "Promotion Impact")


# -----------------------------------------------
# 7. Forecast vs Actual (Test Set)
# -----------------------------------------------
def plot_forecast_vs_actual(
    test_preds: pd.DataFrame,
    store: str = None,
    product: str = None,
    save_path: str = "images/07_forecast_vs_actual.png"
):
    df2 = test_preds.copy()
    if store and product:
        df2 = df2[(df2["store"] == store) & (df2["product"] == product)]

    # Aggregate all if no filter
    df2 = df2.groupby("date").agg(
        actual_sales    = ("actual_sales", "sum"),
        predicted_sales = ("predicted_sales", "sum"),
        pred_lower      = ("pred_lower", "sum"),
        pred_upper      = ("pred_upper", "sum")
    ).reset_index().sort_values("date")

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(df2["date"], df2["actual_sales"],    label="Actual",    color="#2196F3", linewidth=2)
    ax.plot(df2["date"], df2["predicted_sales"], label="Predicted", color="#FF5722",
            linewidth=2, linestyle="--")
    ax.fill_between(df2["date"], df2["pred_lower"], df2["pred_upper"],
                    alpha=0.15, color="#FF5722", label="95% Confidence Band")

    ax.set_title("🎯 Forecast vs Actual Sales (Test Period)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save_fig(fig, save_path, "Forecast vs Actual")


# -----------------------------------------------
# 8. Future Forecast Chart
# -----------------------------------------------
def plot_future_forecast(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    store: str,
    product: str,
    save_path: str = "images/08_future_forecast.png"
):
    hist = (
        df[(df["store"] == store) & (df["product"] == product)]
          .groupby("date")["actual_sales"].sum().reset_index()
          .tail(90)
    )
    fut = (
        forecast_df[(forecast_df["store"] == store) & (forecast_df["product"] == product)]
                   .sort_values("date")
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(hist["date"], hist["actual_sales"], label="Historical", color="#2196F3", linewidth=2)
    ax.plot(fut["date"], fut["forecast_qty"],   label="Forecast",   color="#FF5722",
            linewidth=2, linestyle="--")
    ax.axvline(hist["date"].max(), color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")

    ax.set_title(f"📅 30-Day Sales Forecast — {product} @ {store}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()

    save_fig(fig, save_path, "Future Forecast")


# -----------------------------------------------
# 9. ABC Analysis Chart
# -----------------------------------------------
def plot_abc_analysis(abc_df: pd.DataFrame, save_path: str = "images/09_abc_analysis.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("📦 ABC Analysis — Product Revenue Classification",
                 fontsize=15, fontweight="bold")

    # Pareto chart
    ax = axes[0]
    ax2 = ax.twinx()
    colors_abc = abc_df["abc_class"].map({"A": "#4CAF50", "B": "#FF9800", "C": "#F44336"})
    ax.bar(abc_df["product"], abc_df["revenue_pct"], color=colors_abc, edgecolor="white", alpha=0.8)
    ax2.plot(abc_df["product"], abc_df["cumulative_pct"], color="#212121",
             linewidth=2, marker="o", markersize=4, label="Cumulative %")
    ax2.axhline(70,  color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7, label="A (70%)")
    ax2.axhline(90,  color="#FF9800", linestyle="--", linewidth=1, alpha=0.7, label="B (90%)")
    ax.set_xlabel("Product")
    ax.set_ylabel("Revenue %", color="gray")
    ax2.set_ylabel("Cumulative Revenue %")
    ax.set_xticklabels(abc_df["product"], rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="center right")
    ax.set_title("Pareto Chart (Revenue %)")

    # Pie: ABC class share
    ax = axes[1]
    abc_group = abc_df.groupby("abc_class")["total_revenue"].sum()
    abc_colors = {"A": "#4CAF50", "B": "#FF9800", "C": "#F44336"}
    ax.pie(abc_group.values,
           labels=[f"Class {c}" for c in abc_group.index],
           autopct="%1.1f%%",
           colors=[abc_colors.get(c, "gray") for c in abc_group.index],
           startangle=90)
    ax.set_title("Revenue Share by ABC Class")

    save_fig(fig, save_path, "ABC Analysis")


# -----------------------------------------------
# 10. Inventory Status Dashboard
# -----------------------------------------------
def plot_inventory_status(inv_df: pd.DataFrame, save_path: str = "images/10_inventory_status.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("📦 Inventory Optimization Dashboard", fontsize=16, fontweight="bold")

    # Risk level distribution
    ax = axes[0, 0]
    risk_counts = inv_df["risk_level"].value_counts()
    colors = {"🔴 HIGH - REORDER NOW": "#F44336",
              "🟡 MEDIUM - Monitor Closely": "#FF9800",
              "🟠 OVERSTOCK - Reduce Orders": "#FF5722",
              "🟢 HEALTHY": "#4CAF50"}
    bar_colors = [colors.get(r, "gray") for r in risk_counts.index]
    ax.barh(risk_counts.index, risk_counts.values, color=bar_colors)
    ax.set_title("Inventory Risk Distribution")
    ax.set_xlabel("Number of Store-Product Pairs")

    # Safety Stock vs Current Stock
    ax = axes[0, 1]
    sample = inv_df.groupby("product")[["current_stock", "safety_stock", "reorder_point"]].mean().reset_index()
    x = range(len(sample))
    width = 0.25
    ax.bar([i - width for i in x], sample["current_stock"], width, label="Current Stock", color="#2196F3")
    ax.bar([i for i in x], sample["safety_stock"], width, label="Safety Stock", color="#FF9800")
    ax.bar([i + width for i in x], sample["reorder_point"], width, label="Reorder Point", color="#F44336")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sample["product"], rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.set_title("Stock Levels vs Thresholds (by Product)")

    # Stockout rate by category
    ax = axes[1, 0]
    cat_stockout = inv_df.groupby("category")["stockout_rate_%"].mean().sort_values(ascending=False)
    sns.barplot(x=cat_stockout.values, y=cat_stockout.index, palette="Reds_r", ax=ax)
    ax.set_title("Avg Stockout Rate by Category (%)")
    ax.set_xlabel("Stockout Rate (%)")

    # EOQ vs Order Recommendation
    ax = axes[1, 1]
    need_reorder = inv_df[inv_df["stockout_risk"] == True].head(15)
    if len(need_reorder) > 0:
        ax.barh(need_reorder["product"] + " | " + need_reorder["store"].str[-3:],
                need_reorder["eoq"], color="#4CAF50", edgecolor="white")
        ax.set_title("🛒 Recommended Order Quantity (EOQ) — Reorder Alerts")
        ax.set_xlabel("Units to Order (EOQ)")
    else:
        ax.text(0.5, 0.5, "✅ No immediate reorder needed", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, save_path, "Inventory Dashboard")


# -----------------------------------------------
# 11. Feature Importance Plot
# -----------------------------------------------
def plot_feature_importance(fi_df: pd.DataFrame, save_path: str = "images/11_feature_importance.png"):
    top = fi_df.head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("Blues_r", len(top))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_title("🔍 Top 15 Feature Importances (Best Model)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    save_fig(fig, save_path, "Feature Importance")


# -----------------------------------------------
# 12. Model Comparison Bar
# -----------------------------------------------
def plot_model_comparison(comp_df: pd.DataFrame, save_path: str = "images/12_model_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("🏆 Model Comparison", fontsize=15, fontweight="bold")

    metrics = [("MAE", "MAE (Lower = Better)", "#F44336"),
               ("RMSE", "RMSE (Lower = Better)", "#FF9800"),
               ("R2", "R² Score (Higher = Better)", "#4CAF50")]

    for ax, (metric, title, color) in zip(axes, metrics):
        if metric not in comp_df.columns:
            continue
        bars = ax.bar(comp_df["Model"], comp_df[metric], color=color, edgecolor="white", width=0.5)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xticklabels(comp_df["Model"], rotation=15)
        for bar, val in zip(bars, comp_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontweight="bold", fontsize=9)

    save_fig(fig, save_path, "Model Comparison")


# -----------------------------------------------
# MASTER VISUALIZATION PIPELINE
# -----------------------------------------------
def run_visualization_pipeline(
    processed_path: str = "data/processed_sales_data.csv",
    forecast_path:  str = "outputs/forecast_results.csv",
    test_preds_path:str = "outputs/test_predictions.csv",
    inventory_path: str = "outputs/inventory_recommendations.csv",
    abc_path:       str = "outputs/abc_analysis.csv",
    fi_path:        str = "outputs/feature_importance.csv",
    comp_path:      str = "outputs/model_comparison.csv",
    images_dir:     str = "images/"
):
    ensure_dirs([images_dir])

    logger.info("="*50)
    logger.info("  VISUALIZATION PIPELINE STARTED")
    logger.info("="*50)

    df          = load_csv(processed_path, parse_dates=["date"])
    forecast_df = load_csv(forecast_path,  parse_dates=["date"])
    test_preds  = load_csv(test_preds_path, parse_dates=["date"])
    inv_df      = load_csv(inventory_path)
    abc_df      = load_csv(abc_path)

    logger.info("📸 1. Dataset preview")
    plot_dataset_preview(df)

    logger.info("📸 2. Sales trend")
    plot_sales_trend(df)

    logger.info("📸 3. Category sales")
    plot_category_sales(df)

    logger.info("📸 4. Store performance")
    plot_store_performance(df)

    logger.info("📸 5. Seasonality heatmap")
    plot_seasonality_heatmap(df)

    logger.info("📸 6. Promotion impact")
    plot_promotion_impact(df)

    logger.info("📸 7. Forecast vs actual")
    plot_forecast_vs_actual(test_preds)

    logger.info("📸 8. Future forecast (sample)")
    sample_store   = df["store"].unique()[0]
    sample_product = df["product"].unique()[0]
    plot_future_forecast(df, forecast_df, sample_store, sample_product)

    logger.info("📸 9. ABC analysis")
    plot_abc_analysis(abc_df)

    logger.info("📸 10. Inventory status")
    plot_inventory_status(inv_df)

    try:
        fi_df = load_csv(fi_path)
        logger.info("📸 11. Feature importance")
        plot_feature_importance(fi_df)
    except FileNotFoundError:
        logger.warning("Feature importance file not found, skipping.")

    try:
        comp_df = load_csv(comp_path)
        logger.info("📸 12. Model comparison")
        plot_model_comparison(comp_df)
    except FileNotFoundError:
        logger.warning("Model comparison file not found, skipping.")

    logger.info(f"✅ All visualizations saved to '{images_dir}' folder.")
    print(f"\n✅ All charts saved to '{images_dir}'")


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    run_visualization_pipeline()
