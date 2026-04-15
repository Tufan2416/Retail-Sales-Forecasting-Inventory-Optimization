# ============================================================
# src/inventory_optimization.py
# Safety Stock, EOQ, ROP, ABC Analysis, Reorder Recommendations
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, save_csv, load_csv, ensure_dirs

logger = get_logger(__name__)


# -----------------------------------------------
# 1. Safety Stock Calculation
# -----------------------------------------------
def calculate_safety_stock(
    demand_std:      float,
    lead_time_days:  int   = 7,
    z_score:         float = 1.65
) -> float:
    """
    Safety Stock = Z × σ_demand × √Lead_Time

    Z = 1.65 for 95% service level
    σ_demand = standard deviation of daily demand
    Lead_Time = supplier lead time in days

    Safety stock protects against demand variability during lead time.
    """
    safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
    return round(safety_stock, 2)


# -----------------------------------------------
# 2. Reorder Point (ROP) Calculation
# -----------------------------------------------
def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time_days:   int   = 7,
    safety_stock:     float = 0
) -> float:
    """
    ROP = (Avg Daily Demand × Lead Time) + Safety Stock

    When stock hits this level → trigger a reorder.
    """
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    return round(rop, 2)


# -----------------------------------------------
# 3. Economic Order Quantity (EOQ)
# -----------------------------------------------
def calculate_eoq(
    annual_demand:  float,
    ordering_cost:  float = 50.0,
    unit_cost:      float = 100.0,
    holding_rate:   float = 0.25
) -> float:
    """
    EOQ = √ (2 × D × S / H)

    D = Annual demand (units)
    S = Cost per order (fixed ordering cost)
    H = Holding cost per unit per year = unit_cost × holding_rate

    EOQ minimizes total inventory cost (ordering + holding).
    """
    H = unit_cost * holding_rate
    if H <= 0 or annual_demand <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / H)
    return round(eoq, 2)


# -----------------------------------------------
# 4. ABC Analysis
# -----------------------------------------------
def abc_analysis(
    df: pd.DataFrame,
    a_threshold: float = 0.70,
    b_threshold: float = 0.90
) -> pd.DataFrame:
    """
    ABC Analysis based on revenue contribution:
    - A products: top 70% of revenue (high priority)
    - B products: next 20% of revenue (medium priority)
    - C products: bottom 10% of revenue (low priority)

    Also adds:
    - % of total revenue
    - Cumulative revenue %
    """
    # Aggregate by product
    product_rev = (
        df.groupby(["product", "category"])
          .agg(
              total_revenue  = ("revenue", "sum"),
              total_units    = ("actual_sales", "sum"),
              avg_unit_price = ("unit_price", "mean")
          )
          .reset_index()
          .sort_values("total_revenue", ascending=False)
          .reset_index(drop=True)
    )

    total_rev = product_rev["total_revenue"].sum()
    product_rev["revenue_pct"] = (product_rev["total_revenue"] / total_rev * 100).round(2)
    product_rev["cumulative_pct"] = product_rev["revenue_pct"].cumsum().round(2)

    # Assign ABC class
    def assign_class(cum_pct):
        if cum_pct <= a_threshold * 100:
            return "A"
        elif cum_pct <= b_threshold * 100:
            return "B"
        else:
            return "C"

    product_rev["abc_class"] = product_rev["cumulative_pct"].apply(assign_class)

    # Summary
    summary = product_rev.groupby("abc_class").agg(
        num_products  = ("product", "count"),
        total_revenue = ("total_revenue", "sum"),
        revenue_pct   = ("revenue_pct", "sum")
    ).reset_index()

    logger.info("\n📊 ABC Analysis Summary:")
    logger.info(summary.to_string(index=False))

    return product_rev


# -----------------------------------------------
# 5. Inventory Status Assessment
# -----------------------------------------------
def assess_inventory_status(
    current_stock:  float,
    rop:            float,
    eoq:            float,
    safety_stock:   float,
    avg_daily_demand: float,
    unit_price:     float
) -> dict:
    """
    Assesses current inventory status and returns flags + recommendations.
    """
    days_of_stock = current_stock / avg_daily_demand if avg_daily_demand > 0 else 999

    # Flags
    stockout_risk = current_stock <= rop
    overstock     = current_stock > rop + eoq * 1.5
    low_stock     = current_stock <= safety_stock

    # Risk level
    if low_stock or stockout_risk:
        risk_level = "🔴 HIGH - REORDER NOW"
    elif current_stock <= rop * 1.2:
        risk_level = "🟡 MEDIUM - Monitor Closely"
    elif overstock:
        risk_level = "🟠 OVERSTOCK - Reduce Orders"
    else:
        risk_level = "🟢 HEALTHY"

    # Holding cost estimate (annual, prorated)
    holding_cost = current_stock * unit_price * 0.25 / 365

    return {
        "days_of_stock":     round(days_of_stock, 1),
        "stockout_risk":     stockout_risk,
        "overstock_flag":    overstock,
        "low_stock_flag":    low_stock,
        "risk_level":        risk_level,
        "holding_cost_daily": round(holding_cost, 2),
        "recommended_order": round(eoq, 0) if stockout_risk else 0
    }


# -----------------------------------------------
# 6. Build Full Inventory Recommendation Table
# -----------------------------------------------
def build_inventory_recommendations(
    df:             pd.DataFrame,
    forecast_df:    pd.DataFrame,
    lead_time:      int   = 7,
    z_score:        float = 1.65,
    ordering_cost:  float = 50.0,
    holding_rate:   float = 0.25
) -> pd.DataFrame:
    """
    Generates a comprehensive inventory recommendation table for
    each store-product pair.

    Includes: Safety Stock, ROP, EOQ, Reorder Qty, Risk Level, ABC Class
    """
    # Step 1: Get product stats from historical data
    product_stats = (
        df.groupby(["store", "product", "category"])
          .agg(
              avg_daily_demand = ("actual_sales", "mean"),
              std_daily_demand = ("actual_sales", "std"),
              total_units_sold = ("actual_sales", "sum"),
              total_revenue    = ("revenue", "sum"),
              unit_price       = ("unit_price", "first"),
              current_stock    = ("current_stock", "last"),
              stockout_count   = ("stockout_flag", "sum"),
              total_days       = ("date", "count")
          )
          .reset_index()
    )

    product_stats["std_daily_demand"] = product_stats["std_daily_demand"].fillna(0)
    product_stats["stockout_rate_%"]  = (
        product_stats["stockout_count"] / product_stats["total_days"] * 100
    ).round(2)

    # Step 2: Forecasted demand (next 30 days)
    forecast_avg = (
        forecast_df.groupby(["store", "product"])["forecast_qty"]
                   .mean()
                   .reset_index()
                   .rename(columns={"forecast_qty": "forecast_avg_daily"})
    )
    product_stats = product_stats.merge(forecast_avg, on=["store", "product"], how="left")
    product_stats["forecast_avg_daily"] = product_stats["forecast_avg_daily"].fillna(
        product_stats["avg_daily_demand"]
    )

    # Step 3: Calculate inventory metrics
    results = []
    for _, row in product_stats.iterrows():
        avg_demand  = row["avg_daily_demand"]
        std_demand  = row["std_daily_demand"]
        unit_price  = row["unit_price"]
        cur_stock   = row["current_stock"]
        annual_demand = avg_demand * 365

        ss  = calculate_safety_stock(std_demand, lead_time, z_score)
        rop = calculate_reorder_point(avg_demand, lead_time, ss)
        eoq = calculate_eoq(annual_demand, ordering_cost, unit_price, holding_rate)

        status = assess_inventory_status(
            cur_stock, rop, eoq, ss, avg_demand, unit_price
        )

        results.append({
            **row.to_dict(),
            "safety_stock":       ss,
            "reorder_point":      rop,
            "eoq":                eoq,
            **status
        })

    inv_df = pd.DataFrame(results)

    # Step 4: ABC Analysis
    abc_df = abc_analysis(df)
    abc_map = abc_df.set_index("product")["abc_class"].to_dict()
    inv_df["abc_class"] = inv_df["product"].map(abc_map).fillna("C")

    # Step 5: Priority scoring (A + High Risk = highest priority)
    priority_map = {"A": 3, "B": 2, "C": 1}
    risk_map     = {"🔴 HIGH - REORDER NOW": 3, "🟡 MEDIUM - Monitor Closely": 2,
                    "🟠 OVERSTOCK - Reduce Orders": 1, "🟢 HEALTHY": 0}

    inv_df["abc_score"]      = inv_df["abc_class"].map(priority_map).fillna(1)
    inv_df["risk_score"]     = inv_df["risk_level"].map(risk_map).fillna(0)
    inv_df["priority_score"] = inv_df["abc_score"] + inv_df["risk_score"]

    inv_df = inv_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    # Round numeric cols
    num_cols = ["avg_daily_demand", "std_daily_demand", "safety_stock",
                "reorder_point", "eoq", "current_stock"]
    inv_df[num_cols] = inv_df[num_cols].round(2)

    logger.info(f"✅ Inventory recommendations generated: {len(inv_df)} store-product pairs")
    return inv_df


# -----------------------------------------------
# 7. Print Reorder Alerts
# -----------------------------------------------
def print_reorder_alerts(inv_df: pd.DataFrame):
    """
    Prints a clean reorder alert table for all HIGH risk items.
    """
    alerts = inv_df[inv_df["stockout_risk"] == True][
        ["store", "product", "abc_class", "current_stock",
         "reorder_point", "eoq", "risk_level"]
    ].head(20)

    print("\n" + "="*80)
    print("  🚨 REORDER ALERTS - ITEMS BELOW REORDER POINT")
    print("="*80)
    if len(alerts) == 0:
        print("  ✅ No immediate reorder alerts.")
    else:
        print(alerts.to_string(index=False))
    print("="*80)


# -----------------------------------------------
# 8. Business KPI Summary
# -----------------------------------------------
def compute_business_kpis(df: pd.DataFrame, inv_df: pd.DataFrame) -> dict:
    """
    Computes before/after KPIs to show business impact.
    """
    # BEFORE optimization (actual state)
    stockout_rate_before = df["stockout_flag"].mean() * 100
    total_lost_sales     = df["lost_sales"].sum() if "lost_sales" in df.columns else 0
    lost_revenue         = (df["lost_sales"] * df["unit_price"]).sum() if "lost_sales" in df.columns else 0

    # AFTER optimization (projected)
    # Assume 60% reduction in stockouts with optimized inventory
    stockout_rate_after  = stockout_rate_before * 0.40
    items_optimized      = len(inv_df)
    high_risk_items      = inv_df["stockout_risk"].sum()
    overstock_items      = inv_df["overstock_flag"].sum()

    # Inventory turnover
    avg_inventory_value  = (inv_df["current_stock"] * inv_df["unit_price"]).mean()
    total_revenue        = df["revenue"].sum()
    inv_turnover         = total_revenue / avg_inventory_value if avg_inventory_value > 0 else 0

    kpis = {
        "Total Store-Product Pairs": items_optimized,
        "Stockout Rate (Before %)":  round(stockout_rate_before, 2),
        "Stockout Rate (After %)":   round(stockout_rate_after, 2),
        "High Risk Items":           int(high_risk_items),
        "Overstock Items":           int(overstock_items),
        "Total Lost Units":          int(total_lost_sales),
        "Estimated Lost Revenue ₹":  round(lost_revenue, 0),
        "Projected Revenue Recovery ₹": round(lost_revenue * 0.60, 0),
        "Inventory Turnover Ratio":  round(inv_turnover, 2),
        "Total Revenue ₹":           round(total_revenue, 0),
    }

    print("\n" + "="*55)
    print("  📊 BUSINESS KPI DASHBOARD")
    print("="*55)
    for k, v in kpis.items():
        print(f"  {k:<35}: {v:>15,}")
    print("="*55)

    return kpis


# -----------------------------------------------
# MASTER INVENTORY PIPELINE
# -----------------------------------------------
def run_inventory_pipeline(
    processed_path:   str = "data/processed_sales_data.csv",
    forecast_path:    str = "outputs/forecast_results.csv",
    inventory_output: str = "outputs/inventory_recommendations.csv",
    abc_output:       str = "outputs/abc_analysis.csv"
) -> tuple:
    """
    Full inventory optimization pipeline.
    Returns (inv_df, abc_df, kpis)
    """
    ensure_dirs(["outputs"])

    logger.info("="*50)
    logger.info("  INVENTORY OPTIMIZATION PIPELINE STARTED")
    logger.info("="*50)

    df          = load_csv(processed_path, parse_dates=["date"])
    forecast_df = load_csv(forecast_path,  parse_dates=["date"])

    logger.info("\n--- Building Inventory Recommendations ---")
    inv_df = build_inventory_recommendations(df, forecast_df)

    logger.info("\n--- ABC Analysis ---")
    abc_df = abc_analysis(df)

    logger.info("\n--- Reorder Alerts ---")
    print_reorder_alerts(inv_df)

    logger.info("\n--- Business KPIs ---")
    kpis = compute_business_kpis(df, inv_df)

    save_csv(inv_df, inventory_output, "Inventory Recommendations")
    save_csv(abc_df, abc_output,       "ABC Analysis")

    logger.info("✅ Inventory optimization pipeline complete.")
    return inv_df, abc_df, kpis


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    inv_df, abc_df, kpis = run_inventory_pipeline()
    print(f"\nInventory Table (first 5):")
    print(inv_df[["store","product","abc_class","current_stock",
                  "safety_stock","reorder_point","eoq","risk_level"]].head(5))
    print(f"\nABC Summary:")
    print(abc_df.groupby("abc_class")[["total_revenue","revenue_pct"]].sum())
