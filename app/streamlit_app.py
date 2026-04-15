# ============================================================
# app/streamlit_app.py
# Interactive Retail Analytics Dashboard (Streamlit)
# Run: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Retail Sales & Inventory Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        padding: 10px 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #1565C0;
    }
    .alert-red  { background: #ffebee; border-left: 5px solid #F44336; padding: 10px; border-radius: 5px; }
    .alert-green{ background: #e8f5e9; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px; }
    .alert-orange{ background: #fff3e0; border-left: 5px solid #FF9800; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------
# Load Data (cached)
# -----------------------------------------------
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def try_load(path, parse_dates=None):
        full = os.path.join(base, path)
        if os.path.exists(full):
            if parse_dates:
                return pd.read_csv(full, parse_dates=parse_dates)
            return pd.read_csv(full)
        return None

    df          = try_load("data/processed_sales_data.csv",  parse_dates=["date"])
    forecast_df = try_load("outputs/forecast_results.csv",   parse_dates=["date"])
    test_preds  = try_load("outputs/test_predictions.csv",   parse_dates=["date"])
    inv_df      = try_load("outputs/inventory_recommendations.csv")
    abc_df      = try_load("outputs/abc_analysis.csv")
    comp_df     = try_load("outputs/model_comparison.csv")
    fi_df       = try_load("outputs/feature_importance.csv")

    return df, forecast_df, test_preds, inv_df, abc_df, comp_df, fi_df


# -----------------------------------------------
# Sidebar Navigation
# -----------------------------------------------
def sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
    st.sidebar.title("🛒 Retail Analytics")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📈 Sales Analysis", "🎯 Forecasting",
         "📦 Inventory Optimization", "🏆 Model Performance", "📊 ABC Analysis"]
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project:** Retail Sales Forecasting & Inventory Optimization")
    st.sidebar.markdown("**Tech Stack:** Python, ML, Streamlit")
    return page


# -----------------------------------------------
# PAGE 1: Overview
# -----------------------------------------------
def page_overview(df, inv_df, forecast_df):
    st.markdown('<p class="main-header">🛒 Retail Sales Forecasting & Inventory Optimization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Industry-Level Retail Analytics Dashboard | Powered by Machine Learning</p>', unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Data not found. Please run main.py first.")
        st.code("python main.py", language="bash")
        return

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_revenue   = df["revenue"].sum()
    total_units     = df["actual_sales"].sum()
    stockout_rate   = df["stockout_flag"].mean() * 100
    total_products  = df["product"].nunique()
    total_stores    = df["store"].nunique()

    col1.metric("💰 Total Revenue",    f"₹{total_revenue/1e6:.1f}M")
    col2.metric("📦 Units Sold",       f"{total_units:,.0f}")
    col3.metric("🚨 Stockout Rate",    f"{stockout_rate:.1f}%")
    col4.metric("🏪 Stores",          f"{total_stores}")
    col5.metric("🛍️ Products",        f"{total_products}")

    st.markdown("---")

    # Revenue trend
    col1, col2 = st.columns([2, 1])

    with col1:
        daily = df.groupby("date")["revenue"].sum().reset_index()
        daily["rolling_7d"] = daily["revenue"].rolling(7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"],
                                 name="Daily Revenue", line=dict(color="#E3F2FD", width=1),
                                 fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_7d"],
                                 name="7-day Avg", line=dict(color="#1565C0", width=2)))
        fig.update_layout(title="📈 Daily Revenue Trend", height=350,
                          yaxis_title="Revenue (₹)", template="plotly_white",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Category Revenue Share")
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat_rev, values="revenue", names="category",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.4)
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Inventory alerts summary
    st.markdown("### 🚨 Inventory Alert Summary")
    if inv_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Reorder Now",       inv_df["stockout_risk"].sum())
        c2.metric("🟡 Monitor",           (inv_df["risk_level"] == "🟡 MEDIUM - Monitor Closely").sum())
        c3.metric("🟠 Overstock",         inv_df["overstock_flag"].sum())
        c4.metric("🟢 Healthy",           (inv_df["risk_level"] == "🟢 HEALTHY").sum())


# -----------------------------------------------
# PAGE 2: Sales Analysis
# -----------------------------------------------
def page_sales_analysis(df):
    st.title("📈 Sales Analysis")

    if df is None:
        st.warning("Data not found. Run main.py first.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    stores    = ["All"] + sorted(df["store"].unique().tolist())
    cats      = ["All"] + sorted(df["category"].unique().tolist())
    products  = ["All"] + sorted(df["product"].unique().tolist())

    store_f   = col1.selectbox("Store",    stores)
    cat_f     = col2.selectbox("Category", cats)
    prod_f    = col3.selectbox("Product",  products)

    dff = df.copy()
    if store_f != "All": dff = dff[dff["store"] == store_f]
    if cat_f   != "All": dff = dff[dff["category"] == cat_f]
    if prod_f  != "All": dff = dff[dff["product"] == prod_f]

    # Sales trend
    daily = dff.groupby("date")["actual_sales"].sum().reset_index()
    daily["rolling_30d"] = daily["actual_sales"].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["actual_sales"],
                             name="Daily Sales", line=dict(color="#BBDEFB", width=1),
                             fill="tozeroy"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_30d"],
                             name="30-day Rolling Avg", line=dict(color="#1565C0", width=2.5)))
    fig.update_layout(title="Daily Sales Volume", height=400, template="plotly_white",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Monthly sales
    with col1:
        monthly = dff.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        m_sales = monthly.groupby("month")["actual_sales"].sum().reset_index()
        fig = px.bar(m_sales, x="month", y="actual_sales", title="Monthly Sales Volume",
                     color_discrete_sequence=["#1565C0"])
        fig.update_layout(height=350, template="plotly_white", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Day-of-week
    with col2:
        dff2 = dff.copy()
        dff2["dow"] = dff2["date"].dt.day_name()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_sales = dff2.groupby("dow")["actual_sales"].mean().reindex(day_order).reset_index()
        fig = px.bar(dow_sales, x="dow", y="actual_sales",
                     title="Avg Sales by Day of Week",
                     color="actual_sales", color_continuous_scale="Blues")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Store comparison
    st.subheader("🏪 Store Performance")
    store_perf = dff.groupby("store").agg(
        Revenue=("revenue","sum"), Units=("actual_sales","sum"),
        Stockouts=("stockout_flag","sum")
    ).reset_index()
    fig = px.bar(store_perf, x="store", y="Revenue", color="store",
                 title="Revenue by Store", text_auto=".2s",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=350, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------
# PAGE 3: Forecasting
# -----------------------------------------------
def page_forecasting(df, forecast_df, test_preds):
    st.title("🎯 Sales Forecasting")

    if forecast_df is None or test_preds is None:
        st.warning("Forecast data not found. Run main.py first.")
        return

    stores   = sorted(df["store"].unique().tolist())
    products = sorted(df["product"].unique().tolist())

    col1, col2 = st.columns(2)
    sel_store   = col1.selectbox("Select Store",   stores)
    sel_product = col2.selectbox("Select Product", products)

    # Historical + forecast
    hist = (
        df[(df["store"] == sel_store) & (df["product"] == sel_product)]
          .groupby("date")["actual_sales"].sum().reset_index()
          .tail(120)
    )
    fut = (
        forecast_df[(forecast_df["store"] == sel_store) & (forecast_df["product"] == sel_product)]
                   .sort_values("date")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual_sales"],
                             name="Historical Sales", line=dict(color="#1565C0", width=2)))
    if len(fut) > 0:
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast_qty"],
                                 name="30-Day Forecast", line=dict(color="#FF5722", width=2.5, dash="dash")))
        cutoff_date = pd.to_datetime(hist["date"].max())
        # ✅ FIX: Use add_shape instead of add_vline

        cutoff_date = hist["date"].max()

        fig.add_shape(
        type="line",
        x0=cutoff_date,
        x1=cutoff_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", dash="dot")
        )

        fig.add_annotation(
        x=cutoff_date,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="left"
        )

    fig.update_layout(title=f"📅 Forecast: {sel_product} @ {sel_store}",
                      height=450, template="plotly_white",
                      hovermode="x unified", yaxis_title="Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted (Test Period)")
    tp = test_preds.groupby("date").agg(
        actual=("actual_sales","sum"),
        predicted=("predicted_sales","sum"),
        lower=("pred_lower","sum"),
        upper=("pred_upper","sum")
    ).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tp["date"], y=tp["actual"],
                              name="Actual", line=dict(color="#1565C0", width=2)))
    fig2.add_trace(go.Scatter(x=tp["date"], y=tp["predicted"],
                              name="Predicted", line=dict(color="#FF5722", width=2, dash="dash")))
    fig2.add_trace(go.Scatter(x=pd.concat([tp["date"], tp["date"][::-1]]),
                              y=pd.concat([tp["upper"], tp["lower"][::-1]]),
                              fill="toself", fillcolor="rgba(255,87,34,0.1)",
                              line=dict(color="rgba(255,87,34,0)"),
                              name="95% Confidence Band"))
    fig2.update_layout(height=400, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast table
    if len(fut) > 0:
        st.subheader("📋 Forecast Table (Next 30 Days)")
        st.dataframe(fut[["date","product","forecast_qty","forecast_revenue"]].round(2),
                     use_container_width=True, height=300)


# -----------------------------------------------
# PAGE 4: Inventory Optimization
# -----------------------------------------------
def page_inventory(inv_df):
    st.title("📦 Inventory Optimization")

    if inv_df is None:
        st.warning("Inventory data not found. Run main.py first.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Reorder Alerts",  inv_df["stockout_risk"].sum())
    c2.metric("🟠 Overstock Items", inv_df["overstock_flag"].sum())
    c3.metric("📦 Avg Safety Stock", f"{inv_df['safety_stock'].mean():.0f} units")
    c4.metric("📐 Avg EOQ",          f"{inv_df['eoq'].mean():.0f} units")

    # Risk distribution
    st.subheader("🎯 Inventory Risk Distribution")
    risk_counts = inv_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    colors = {
        "🔴 HIGH - REORDER NOW":        "#F44336",
        "🟡 MEDIUM - Monitor Closely":  "#FF9800",
        "🟠 OVERSTOCK - Reduce Orders": "#FF5722",
        "🟢 HEALTHY":                   "#4CAF50"
    }
    fig = px.bar(risk_counts, x="Risk Level", y="Count",
                 color="Risk Level",
                 color_discrete_map=colors,
                 title="Inventory Risk Status")
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.subheader("🔍 Filter Inventory Table")
    col1, col2 = st.columns(2)
    risk_filter = col1.multiselect("Risk Level", inv_df["risk_level"].unique(),
                                   default=inv_df["risk_level"].unique())
    abc_filter  = col2.multiselect("ABC Class", inv_df["abc_class"].unique(),
                                   default=inv_df["abc_class"].unique())

    filtered = inv_df[inv_df["risk_level"].isin(risk_filter) &
                      inv_df["abc_class"].isin(abc_filter)]

    display_cols = ["store", "product", "category", "abc_class", "current_stock",
                    "safety_stock", "reorder_point", "eoq", "stockout_rate_%", "risk_level"]
    def highlight_risk(val):
        if "HIGH" in str(val):
            return "background-color: #ffebee; color: red"
        elif "MEDIUM" in str(val):
            return "background-color: #fff3cd"
        elif "OVERSTOCK" in str(val):
         return "background-color: #ffe0b2"
        elif "HEALTHY" in str(val):
            return "background-color: #e8f5e9"
    return ""

    styled_df = filtered[display_cols].style.apply(
    lambda col: col.map(highlight_risk) if col.name == "risk_level" else [""] * len(col)
    )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Download
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download Inventory Report", csv,
                       "inventory_report.csv", "text/csv")


# -----------------------------------------------
# PAGE 5: Model Performance
# -----------------------------------------------
def page_model_performance(comp_df, fi_df, test_preds):
    st.title("🏆 Model Performance")

    if comp_df is None:
        st.warning("Model comparison data not found. Run main.py first.")
        return

    st.subheader("📊 Model Comparison Table")
    st.dataframe(comp_df.round(4).style.highlight_min(subset=["MAE","RMSE"], color="#c8e6c9")
                                       .highlight_max(subset=["R2"], color="#c8e6c9"),
                 use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(comp_df, x="Model", y="MAE", color="Model",
                     title="MAE Comparison (Lower = Better)",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(comp_df, x="Model", y="R2", color="Model",
                     title="R² Score Comparison (Higher = Better)",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if fi_df is not None:
        st.subheader("🔍 Feature Importance (Best Model)")
        top_fi = fi_df.head(15)
        fig = px.bar(top_fi, x="importance", y="feature", orientation="h",
                     title="Top 15 Feature Importances",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(height=500, template="plotly_white", yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------
# PAGE 6: ABC Analysis
# -----------------------------------------------
def page_abc_analysis(abc_df):
    st.title("📊 ABC Analysis")

    if abc_df is None:
        st.warning("ABC data not found. Run main.py first.")
        return

    # Summary
    summary = abc_df.groupby("abc_class").agg(
        Products=("product","count"),
        Revenue=("total_revenue","sum"),
        Revenue_pct=("revenue_pct","sum")
    ).reset_index()

    c1, c2, c3 = st.columns(3)
    for i, row in summary.iterrows():
        col = [c1, c2, c3][i]
        label = {"A": "⭐ Class A (High Value)", "B": "🔵 Class B (Medium Value)",
                 "C": "⚪ Class C (Low Value)"}.get(row["abc_class"], row["abc_class"])
        col.metric(label, f"{row['Products']} products | {row['Revenue_pct']:.1f}% revenue")

    # Pareto chart
    st.subheader("📈 Pareto Chart (Revenue Distribution)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    abc_colors = {"A": "#4CAF50", "B": "#FF9800", "C": "#F44336"}
    bar_colors = [abc_colors.get(c, "gray") for c in abc_df["abc_class"]]

    fig.add_trace(go.Bar(x=abc_df["product"], y=abc_df["revenue_pct"],
                         name="Revenue %", marker_color=bar_colors, opacity=0.8))
    fig.add_trace(go.Scatter(x=abc_df["product"], y=abc_df["cumulative_pct"],
                             name="Cumulative %", line=dict(color="#212121", width=2),
                             mode="lines+markers"), secondary_y=True)

    fig.add_hline(y=70,  line_dash="dash", line_color="#4CAF50", annotation_text="A (70%)",
                  secondary_y=True)
    fig.add_hline(y=90,  line_dash="dash", line_color="#FF9800", annotation_text="B (90%)",
                  secondary_y=True)

    fig.update_layout(height=500, title="Product Revenue Pareto Chart",
                      template="plotly_white", xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.subheader("📋 Full ABC Table")
    def highlight_abc(val):
        if val == "A":
            return "background-color: #ffcccc"
        elif val == "B":
            return "background-color: #fff3cd"
        elif val == "C":
            return "background-color: #d4edda"
        return ""

    styled_df = abc_df.style.apply(
        lambda col: col.map(highlight_abc) if col.name == "abc_class" else [""] * len(col)
    )

    st.dataframe(styled_df, use_container_width=True)

# -----------------------------------------------
# MAIN APP
# -----------------------------------------------
def main():
    df, forecast_df, test_preds, inv_df, abc_df, comp_df, fi_df = load_data()
    page = sidebar()

    if page == "🏠 Overview":
        page_overview(df, inv_df, forecast_df)
    elif page == "📈 Sales Analysis":
        page_sales_analysis(df)
    elif page == "🎯 Forecasting":
        page_forecasting(df, forecast_df, test_preds)
    elif page == "📦 Inventory Optimization":
        page_inventory(inv_df)
    elif page == "🏆 Model Performance":
        page_model_performance(comp_df, fi_df, test_preds)
    elif page == "📊 ABC Analysis":
        page_abc_analysis(abc_df)


if __name__ == "__main__":
    main()
