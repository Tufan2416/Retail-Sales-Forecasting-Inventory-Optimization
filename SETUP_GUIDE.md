# ⚡ Quick Setup Guide

## How to Run This Project in VS Code

---

### Step 1 — Open Project in VS Code

1. Extract the ZIP file
2. Open VS Code
3. File → Open Folder → Select `Retail-Sales-Forecasting-Inventory-Optimization`

---

### Step 2 — Create Virtual Environment

**Windows (Command Prompt or VS Code Terminal):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux (Terminal):**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly, streamlit, statsmodels, PyYAML, scipy, joblib

⏱️ Takes 2–5 minutes.

---

### Step 4 — Verify Setup

```bash
python verify_setup.py
```

All should show ✅. If any show ❌, re-run pip install.

---

### Step 5 — Run the Full Pipeline

```bash
python main.py
```

This runs ALL steps:
1. Generates synthetic retail dataset (~550k rows)
2. Preprocesses and cleans data
3. Engineers 20+ features
4. Trains Random Forest + XGBoost models
5. Generates 30-day forecasts
6. Computes Safety Stock, EOQ, ROP, ABC Analysis
7. Generates 12 visualization charts

⏱️ Takes 3–8 minutes depending on your machine.

---

### Step 6 — View Results

After running:
```
outputs/
  ├── forecast_results.csv         ← 30-day sales predictions
  ├── inventory_recommendations.csv ← Full inventory plan
  ├── abc_analysis.csv             ← Product ABC classes
  ├── model_comparison.csv         ← RF vs XGBoost vs Naive
  └── test_predictions.csv         ← Actual vs Predicted

images/                            ← 12 chart PNG files
models/                            ← Trained .pkl files
```

---

### Step 7 — Launch Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at: http://localhost:8501

Dashboard pages:
- 🏠 Overview (KPIs, revenue trend, category share)
- 📈 Sales Analysis (filterable trends, seasonality)
- 🎯 Forecasting (store/product selector, 30-day forecast)
- 📦 Inventory Optimization (alerts, risk table, download)
- 🏆 Model Performance (comparison, feature importance)
- 📊 ABC Analysis (Pareto chart, classification table)

---

### Step 8 — Open Notebook (Optional)

```bash
jupyter notebook notebooks/EDA_and_Walkthrough.ipynb
```

Step-by-step EDA with charts and explanations.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No such file: data/processed...` | Run `python main.py` first |
| `XGBoost not found` | Run `pip install xgboost` |
| `Streamlit not found` | Run `pip install streamlit` |
| Slow performance | Reduce `num_stores` or `num_products` in `config/config.yaml` |
| `venv\Scripts\activate` fails (Windows) | Run `Set-ExecutionPolicy Unrestricted` in PowerShell as Admin |

---

## Run Individual Steps

```bash
# Only generate data
python main.py --steps data

# Only preprocess
python main.py --steps preprocess

# Only train models
python main.py --steps forecast

# Data + Preprocess + Features only
python main.py --steps data preprocess features

# Just visualizations (after running full pipeline once)
python main.py --steps viz
```
