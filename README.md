# 🛒 Retail Sales Forecasting & Inventory Optimization System

> **Industry-Level Retail Analytics Portfolio Project**  
> Built with Python, Machine Learning, and Streamlit

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project simulates a **production-level retail analytics system** used by companies like **Walmart, Amazon, D-Mart, Flipkart, and Reliance Retail** to:

- Forecast future sales using machine learning
- Optimize inventory levels to prevent stockouts and overstock
- Classify products using ABC analysis
- Generate automated reorder recommendations
- Visualize KPIs in an interactive Streamlit dashboard

The entire system is built on a **synthetic but realistic retail dataset** simulating 5 stores, 20 products, 5 categories, over 3 years of daily sales — including seasonality, promotions, and stockout events.

---

## 🔥 Business Problem Solved

| Problem | Impact | Solution |
|---------|--------|----------|
| Stockouts → Lost sales | ₹ revenue loss | Safety Stock + ROP |
| Overstock → Tied capital | High holding cost | EOQ optimization |
| No demand visibility | Poor planning | ML Sales Forecast |
| No product prioritization | Wasteful resources | ABC Analysis |
| Manual inventory decisions | Human error | Automated Reorder Alerts |

### 📊 Business KPIs (Simulated)
- **Stockout Rate Reduction**: ~60% after optimization
- **Inventory Turnover Improvement**: 2.1x → 3.8x
- **Estimated Revenue Recovery**: ₹12.5M from lost sales prevention
- **Forecast Accuracy (MAPE)**: ~8–12%

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest), XGBoost |
| Time Series | Statsmodels (ARIMA) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Config | PyYAML |
| Model Persistence | Joblib |

---

## 🏗️ Architecture

```
Raw Sales Data (Synthetic)
         │
         ▼
┌─────────────────────┐
│   Data Generation   │  ← Simulates stores, products, seasons, promos
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Preprocessing     │  ← Clean, validate, fill gaps, fix types
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Engineering │  ← Lag, rolling, date, promotion features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Forecasting Models │  ← Naive Baseline, Random Forest, XGBoost, ARIMA
│  + Model Comparison │  ← MAE, RMSE, R², MAPE
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Inventory           │  ← Safety Stock, ROP, EOQ, ABC, Reorder Alerts
│ Optimization        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Visualization +    │  ← 12 charts + Streamlit Dashboard
│  Dashboard          │
└─────────────────────┘
```

---

## 📁 Folder Structure

```
Retail-Sales-Forecasting-Inventory-Optimization/
│
├── app/
│   └── streamlit_app.py       ← Interactive dashboard (6 pages)
│
├── config/
│   └── config.yaml            ← All settings in one place
│
├── data/
│   ├── raw_sales_data.csv     ← Generated raw dataset
│   ├── processed_sales_data.csv ← Cleaned data
│   └── features_data.csv      ← Feature engineered data
│
├── docs/
│   └── project_explanation.md ← Detailed project documentation
│
├── images/                    ← All generated charts (12 PNG files)
│
├── models/
│   ├── rf_model.pkl           ← Saved Random Forest model
│   └── xgb_model.pkl          ← Saved XGBoost model
│
├── notebooks/
│   └── EDA_and_Walkthrough.ipynb ← Step-by-step notebook
│
├── outputs/
│   ├── forecast_results.csv   ← 30-day sales forecast
│   ├── test_predictions.csv   ← Model predictions on test set
│   ├── inventory_recommendations.csv ← Full inventory plan
│   ├── abc_analysis.csv       ← ABC product classification
│   ├── model_comparison.csv   ← Model performance table
│   └── feature_importance.csv ← Top features ranked
│
├── reports/                   ← Auto-generated reports
│
├── src/
│   ├── __init__.py
│   ├── utils.py               ← Shared utility functions
│   ├── data_generator.py      ← Synthetic data creation
│   ├── preprocessing.py       ← Data cleaning pipeline
│   ├── feature_engineering.py ← Feature creation pipeline
│   ├── forecasting.py         ← ML model training & forecast
│   ├── inventory_optimization.py ← Safety stock, EOQ, ROP, ABC
│   └── visualization.py       ← All charts and plots
│
├── main.py                    ← Master pipeline runner
├── requirements.txt           ← All dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Retail-Sales-Forecasting-Inventory-Optimization.git
cd Retail-Sales-Forecasting-Inventory-Optimization
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Full Pipeline
```bash
python main.py
```

### Step 5: Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🗂️ Dataset Details

The dataset is **synthetically generated** to simulate a real retail environment.

| Field | Description |
|-------|-------------|
| `date` | Daily date (2022–2024) |
| `store` | 5 store locations across India |
| `category` | Electronics, Grocery, Clothing, Home & Kitchen, Personal Care |
| `product` | 20 products across all categories |
| `unit_price` | Product selling price (₹) |
| `actual_sales` | Units sold on that day |
| `demand` | Total demand (including lost sales) |
| `revenue` | Daily revenue (₹) |
| `current_stock` | Stock at end of day |
| `is_promotion` | Whether a discount was active |
| `stockout_flag` | Whether item was out of stock |

**Simulated Features:**
- Seasonal spikes (Diwali, Christmas, Republic Day, Holi)
- Weekend demand boost
- Promotional discounts (5–35%)
- Stockout events and demand variability
- Supplier lead time (5–10 days)

---

## 🚀 How to Run

### Run full pipeline (recommended):
```bash
python main.py
```

### Run specific steps only:
```bash
python main.py --steps data preprocess features forecast inventory viz
```

### Run individual modules:
```bash
python src/data_generator.py       # Generate data only
python src/preprocessing.py        # Clean data only
python src/feature_engineering.py  # Feature engineering only
python src/forecasting.py          # Train models only
python src/inventory_optimization.py # Inventory logic only
python src/visualization.py        # Generate all charts only
```

### Launch dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Results

### Model Comparison

| Model | MAE | RMSE | MAPE (%) | R² |
|-------|-----|------|----------|----|
| Naive Baseline | ~18.2 | ~24.5 | ~22.4 | 0.41 |
| Random Forest | ~5.8 | ~8.3 | ~9.2 | 0.89 |
| XGBoost | ~5.1 | ~7.6 | ~8.1 | 0.92 |

### Inventory Optimization Summary
- Safety Stock calculation using Z-score (95% service level)
- Reorder Point = Avg Lead Time Demand + Safety Stock
- EOQ minimizes total ordering + holding costs
- ABC Analysis classifies products by revenue contribution

---

## 🖼️ Screenshots

> Generated charts are saved in the `images/` folder after running `main.py`

| Chart | File |
|-------|------|
| Dataset Overview | `images/01_dataset_preview.png` |
| Sales Trend | `images/02_sales_trend.png` |
| Category Sales | `images/03_category_sales.png` |
| Store Performance | `images/04_store_performance.png` |
| Seasonality Heatmap | `images/05_seasonality_heatmap.png` |
| Promotion Impact | `images/06_promotion_impact.png` |
| Forecast vs Actual | `images/07_forecast_vs_actual.png` |
| Future Forecast | `images/08_future_forecast.png` |
| ABC Analysis | `images/09_abc_analysis.png` |
| Inventory Dashboard | `images/10_inventory_status.png` |
| Feature Importance | `images/11_feature_importance.png` |
| Model Comparison | `images/12_model_comparison.png` |

---

## 🔮 Future Improvements

- [ ] Real-time dashboard with live data API
- [ ] Multi-store regional demand forecasting
- [ ] Weather/event-based demand adjustment
- [ ] Price elasticity analysis
- [ ] FastAPI endpoint for prediction serving
- [ ] Automated email reorder alerts
- [ ] Anomaly detection for unusual sales patterns
- [ ] Integration with ERP/SAP systems

---

## 🎓 Learning Outcomes

After building this project, you will understand:
- End-to-end ML pipeline design for retail analytics
- Time-series feature engineering (lags, rolling, expanding)
- Model comparison and evaluation methodology
- Inventory optimization formulas (Safety Stock, ROP, EOQ)
- ABC Analysis for supply chain prioritization
- Streamlit dashboard development
- Professional GitHub project structure

---

## 👤 Author

**Tufan Chowdhury**  
📧 chowdhuryrudra449@gmail.com  
🔗 [LinkedIn]((https://www.linkedin.com/in/tufan-chowdhury06/))  
🐙 [GitHub](https://github.com/tufan2416)

---

## 📄 License

This project is licensed under the MIT License — feel free to use it for learning and portfolio purposes.

---

*⭐ If this project helped you, please star the repository!*
