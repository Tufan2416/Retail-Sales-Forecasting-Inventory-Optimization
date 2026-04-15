# ============================================================
# main.py
# Master Pipeline Runner — Retail Sales Forecasting &
# Inventory Optimization System
#
# Run: python main.py
# ============================================================

import os
import sys
import time
import argparse

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import get_logger, ensure_dirs, load_config

logger = get_logger("main")


# ============================================================
# Banner
# ============================================================
def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║   🛒  RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION     ║
║         Industry-Level Analytics System  v1.0               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


# ============================================================
# Pipeline Steps
# ============================================================

def step_generate_data(config):
    logger.info("\n" + "━"*55)
    logger.info("  STEP 1: DATA GENERATION")
    logger.info("━"*55)
    from src.data_generator import generate_retail_dataset, print_dataset_summary
    sim = config["data"]["simulation"]
    df = generate_retail_dataset(
        start_date  = sim["start_date"],
        end_date    = sim["end_date"],
        seed        = sim["seed"],
        output_path = config["data"]["raw_path"]
    )
    print_dataset_summary(df)
    return df


def step_preprocess(config):
    logger.info("\n" + "━"*55)
    logger.info("  STEP 2: PREPROCESSING")
    logger.info("━"*55)
    from src.preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline(
        raw_path       = config["data"]["raw_path"],
        processed_path = config["data"]["processed_path"]
    )
    return df


def step_feature_engineering(config):
    logger.info("\n" + "━"*55)
    logger.info("  STEP 3: FEATURE ENGINEERING")
    logger.info("━"*55)
    from src.feature_engineering import run_feature_engineering
    df, enc_maps = run_feature_engineering(
        processed_path = config["data"]["processed_path"],
        features_path  = "data/features_data.csv"
    )
    return df, enc_maps


def step_forecasting(config):
    logger.info("\n" + "━"*55)
    logger.info("  STEP 4: FORECASTING")
    logger.info("━"*55)
    from src.forecasting import run_forecasting_pipeline
    forecast_df, comp_df, best_model, test_preds = run_forecasting_pipeline(
        features_path = "data/features_data.csv",
        forecast_path = config["data"]["forecast_output"]
    )
    return forecast_df, comp_df, best_model, test_preds


def step_inventory(config):
    logger.info("\n" + "━"*55)
    logger.info("  STEP 5: INVENTORY OPTIMIZATION")
    logger.info("━"*55)
    from src.inventory_optimization import run_inventory_pipeline
    inv = config["inventory"]
    inv_df, abc_df, kpis = run_inventory_pipeline(
        processed_path   = config["data"]["processed_path"],
        forecast_path    = config["data"]["forecast_output"],
        inventory_output = config["data"]["inventory_output"],
        abc_output       = config["data"]["abc_output"]
    )
    return inv_df, abc_df, kpis


def step_visualization():
    logger.info("\n" + "━"*55)
    logger.info("  STEP 6: VISUALIZATION")
    logger.info("━"*55)
    from src.visualization import run_visualization_pipeline
    run_visualization_pipeline()


# ============================================================
# Summary Report
# ============================================================
def print_final_summary(start_time: float):
    elapsed = time.time() - start_time
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  ✅ PIPELINE COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║  Time Elapsed    : {elapsed:.1f} seconds                          
║                                                              
║  📁 Generated Files:                                          
║    data/raw_sales_data.csv          ← Raw dataset           
║    data/processed_sales_data.csv    ← Clean data            
║    data/features_data.csv           ← Feature engineered    
║    outputs/forecast_results.csv     ← 30-day forecast       
║    outputs/test_predictions.csv     ← Model predictions     
║    outputs/inventory_recommendations.csv ← Inventory plan   
║    outputs/abc_analysis.csv         ← ABC classification    
║    outputs/model_comparison.csv     ← Model metrics         
║    outputs/feature_importance.csv   ← Feature importance    
║    models/rf_model.pkl              ← Saved RF model        
║    images/                          ← All charts (12 plots) 
║                                                              
║  🚀 Next Steps:                                              
║    1. Open images/ to view all charts                       
║    2. Run the dashboard:                                     
║       streamlit run app/streamlit_app.py                    
║    3. Open notebooks/ for EDA walkthrough                   
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Retail Sales Forecasting & Inventory Optimization"
    )
    parser.add_argument("--steps", nargs="+",
                        choices=["data", "preprocess", "features",
                                 "forecast", "inventory", "viz", "all"],
                        default=["all"],
                        help="Which pipeline steps to run (default: all)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config file")
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    print_banner()
    start_time = time.time()

    args   = parse_args()
    config = load_config(args.config)
    steps  = args.steps

    # Ensure all output directories exist
    ensure_dirs(["data", "models", "outputs", "images", "reports"])

    run_all = "all" in steps

    try:
        if run_all or "data" in steps:
            step_generate_data(config)

        if run_all or "preprocess" in steps:
            step_preprocess(config)

        if run_all or "features" in steps:
            step_feature_engineering(config)

        if run_all or "forecast" in steps:
            step_forecasting(config)

        if run_all or "inventory" in steps:
            step_inventory(config)

        if run_all or "viz" in steps:
            step_visualization()

        print_final_summary(start_time)

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed at step: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check that requirements are installed: pip install -r requirements.txt")
        print("  2. Make sure you're running from the project root directory")
        print("  3. Check error message above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
