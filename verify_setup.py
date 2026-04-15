"""
verify_setup.py
Run this FIRST to check your Python environment is ready.
Usage: python verify_setup.py
"""

import sys
import importlib

print("=" * 55)
print("  SETUP VERIFICATION")
print("  Retail Sales Forecasting Project")
print("=" * 55)

# Check Python version
print(f"\n✅ Python version: {sys.version}")
if sys.version_info < (3, 9):
    print("⚠️  Warning: Python 3.9+ recommended")

# Required packages
packages = {
    "numpy":        "numpy",
    "pandas":       "pandas",
    "sklearn":      "scikit-learn",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "plotly":       "plotly",
    "streamlit":    "streamlit",
    "yaml":         "PyYAML",
    "joblib":       "joblib",
    "statsmodels":  "statsmodels",
    "scipy":        "scipy",
    "xgboost":      "xgboost",
}

print("\n📦 Checking packages:")
missing = []
for import_name, pip_name in packages.items():
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✅ {pip_name:<20} {ver}")
    except ImportError:
        print(f"  ❌ {pip_name:<20} NOT INSTALLED")
        missing.append(pip_name)

print("\n" + "=" * 55)
if missing:
    print(f"\n⚠️  Missing packages: {missing}")
    print("\nInstall them with:")
    print("  pip install " + " ".join(missing))
    print("  OR:")
    print("  pip install -r requirements.txt")
else:
    print("\n✅ All packages installed!")
    print("\n🚀 You're ready to run:")
    print("   python main.py")
    print("   streamlit run app/streamlit_app.py")
print("=" * 55)
