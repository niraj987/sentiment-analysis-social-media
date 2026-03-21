"""
run_all.py
───────────
Quick-start script: generates dataset, runs VADER analysis,
and launches the Streamlit dashboard.

Run: python run_all.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, cwd=None):
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or BASE_DIR)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("  SENTIMENT ANALYSIS PROJECT — Quick Start")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n[1/3] Generating synthetic review dataset...")
    data_path = os.path.join(BASE_DIR, 'data', 'raw', 'reviews.csv')
    if not os.path.exists(data_path):
        ok = run(f"python data/raw/generate_dataset.py")
        if not ok:
            print("[ERR] Failed to generate dataset.")
            sys.exit(1)
    else:
        print(f"  [OK] Dataset already exists: {data_path}")

    # Step 2: Run VADER analysis
    print("\n[2/3] Running VADER sentiment analysis...")
    vader_path = os.path.join(BASE_DIR, 'data', 'processed', 'vader_results.csv')
    if not os.path.exists(vader_path):
        ok = run("python src/vader_analysis.py")
        if not ok:
            print("[WARN] VADER analysis encountered an issue (non-critical).")
    else:
        print(f"  [OK] VADER results already exist: {vader_path}")

    # Step 3: Launch Streamlit dashboard
    print("\n[3/3] Launching Streamlit Dashboard...")
    print("  [WEB] Opening http://localhost:8501 in your browser...")
    run("streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
