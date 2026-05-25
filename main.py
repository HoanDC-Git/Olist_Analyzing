import sys
import os

# Add src/ to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_cleaning import run_cleaning_pipeline
from feature_engineering import run_all_feature_engineering
from models import run_modeling_pipeline
from visualization import run_visualization_pipeline

def main():
    print("==========================================================")
    print("      OLIST DATA ANALYSIS PIPELINE RUNNER")
    print("==========================================================\n")
    
    # 1. Data Cleaning
    run_cleaning_pipeline()
    
    # 2. Feature Engineering & Business Analytics (RFM, Cohort)
    run_all_feature_engineering()
    
    # 3. Modeling & Cross-Validation
    run_modeling_pipeline()
    
    # 4. Visualization & Plotting
    run_visualization_pipeline()
    
    print("==========================================================")
    print("    PIPELINE EXECUTED SUCCESSFULLY - ALL ARTIFACTS READY")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
