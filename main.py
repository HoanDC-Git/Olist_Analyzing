import sys
import os
import argparse

# Add src/ to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_cleaning import run_cleaning_pipeline
from feature_engineering import run_all_feature_engineering
from models import run_modeling_pipeline
from visualization import run_visualization_pipeline
from analysis_deep_dive import run_deep_dive_analysis
from nlp_pipeline import run_nlp_pipeline

def run_data_prep():
    print("--- Running Data Prep Pipeline ---")
    run_cleaning_pipeline()
    run_all_feature_engineering()
    run_deep_dive_analysis()

def run_nlp():
    print("--- Running NLP Pipeline ---")
    run_nlp_pipeline()

def run_forecast():
    print("--- Running Forecasting Pipeline ---")
    run_modeling_pipeline()

def run_visualization():
    print("--- Running Visualization Pipeline ---")
    run_visualization_pipeline()

def main():
    parser = argparse.ArgumentParser(description="Olist Data Analysis Pipeline")
    parser.add_argument("--step", type=str, choices=["data_prep", "nlp", "forecast", "viz", "all"], default="all",
                        help="Which pipeline step to run")
    
    args = parser.parse_args()

    print("==========================================================")
    print("      OLIST DATA ANALYSIS PIPELINE RUNNER")
    print(f"      MODE: {args.step.upper()}")
    print("==========================================================\n")
    
    if args.step in ["data_prep", "all"]:
        run_data_prep()
        
    if args.step in ["nlp", "all"]:
        run_nlp()
        
    if args.step in ["forecast", "all"]:
        run_forecast()
        
    if args.step in ["viz", "all"]:
        run_visualization()
    
    print("==========================================================")
    print("    PIPELINE EXECUTED SUCCESSFULLY - ALL ARTIFACTS READY")
    print("==========================================================\n")

if __name__ == "__main__":
    main()

