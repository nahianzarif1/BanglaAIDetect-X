"""
run_pipeline.py

Complete pipeline runner for BanglaAIDetect-X project.
This script runs the entire pipeline from data collection to model training.
"""

import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    print("="*60)
    print("BanglaAIDetect-X Complete Pipeline Runner")
    print("="*60)
    
    # Check if we have the IEEE dataset or need to use sample data
    ieee_path = "data/raw/ieee_dataport_dataset.xlsx"
    
    if not os.path.exists(ieee_path):
        print("\nIEEE DataPort dataset not found.")
        print("Creating sample dataset for testing...")
        if not run_command("python create_sample_dataset.py", "Create sample dataset"):
            print("Failed to create sample dataset. Exiting.")
            return
    else:
        print(f"\n✓ IEEE DataPort dataset found at {ieee_path}")
        if not run_command("python -m src.data.collector", "Data collection"):
            print("Data collection failed. Exiting.")
            return
    
    # Run the data processing pipeline
    steps = [
        ("python -m src.data.cleaner", "Data cleaning and normalization"),
        ("python -m src.data.splitter", "Data splitting (train/val/test)"),
        ("python -m src.data.validator", "Data validation"),
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            print(f"{description} failed. Exiting.")
            return
    
    # Train models
    model_steps = [
        ("python -m src.models.baseline", "TF-IDF baseline model training"),
        ("python -m src.models.banglabert", "BanglaBERT model training"),
    ]
    
    for command, description in model_steps:
        if not run_command(command, description):
            print(f"{description} failed. You can continue with other models or fix the issue.")
            # Don't exit, allow user to continue with other models
    
    # Test inference
    print("\n" + "="*60)
    print("Testing inference pipeline")
    print("="*60)
    if not run_command("python -m src.inference", "Inference test"):
        print("Inference test failed, but models may still work.")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)
    print("\nTo run the Streamlit app:")
    print("streamlit run app/app.py")
    print("\nTo run tests:")
    print("pytest tests/")


if __name__ == "__main__":
    main()