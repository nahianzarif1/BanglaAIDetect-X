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
    print("=" * 60)
    print("BanglaAIDetect-X Complete Pipeline Runner")
    print("=" * 60)

    with_banglabert = "--with-banglabert" in sys.argv

    ieee_path = "data/raw/ieee_dataport_dataset.xlsx"

    if not os.path.exists(ieee_path):
        print("\nIEEE DataPort dataset not found — using realistic dataset for better accuracy.")
        if not run_command("python create_realistic_dataset.py", "Create realistic dataset"):
            print("Failed to create dataset. Exiting.")
            return
    else:
        print(f"\n✓ Custom dataset found at {ieee_path}")
        if not run_command("python load_custom_dataset.py", "Load and convert custom dataset"):
            print("Custom dataset loading failed. Exiting.")
            return

    steps = [
        ("python -m src.data.cleaner", "Data cleaning and normalization"),
        ("python -m src.data.splitter", "Topic-based train/val/test split"),
        ("python -m src.data.validator", "Data validation"),
        ("python -m src.models.baseline", "Fusion TF-IDF + stylometry training"),
    ]

    for command, description in steps:
        if not run_command(command, description):
            print(f"{description} failed. Exiting.")
            return

    if with_banglabert:
        run_command("python -m src.models.banglabert", "BanglaBERT fine-tune (optional)")
    else:
        print("\nSkipping BanglaBERT (optional). Pass --with-banglabert if PyTorch is installed.")

    print("\n" + "=" * 60)
    print("Testing inference")
    print("=" * 60)
    if not run_command("python -m src.inference", "Inference test"):
        print("Inference test failed, but you can still inspect reports.")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    print("\n🚀 To run the Streamlit app:")
    print("streamlit run app/app.py")
    print("\n🧪 To run tests:")
    print("pytest tests/")
    print("\n📈 View results in:")
    print("- results/reports/fusion_metrics.json (Fusion model)")
    print("- results/reports/baseline_metrics.json (Baseline model)")


if __name__ == "__main__":
    main()