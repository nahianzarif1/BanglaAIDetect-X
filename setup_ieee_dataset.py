"""
setup_ieee_dataset.py

Helper script to guide users through downloading and setting up the IEEE DataPort dataset.
This script does not automatically download the dataset (requires IEEE DataPort account),
but provides clear instructions and validates the setup.
"""

import os
import sys


def print_instructions():
    print("=" * 70)
    print("IEEE DataPort Dataset Setup Instructions")
    print("=" * 70)
    print()
    print("1. Download the dataset:")
    print("   - Visit: https://ieee-dataport.org/documents/bangla-ai-generated-and-human-written-text-dataset")
    print("   - Create a free IEEE DataPort account (no payment required)")
    print("   - Download the .xlsx file")
    print()
    print("2. Place the downloaded file:")
    print("   - Move the .xlsx file to: data/raw/ieee_dataport_dataset.xlsx")
    print()
    print("3. Run the data collection:")
    print("   - python -m src.data.collector")
    print()
    print("4. Continue with the normal pipeline:")
    print("   - python -m src.data.cleaner")
    print("   - python -m src.data.splitter")
    print("   - python -m src.data.validator")
    print()
    print("=" * 70)


def check_setup():
    """Check if the IEEE dataset file is in place."""
    excel_path = "data/raw/ieee_dataport_dataset.xlsx"
    
    if os.path.exists(excel_path):
        print(f"✓ IEEE DataPort dataset found at {excel_path}")
        print(f"  File size: {os.path.getsize(excel_path) / 1024:.2f} KB")
        return True
    else:
        print(f"✗ IEEE DataPort dataset NOT found at {excel_path}")
        print()
        print("Please follow the instructions above to download and place the dataset.")
        return False


def main():
    print_instructions()
    print()
    
    # Create the raw directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Check if dataset is already in place
    if check_setup():
        print()
        print("You're ready to run the data collection pipeline!")
        print("Run: python -m src.data.collector")
    else:
        print()
        print("Dataset not found. Please download it first.")


if __name__ == "__main__":
    main()