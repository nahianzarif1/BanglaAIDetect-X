"""
load_custom_dataset.py

Load and prepare the custom Excel dataset for training.
"""

import pandas as pd
import os

def load_custom_dataset():
    """Load the custom Excel dataset and convert to project format."""
    
    # Load the Excel file
    excel_path = "data/raw/ieee_dataport_dataset.xlsx"
    df = pd.read_excel(excel_path)
    
    print(f"Loaded dataset with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSource distribution:")
    print(df['Source'].value_counts())
    
    # Map to our schema
    rows = []
    for idx, row in df.iterrows():
        text = row['source_text']
        source = row['Source']
        
        # Map source to label
        if source == 'AI':
            label = 'ai'
            generator = 'ai'
        else:  # 'ittefaq' or 'wiki' or any human source
            label = 'human'
            generator = 'human'
        
        # Create topic_id based on index (simplified)
        topic_id = f"topic_{idx % 100}"  # Group into 100 topics
        
        rows.append({
            'id': f"custom_{idx}",
            'text': text,
            'label': label,
            'generator': generator,
            'genre': 'mixed',
            'language_type': 'bangla',
            'edit_type': 'none',
            'topic_id': topic_id,
            'writer_subgroup': 'standard' if label == 'human' else None,
            'split': None,
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(rows)
    
    # Save to processed directory
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/collected_raw.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved converted dataset to {output_path}")
    print(f"✓ Total rows: {len(result_df)}")
    print(f"✓ AI samples: {len(result_df[result_df['label'] == 'ai'])}")
    print(f"✓ Human samples: {len(result_df[result_df['label'] == 'human'])}")
    
    return result_df

if __name__ == "__main__":
    load_custom_dataset()