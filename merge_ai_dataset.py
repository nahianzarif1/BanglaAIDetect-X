"""
merge_ai_dataset.py

Merge the new AI dataset with the existing dataset for better training.
"""

import pandas as pd
import os
import uuid

def merge_ai_dataset():
    """Merge the new AI dataset with the existing dataset."""
    
    # Load existing dataset
    existing_df = pd.read_csv('data/processed/dataset.csv')
    print(f"Existing dataset: {len(existing_df)} rows")
    print(f"Label distribution:\n{existing_df['label'].value_counts()}")
    
    # Load new AI dataset
    ai_df = pd.read_csv('data/raw/ai_dataset.csv')
    print(f"\nNew AI dataset: {len(ai_df)} rows")
    print(f"Columns: {ai_df.columns.tolist()}")
    
    # The 'output' column likely contains AI-generated text
    # Extract AI samples from the new dataset
    ai_samples = []
    for idx, row in ai_df.iterrows():
        ai_text = row['output']  # Using 'output' as AI text
        if pd.notna(ai_text) and len(str(ai_text).strip()) > 20:  # Minimum length check
            ai_samples.append({
                'id': f"ai_extra_{idx}",
                'text': str(ai_text).strip(),
                'label': 'ai',
                'generator': 'ai',  # Generic AI
                'genre': 'mixed',
                'language_type': 'bangla',
                'edit_type': 'none',
                'topic_id': f"ai_topic_{idx % 100}",  # Group into 100 topics
                'writer_subgroup': None,
                'split': None,
            })
    
    print(f"\nValid AI samples extracted: {len(ai_samples)}")
    
    # Create DataFrame for new AI samples
    new_ai_df = pd.DataFrame(ai_samples)
    
    # Combine with existing dataset
    # Remove existing AI samples to avoid duplication, keep only human
    human_df = existing_df[existing_df['label'] == 'human'].copy()
    
    # Combine human + new AI
    combined_df = pd.concat([human_df, new_ai_df], ignore_index=True)
    
    print(f"\nCombined dataset: {len(combined_df)} rows")
    print(f"Label distribution:\n{combined_df['label'].value_counts()}")
    
    # Save combined dataset
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/dataset_expanded.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved expanded dataset to {output_path}")
    print(f"✓ Total samples: {len(combined_df)}")
    print(f"✓ Human samples: {len(combined_df[combined_df['label'] == 'human'])}")
    print(f"✓ AI samples: {len(combined_df[combined_df['label'] == 'ai'])}")
    
    return combined_df

if __name__ == "__main__":
    merge_ai_dataset()