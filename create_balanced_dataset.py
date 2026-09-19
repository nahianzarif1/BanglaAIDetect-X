"""
create_balanced_dataset.py

Create a balanced dataset from the expanded dataset for better training.
"""

import pandas as pd
import os

def create_balanced_dataset():
    """Create a balanced dataset from the expanded dataset."""
    
    # Load expanded dataset
    df = pd.read_csv('data/processed/dataset_expanded.csv')
    print(f"Expanded dataset: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Separate human and AI
    human_df = df[df['label'] == 'human'].copy()
    ai_df = df[df['label'] == 'ai'].copy()
    
    # Sample AI to match human count (undersampling)
    ai_sampled = ai_df.sample(n=len(human_df), random_state=42)
    
    # Combine
    balanced_df = pd.concat([human_df, ai_sampled], ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset: {len(balanced_df)} rows")
    print(f"Label distribution:\n{balanced_df['label'].value_counts()}")
    
    # Save balanced dataset
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/dataset_balanced.csv'
    balanced_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved balanced dataset to {output_path}")
    print(f"✓ Total samples: {len(balanced_df)}")
    print(f"✓ Human samples: {len(balanced_df[balanced_df['label'] == 'human'])}")
    print(f"✓ AI samples: {len(balanced_df[balanced_df['label'] == 'ai'])}")
    
    return balanced_df

if __name__ == "__main__":
    create_balanced_dataset()