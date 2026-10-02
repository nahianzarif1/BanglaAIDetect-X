"""
create_block_dataset.py

Create a dataset with AI text blocks (multiple AI responses combined)
to improve detection of multi-line AI text.
"""

import pandas as pd
import random
import os

def create_block_dataset():
    """Create a dataset with AI text blocks."""
    
    # Load balanced dataset
    df = pd.read_csv('data/processed/dataset_balanced.csv')
    print(f"Original balanced dataset: {len(df)} rows")
    
    # Separate AI and human
    ai_df = df[df['label'] == 'ai'].copy()
    human_df = df[df['label'] == 'human'].copy()
    
    print(f"AI samples: {len(ai_df)}")
    print(f"Human samples: {len(human_df)}")
    
    # Create AI blocks (combine 3-10 AI responses randomly)
    ai_blocks = []
    for i in range(500):  # Create 500 AI blocks
        block_size = random.randint(3, 10)
        selected = ai_df.sample(n=block_size, random_state=i*42)
        combined_text = ' '.join(selected['text'].tolist())
        ai_blocks.append({
            'id': f"ai_block_{i}",
            'text': combined_text,
            'label': 'ai',
            'generator': 'ai',
            'genre': 'mixed',
            'language_type': 'bangla',
            'edit_type': 'none',
            'topic_id': f"ai_block_topic_{i % 50}",
            'writer_subgroup': None,
            'split': None,
        })
    
    # Create human blocks (combine 3-10 human responses randomly)
    human_blocks = []
    for i in range(500):  # Create 500 human blocks
        block_size = random.randint(3, 10)
        selected = human_df.sample(n=block_size, random_state=i*42)
        combined_text = ' '.join(selected['text'].tolist())
        human_blocks.append({
            'id': f"human_block_{i}",
            'text': combined_text,
            'label': 'human',
            'generator': 'human',
            'genre': 'mixed',
            'language_type': 'bangla',
            'edit_type': 'none',
            'topic_id': f"human_block_topic_{i % 50}",
            'writer_subgroup': None,
            'split': None,
        })
    
    # Combine original individual samples with block samples
    ai_blocks_df = pd.DataFrame(ai_blocks)
    human_blocks_df = pd.DataFrame(human_blocks)
    
    # Original individual samples
    original_ai = ai_df.copy()
    original_human = human_df.copy()
    
    # Combine: 50% individual, 50% blocks
    final_ai = pd.concat([original_ai, ai_blocks_df], ignore_index=True)
    final_human = pd.concat([original_human, human_blocks_df], ignore_index=True)
    
    # Balance the dataset
    min_count = min(len(final_ai), len(final_human))
    final_ai = final_ai.sample(n=min_count, random_state=42)
    final_human = final_human.sample(n=min_count, random_state=42)
    
    # Combine and shuffle
    final_df = pd.concat([final_ai, final_human], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal block-enhanced dataset: {len(final_df)} rows")
    print(f"Label distribution:\n{final_df['label'].value_counts()}")
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/dataset_block_enhanced.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved block-enhanced dataset to {output_path}")
    print(f"✓ Individual samples: {len(original_ai) + len(original_human)}")
    print(f"✓ Block samples: {len(ai_blocks) + len(human_blocks)}")
    
    return final_df

if __name__ == "__main__":
    create_block_dataset()