import pandas as pd

df = pd.read_csv('data/raw/ai_dataset.csv')
print('Sample AI texts:')
for i in range(3):
    print(f'\nSample {i+1}:')
    print(df.iloc[i]['output'][:200])