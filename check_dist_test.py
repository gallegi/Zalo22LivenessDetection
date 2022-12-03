import pandas as pd
df = pd.read_csv('submissions/ensemble_v3.csv')

df['hard_score'] = df.liveness_score > 0.4
print(df['hard_score'].value_counts())