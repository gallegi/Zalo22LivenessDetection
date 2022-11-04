import pandas as pd

df = pd.read_csv('/Users/namnguyenthe/Workspace/AIChallenges/Zalo22/Liveness/submissions/v1_baseline.csv')

df['hard_score'] = df.liveness_score > 0.5
print(df['hard_score'].value_counts())