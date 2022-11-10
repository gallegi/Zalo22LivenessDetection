import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

from configs.config_v3 import CFG


df = pd.read_csv('data/train/label.csv')


print('Number of videos', len(df))


FOLDER = 'data/train/videos'


frame_counts = []

for fname in tqdm(df.fname.tolist()):
    vid_name = fname
    vid_path = os.path.join(FOLDER, vid_name)
    cap = cv2.VideoCapture(vid_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0:
        print('vid', vid_name, 'corrupts')
    
    frame_counts.append(length)


df['frame_count'] = frame_counts


invalid_df = df[df.frame_count==0]
print('Number of invalid videos:', len(invalid_df))


valid_df = df[df.frame_count>0].reset_index(drop=True)
print('Number of valid videos:', len(valid_df))


plt.boxplot(valid_df.frame_count)
plt.title('Frame count distribution')
plt.show()


# The less frame a vid has, the more likely it is to be real
valid_df[valid_df.frame_count < 100].liveness_score.value_counts()


from sklearn.model_selection import StratifiedKFold


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)


fold = 0
for train_indices, val_indices in kfold.split(valid_df, valid_df['liveness_score'], valid_df['frame_count']):
    valid_df.loc[val_indices, 'fold'] = fold
    fold+=1


valid_df.groupby('fold').liveness_score.value_counts()


valid_df.to_csv('data/label_5folds.csv', index=False)


vid_names = []
frame_indices = []
for i, row in valid_df.iterrows():
    indices = np.arange(0, row['frame_count'], CFG.frame_sampling_rate)
    for ind in indices:
        vid_names.append(row['fname'])
        frame_indices.append(ind)


ind_df = pd.DataFrame({'fname': vid_names, 'frame_index': frame_indices})
ind_df = ind_df.merge(valid_df, on=['fname'])


ind_df.to_csv(f'data/label_sr{CFG.frame_sampling_rate}_frame_5folds.csv', index=False)


len(ind_df)





