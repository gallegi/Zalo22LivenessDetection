import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold

from configs.config_v1 import CFG


df = pd.read_csv('data/identified_metadata.csv')
df = df[df.set=='train']

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


vid_names = []
frame_indices = []
for i, row in valid_df.iterrows():
    # indices = np.arange(0, row['frame_count'], CFG.frame_sampling_rate)
    stride = row['frame_count'] // CFG.frames_per_vid
    for ind in range(row['frame_count']):
        if ind % stride == 0:
            vid_names.append(row['fname'])
            frame_indices.append(ind)

ind_df = pd.DataFrame({'fname': vid_names, 'frame_index': frame_indices})
ind_df = ind_df.merge(valid_df, on=['fname'])


ind_df.to_csv(f'data/label_{CFG.frames_per_vid}frames_10folds.csv', index=False)

