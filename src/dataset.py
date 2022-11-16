import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import numpy as np

class LivenessDataset(Dataset):
    def __init__(self, cfg, df, image_dir, transforms):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        vid_name = row['fname']
        frame_no = row['frame_index']

        img_path = os.path.join(self.image_dir, vid_name+f'_{frame_no:03d}.jpg')
        im = cv2.imread(img_path)
        im_ts = self.transforms(image=im)['image'].float()
        if 'liveness_score' in self.df.columns:
            label = torch.tensor(row['liveness_score']).float()
        else:
            label = -1
        return im_ts, label