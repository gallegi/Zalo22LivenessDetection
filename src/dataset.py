import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import numpy as np

class LivenessDataset(Dataset):
    def __init__(self, cfg, df, video_dir, transforms):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        vid_name = row['fname']
        
        vid_path = os.path.join(self.video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_no = np.random.choice(range(length))
        cap.set(1, frame_no)  # Where frame_no is the frame you want
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_ts = self.transforms(image=im)['image'].float()

        if 'liveness_score' in self.df.columns:
            label = torch.tensor(row['liveness_score']).float()
        else:
            label = -1

        if 'individual_id' in self.df.columns:
            indv_id = torch.tensor(row['individual_id']).long()
        else:
            indv_id = -1

        return im_ts, label, indv_id


class LivenessTestDataset(Dataset):
    def __init__(self, cfg, df, video_dir, transforms):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        vid_name = row['fname']
        vid_path = os.path.join(self.video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_no = row['frame_index']
        cap.set(1, frame_no)  # Where frame_no is the frame you want
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_ts = self.transforms(image=im)['image'].float()
        if 'liveness_score' in self.df.columns:
            label = torch.tensor(row['liveness_score']).float()
        else:
            label = -1
        return im_ts, label