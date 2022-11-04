import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import numpy as np
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(cfg):
    return A.Compose(
        [
            # A.ShiftScaleRotate(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),

            # # pixel aug
            # A.GaussNoise(p=0.2),
            # A.OneOf([
            #     A.MotionBlur(p=1),
            #     A.MedianBlur(blur_limit=5, p=1),
            #     A.Blur(blur_limit=5, p=1),
            # ], p=0.3),
            # A.OneOf([
            #     A.CLAHE(p=1),
            #     A.IAASharpen(p=1),
            #     A.IAAEmboss(p=1),
            #     A.RandomBrightnessContrast(p=1),
            # ], p=0.25),
            # A.HueSaturationValue(p=0.25),
            # A.ToGray(p=0.25),
         
            A.Resize(height=cfg.im_size, width=cfg.im_size, always_apply=True),
            # A.CoarseDropout(max_height=6, max_width=6, max_holes=3, p=0.2),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
            
        ],
        p=1.0,
        
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.im_size, width=cfg.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
       
    )

class LivenessDataset(Dataset):
    def __init__(self, cfg, df, transforms):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        vid_name = row['fname']
        vid_len = row['frame_count']
        vid_path = os.path.join(self.cfg.video_dir, vid_name)
        cap = cv2.VideoCapture(vid_path)
        frame_no = np.random.randint(0, vid_len)
        cap.set(1, frame_no)  # Where frame_no is the frame you want
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_ts = self.transforms(image=im)['image'].float()
        label = torch.tensor(row['liveness_score']).float()
        return im_ts, label