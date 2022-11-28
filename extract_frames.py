import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold

from configs.config import CFG


df = pd.read_csv('data/train/label.csv')

FOLDER = 'data/train/videos'
OUT_DIR = 'data/train_frames'

os.makedirs(OUT_DIR, exist_ok=True)

frame_counts = []

for fname in tqdm(df.fname.tolist()):
    vid_name = fname
    vid_path = os.path.join(FOLDER, vid_name)
    cap = cv2.VideoCapture(vid_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0:
        print('vid', vid_name, 'corrupts')
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            outpath = os.path.join(OUT_DIR, f'{vid_name}_{frame_index:03d}.jpg')
            cv2.imwrite(outpath, frame)
            frame_index += 1
        else:
            break
    cap.release()

    # break