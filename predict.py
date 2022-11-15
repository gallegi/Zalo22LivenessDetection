import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, CometLogger

from src.model import LivenessLit, LivenessModel
from src.dataset import LivenessDataset

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--test_video_dir', type=str, default='data/public/videos',
                    help='path to test folder')
parser.add_argument('--fold', type=int, default=0,
                    help='the fold of the trained weight')
parser.add_argument('--weight', type=str, default='models/v1_baseline_tf_efficientnet_b0/fold0/epoch=3-val_loss=0.155-val_acc=0.950.ckpt',
                    help='trained weight file')
parser.add_argument('--submission_folder', type=str, default='./submissions',
                    help='trained weight file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

CFG.submission_folder = args.submission_folder
CFG.test_video_dir = args.test_video_dir

test_dir_name = CFG.test_video_dir.split('/')[-2]
print('Predict on:', test_dir_name)

# Load model
model = LivenessModel(CFG.backbone, backbone_pretrained=False)
model.load_state_dict(torch.load(args.weight, map_location='cpu'))
model.to(CFG.device)

# Choose frames at each vid to infer
fnames = os.listdir(CFG.test_video_dir)
test_df = pd.DataFrame(fnames)
test_df.columns = ['fname']

vid_names = []
frame_indices = []
for i, row in test_df.iterrows():
    # np.random.seed(CFG.seed)
    vid_path = os.path.join(CFG.test_video_dir, row['fname'])
    cap = cv2.VideoCapture(vid_path)

    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.random.choice(range(frame_counts), CFG.frames_per_vid)
    for ind in indices:
        vid_names.append(row['fname'])
        frame_indices.append(ind)

ind_df = pd.DataFrame({'fname': vid_names, 'frame_index': frame_indices})
test_df = ind_df.merge(test_df, on=['fname'])

test_ds = LivenessDataset(CFG, test_df, CFG.test_video_dir, CFG.val_transforms)

batch_size = CFG.batch_size
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,num_workers=CFG.num_workers,
                                            shuffle=False,pin_memory=True,drop_last=False)

# Predict
test_preds = []
for X,y in tqdm(test_loader, total=len(test_loader)):
    with torch.no_grad():
        y_prob = model(X).sigmoid().view(-1).cpu().numpy()
        test_preds.append(y_prob)
test_preds = np.concatenate(test_preds)

test_df['prob'] = test_preds

test_df_grouped = test_df.groupby('fname').mean().reset_index()

sub = test_df_grouped[['fname', 'prob']]
sub.columns = ['fname', 'liveness_score']

os.makedirs(CFG.submission_folder, exist_ok=True)
sub.to_csv(os.path.join(CFG.submission_folder, CFG.output_dir_name + f'_fold{args.fold}_{test_dir_name}.csv'), index=False)