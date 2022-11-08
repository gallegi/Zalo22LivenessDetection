import os
import glob
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, CometLogger

from src.model import LivenessLit
from src.dataset import LivenessDataset

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--weights', type=str, default='models/v1_baseline_tf_efficientnet_b0',
                    help='trained weight file')
parser.add_argument('--submission_folder', type=str, default='./submissions',
                    help='trained weight file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

CFG.submission_folder = args.submission_folder

WEIGHTS = glob.glob(args.weights + '/*/*.ckpt')


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

# test_df = test_df[test_df.fname.isin(np.random.choice(test_df.fname.unique(), 10))]


test_ds = LivenessDataset(CFG, test_df, CFG.test_video_dir, CFG.val_transforms)

batch_size = CFG.batch_size
test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size,num_workers=CFG.num_workers,
                                            shuffle=False,pin_memory=True,drop_last=False)

trainer = pl.Trainer(default_root_dir=CFG.output_dir,  
                    # logger=logger,
                    accelerator=CFG.accelerator, devices=CFG.devices)

# Predict ensembling
test_df['prob'] = 0
for weight in WEIGHTS:
    # Load model
    print('Predict using weight:', weight)
    model = LivenessLit.load_from_checkpoint(weight, cfg=CFG)
    test_preds = trainer.predict(model, dataloaders=test_loader)
    test_preds = torch.cat(test_preds)
    test_preds = test_preds.cpu().numpy()
    test_df['prob'] += test_preds

test_df['prob'] /= len(WEIGHTS)

test_df_grouped = test_df.groupby('fname').mean().reset_index()

sub = test_df_grouped[['fname', 'prob']]
sub.columns = ['fname', 'liveness_score']

os.makedirs(CFG.submission_folder, exist_ok=True)
sub.to_csv(os.path.join(CFG.submission_folder, CFG.output_dir_name + f'_ensemble.csv'), index=False)