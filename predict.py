import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from src.model import LivenessModel
from src.dataset import LivenessTestDataset

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config',
                    help='config file to run an experiment')
parser.add_argument('--test_video_dir', type=str, default='data/public/videos',
                    help='path to test folder')
parser.add_argument('--fold', type=int, default=0,
                    help='the fold of the trained weight')
parser.add_argument('--weight', type=str, default='models/v5_seg_head_regnet_y_16gf/fold0/best.pth',
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

if not torch.cuda.is_available():
    CFG.device = 'cpu'

# Load model
model = LivenessModel(CFG.backbone, CFG.pretrained_weights, CFG.embedding_size)
print('Delete auxilary heads for faster inference')
del model.metric_learning_head, model.decoder, model.seg_head
print(model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'], strict=False))
model.to(CFG.device)
model.eval()

# Choose frames at each vid to infer
fnames = os.listdir(CFG.test_video_dir)
test_df = pd.DataFrame(fnames)
test_df.columns = ['fname']

# Predict
test_preds = []
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    vid_path = os.path.join(CFG.test_video_dir, row['fname'])
    cap = cv2.VideoCapture(vid_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = length // CFG.frames_per_vid

    frame_idx = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_idx % stride == 0 and len(frames) < CFG.frames_per_vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = CFG.val_transforms(image=frame)['image']
                frames.append(frame)
            frame_idx += 1
        else:
            break

    cap.release()

    X = torch.stack(frames)
    with torch.no_grad():
        y_prob = model(X, aux_heads=False).sigmoid().view(-1).cpu().numpy()
        y_prob = y_prob.mean() # avg over multi frames
        test_preds.append(y_prob)

test_df['prob'] = test_preds

sub = test_df[['fname', 'prob']]
sub.columns = ['fname', 'liveness_score']

os.makedirs(CFG.submission_folder, exist_ok=True)
sub.to_csv(os.path.join(CFG.submission_folder, CFG.output_dir_name + f'_fold{args.fold}_{test_dir_name}.csv'), index=False)