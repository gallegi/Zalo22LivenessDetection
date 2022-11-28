import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import numpy as np

from src.model import  LivenessModel
from src.metric import compute_eer

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config',
                    help='config file to run an experiment')
parser.add_argument('--fold', type=int, default=0,
                    help='fold to evaluate')
parser.add_argument('--weight', type=str, default='models/v5_seg_head_regnet_y_16gf/fold0/best.pth',
                    help='trained weight file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

if not torch.cuda.is_available():
    CFG.device = 'cpu'

# Load metadata
df = pd.read_csv(CFG.metadata_file)
df = df[df.set == 'train']

n_individuals = df.individual_id.nunique()

# Load model
model = LivenessModel(CFG.backbone, CFG.pretrained_weights, CFG.embedding_size)
print('Delete auxilary heads for faster inference')
del model.metric_learning_head, model.decoder, model.seg_head
print(model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'], strict=False))
model.to(CFG.device)
model.eval()

if CFG.sample is not None:
    df = df.sample(CFG.sample).reset_index(drop=True)

train_df = df[df.fold != args.fold]
val_df = df[df.fold == args.fold]

# Predict
val_preds = []
for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
    vid_path = os.path.join(CFG.train_video_dir, row['fname'])
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
    X = torch.stack(frames)
    with torch.no_grad():
        y_prob = model(X, aux_heads=False).sigmoid().view(-1).cpu().numpy()
        y_prob = y_prob.mean() # avg over multi frames
        val_preds.append(y_prob)

val_df.loc[:, 'prob'] = val_preds

val_df_grouped = val_df.groupby('fname').mean().reset_index()

y = val_df_grouped['liveness_score']
y_pred = val_df_grouped['prob']

os.makedirs(CFG.valid_pred_folder, exist_ok=True)
val_df_grouped.to_csv(os.path.join(CFG.valid_pred_folder, CFG.output_dir_name +  f'_valid_fold{args.fold}' + '.csv'), index=False)

eer_threshold, eer = compute_eer(y, y_pred)
print('Threshold at the intersection of FRR and FAR:', eer_threshold)
print(f'Equal Error Rate (EER) on valid fold {args.fold}:', eer)



