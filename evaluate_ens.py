import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import numpy as np

from src.model import  LivenessModel, LivenessSequenceModel
from src.metric import compute_eer

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--config_seq', type=str, default='config_seq',
                    help='config file for sequence model')
parser.add_argument('--fold', type=int, default=0,
                    help='fold to evaluate')
parser.add_argument('--weight', type=str, default='models/v1_baseline_tf_efficientnet_b0/fold0/epoch=3-val_loss=0.155-val_acc=0.950.ckpt',
                    help='trained weight file')
parser.add_argument('--weight_seq', type=str, default='models/v1_baseline_tf_efficientnet_b0/fold0/epoch=3-val_loss=0.155-val_acc=0.950.ckpt',
                    help='sequence model trained weight file')
parser.add_argument('--output_name', type=str, default='ensemble',
                    help='name for submission file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
config_module_seq = importlib.import_module(f'configs.{args.config_seq}')

CFG = config_module.CFG
CFG_SEQ = config_module_seq.CFG

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

if not torch.cuda.is_available():
    CFG.device = 'cpu'

# Load data
df = pd.read_csv(CFG.metadata_file)
df = df[df.set == 'train']
n_individuals = df.individual_id.nunique()

# Load image model
model = LivenessModel(CFG.backbone, CFG.pretrained_weights, CFG.embedding_size, n_individuals=n_individuals)
model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'])
model.to(CFG.device)
model.eval()

# Load sequence model
seq_model = LivenessSequenceModel(CFG_SEQ.backbone)
seq_model.load_state_dict(torch.load(args.weight_seq, map_location='cpu')['state_dict'])
seq_model.to(CFG.device)
seq_model.eval()

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

    cap.release()
    
    X = torch.stack(frames)
    with torch.no_grad():
        # model prediction
        y_prob = model(X, metric_learning_output=False).sigmoid().view(-1).cpu().numpy()
        y_prob = y_prob.mean() # avg over multi frames

        # sequence model prediction
        X = X.unsqueeze(0)
        y_prob_seq = seq_model(X).sigmoid().view(-1).cpu().item()

    y_prob_ens = (y_prob + y_prob_seq) / 2
    val_preds.append(y_prob_ens)

val_df.loc[:, 'prob'] = val_preds
val_df_grouped = val_df.groupby('fname').mean().reset_index()

y = val_df_grouped['liveness_score']
y_pred = val_df_grouped['prob']

os.makedirs(CFG.valid_pred_folder, exist_ok=True)
val_df_grouped.to_csv(os.path.join(CFG.valid_pred_folder, args.output_name + '.csv'), index=False)

eer_threshold, eer = compute_eer(y, y_pred)
print('Threshold at the intersection of FRR and FAR:', eer_threshold)
print(f'Equal Error Rate (EER) on valid fold {args.fold}:', eer)



