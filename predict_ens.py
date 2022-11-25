import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from src.model import LivenessModel, LivenessSequenceModel

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--config_seq', type=str, default='config_seq',
                    help='config file for sequence model')
parser.add_argument('--test_video_dir', type=str, default='data/public/videos',
                    help='path to test folder')
parser.add_argument('--submission_folder', type=str, default='./submissions',
                    help='trained weight file')
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

CFG.submission_folder = args.submission_folder
CFG.test_video_dir = args.test_video_dir

test_dir_name = CFG.test_video_dir.split('/')[-2]
print('Predict on:', test_dir_name)

if not torch.cuda.is_available():
    CFG.device = 'cpu'

# Load model
model = LivenessModel(CFG.backbone, embedding_size=CFG.embedding_size)
del model.metric_learning_head # not necessary in prediction flow
model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'], strict=False)
model.to(CFG.device)
model.eval()

# Load sequence model
seq_model = LivenessSequenceModel(CFG_SEQ.backbone)
seq_model.load_state_dict(torch.load(args.weight_seq, map_location='cpu')['state_dict'])
seq_model.to(CFG.device)
seq_model.eval()


# Choose frames at each vid to infer
fnames = os.listdir(CFG.test_video_dir)
test_df = pd.DataFrame(fnames)
test_df.columns = ['fname']
test_df = test_df.sort_values('fname')

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
        # model prediction
        y_prob = model(X, metric_learning_output=False).sigmoid().view(-1).cpu().numpy()
        y_prob = y_prob.mean() # avg over multi frames

        # sequence model prediction
        X = X.unsqueeze(0)
        y_prob_seq = seq_model(X).sigmoid().view(-1).cpu().item()

    y_prob_ens = (y_prob + y_prob_seq) / 2
    test_preds.append(y_prob_ens)

test_df['prob'] = test_preds

test_df_grouped = test_df.groupby('fname').mean().reset_index()

sub = test_df_grouped[['fname', 'prob']]
sub.columns = ['fname', 'liveness_score']

os.makedirs(CFG.submission_folder, exist_ok=True)
sub.to_csv(os.path.join(CFG.submission_folder, args.output_name + '.csv'), index=False)